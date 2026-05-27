"""Phase 2 training loop: KL-Divergence distillation + in-batch negatives."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.dataset.collection import Collection
from colbert.dataset.distillation import DistillationDataset, DistillationCollator
from colbert.training.loss import distillation_loss, in_batch_negative_loss
from colbert.training.utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    save_checkpoint,
    load_checkpoint,
    prune_step_checkpoints,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def train_phase2(
    config: ColBERTConfig,
    init_from: str | None = None,
    tuples_path: str | None = None,
    resume_from: str | None = None,
) -> None:
    """Run Phase 2 distillation training with DDP."""
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = config.resolved_torch_dtype
    use_scaler = amp_dtype == torch.float16

    logger.info("Loading model ...")
    model = ColBERT(config).to(device)

    if init_from and not resume_from:
        logger.info(f"Initializing from Phase 1 checkpoint: {init_from}")
        load_checkpoint(init_from, model)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process():
        base = model.module if hasattr(model, "module") else model
        total_params = sum(p.numel() for p in base.parameters())
        trainable_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        logger.info(
            f"Model: checkpoint='{config.checkpoint}', dim={config.dim}, "
            f"attn='{config.attn_implementation}', dtype={config.torch_dtype}"
        )
        logger.info(
            f"Parameters: {total_params:,} total, {trainable_params:,} trainable"
        )
        logger.info(f"Device: {device} | GradScaler: {'fp16' if use_scaler else 'disabled'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.distill_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.distill_warmup, config.distill_maxsteps
    )
    scaler = GradScaler("cuda") if use_scaler else None

    start_step = 0
    start_epoch = 0
    if resume_from:
        ckpt_info = load_checkpoint(resume_from, model, optimizer, scheduler, scaler)
        start_step = ckpt_info["step"]
        start_epoch = ckpt_info["epoch"]
        logger.info(f"Resuming from step {start_step}, epoch {start_epoch}")

    tuples_file = tuples_path or os.path.join(config.tuples_dir, "tuples.jsonl")
    logger.info(f"Loading collection from {config.collection} ...")
    collection = Collection(config.collection)
    logger.info(f"Loading distillation tuples from {tuples_file} ...")
    dataset = DistillationDataset(tuples_file, collection)
    collator = DistillationCollator(config)

    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=config.distill_bsize,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = max(1, len(loader) // max(1, config.accumsteps))
    logger.info(
        f"Data: {len(dataset):,} tuples, batch_size={config.distill_bsize}, "
        f"~{steps_per_epoch:,} steps/epoch, target={config.distill_maxsteps:,} total steps"
    )

    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(config.output_dir, "logs", "phase2"))
        except ImportError:
            writer = None

        if config.wandb_enabled:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity or None,
                name=config.wandb_run_name or "phase2",
                config={fld.name: getattr(config, fld.name) for fld in __import__("dataclasses").fields(config)},
                resume="allow" if resume_from else None,
            )
    else:
        writer = None

    model.train()
    step = start_step
    epoch = start_epoch
    micro_step = 0
    accumulated_loss = 0.0
    last_kl_val = 0.0
    last_ib_val = 0.0

    pbar = tqdm(
        total=config.distill_maxsteps, initial=start_step,
        desc="Phase 2", unit="step",
        disable=not is_main_process(),
    )

    while step < config.distill_maxsteps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= config.distill_maxsteps:
                break

            Q_ids = batch["Q_ids"].to(device)
            Q_mask = batch["Q_mask"].to(device)
            D_ids = batch["D_ids"].to(device)
            D_mask = batch["D_mask"].to(device)
            teacher_scores = batch["teacher_scores"].to(device)
            positive_idxs = batch["positive_idxs"].to(device)
            nway = batch["nway"]

            with autocast("cuda", dtype=amp_dtype):
                base = model.module if hasattr(model, "module") else model
                Q = base.query(Q_ids, Q_mask)  # (bsz, qlen, dim)

                D_all, D_all_mask = base.doc(D_ids, D_mask)  # (bsz*nway, dlen, dim)

                bsz = Q.size(0)
                dlen = D_all.size(1)
                dim = D_all.size(2)

                # Reshape D to (bsz, nway, dlen, dim)
                D_reshaped = D_all.view(bsz, nway, dlen, dim)
                D_mask_reshaped = D_all_mask.view(bsz, nway, dlen)

                # Compute scores for all nway passages per query
                student_scores = torch.zeros(bsz, nway, device=device)
                for w in range(nway):
                    student_scores[:, w] = base.score(
                        Q, D_reshaped[:, w], D_mask_reshaped[:, w]
                    )

                loss_kl = distillation_loss(student_scores, teacher_scores)

                # In-batch negatives: use positive passages only
                D_pos_list = []
                D_pos_mask_list = []
                for i in range(bsz):
                    pidx = positive_idxs[i].item()
                    D_pos_list.append(D_reshaped[i, pidx])
                    D_pos_mask_list.append(D_mask_reshaped[i, pidx])

                D_pos = torch.stack(D_pos_list)  # (bsz, dlen, dim)
                D_pos_mask = torch.stack(D_pos_mask_list)  # (bsz, dlen)

                loss_ib = in_batch_negative_loss(Q, D_pos, D_pos_mask)

                loss = (loss_kl + loss_ib) / config.accumsteps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()
            last_kl_val = loss_kl.item()
            last_ib_val = loss_ib.item()
            micro_step += 1

            if micro_step % config.accumsteps != 0:
                continue

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            total_loss = accumulated_loss
            accumulated_loss = 0.0
            pbar.update(1)
            pbar.set_postfix(loss=f"{total_loss:.4f}", epoch=epoch)

            if is_main_process() and step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{config.distill_maxsteps} | "
                    f"Loss: {total_loss:.4f} (KL: {last_kl_val:.4f}, IB: {last_ib_val:.4f}) | "
                    f"LR: {lr:.2e}"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", total_loss, step)
                    writer.add_scalar("train/loss_kl", last_kl_val, step)
                    writer.add_scalar("train/loss_ib", last_ib_val, step)
                    writer.add_scalar("train/lr", lr, step)
                if config.wandb_enabled:
                    import wandb
                    wandb.log({
                        "train/loss": total_loss,
                        "train/loss_kl": last_kl_val,
                        "train/loss_ib": last_ib_val,
                        "train/lr": lr,
                    }, step=step)

            if step % config.save_every == 0:
                ckpt_path = Path(config.checkpoint_dir) / f"phase2_step{step}.pt"
                save_checkpoint(
                    model, optimizer, step, ckpt_path,
                    scheduler=scheduler, scaler=scaler, epoch=epoch,
                )
                prune_step_checkpoints(
                    config.checkpoint_dir, "phase2", config.save_total_limit,
                )

        epoch += 1

    pbar.close()

    final_path = Path(config.checkpoint_dir) / "phase2_final.pt"
    save_checkpoint(
        model, optimizer, step, final_path,
        scheduler=scheduler, scaler=scaler, epoch=epoch,
    )

    if writer is not None:
        writer.close()
    if is_main_process() and config.wandb_enabled:
        import wandb
        wandb.finish()

    cleanup_distributed()
    if is_main_process():
        logger.info(f"Phase 2 training complete. Final checkpoint: {final_path}")
