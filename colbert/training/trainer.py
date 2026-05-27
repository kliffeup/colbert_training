"""Phase 1 training loop: pairwise cross-entropy with triples."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.dataset.collection import Collection
from colbert.dataset.triples import StreamingTriplesDataset, TriplesCollator
from colbert.training.loss import pairwise_ce_loss
from colbert.training.utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    save_checkpoint,
    load_checkpoint,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def _log_model_info(model: torch.nn.Module, config: ColBERTConfig, device: torch.device) -> None:
    """Log model architecture summary."""
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
    logger.info(f"Device: {device} | GradScaler: {'fp16' if config.resolved_torch_dtype == torch.float16 else 'disabled'}")


def _estimate_steps_per_epoch(dataset: StreamingTriplesDataset, config: ColBERTConfig) -> int:
    """Estimate optimizer steps per epoch from line count, world size, batch size, and accumsteps."""
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    lines_per_rank = dataset.num_lines // world_size
    micro_batches = max(1, lines_per_rank // config.bsize)
    return max(1, micro_batches // max(1, config.accumsteps))


def train_phase1(config: ColBERTConfig, resume_from: str | None = None) -> None:
    """Run Phase 1 triple-based training with DDP."""
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = config.resolved_torch_dtype
    use_scaler = amp_dtype == torch.float16

    logger.info("Loading model ...")
    model = ColBERT(config).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process():
        _log_model_info(model, config, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup, config.maxsteps)
    scaler = GradScaler("cuda") if use_scaler else None

    start_step = 0
    start_epoch = 0
    if resume_from:
        ckpt_info = load_checkpoint(resume_from, model, optimizer, scheduler, scaler)
        start_step = ckpt_info["step"]
        start_epoch = ckpt_info["epoch"]
        logger.info(f"Resuming from step {start_step}, epoch {start_epoch}")

    logger.info(f"Loading training data from {config.triples} ...")
    collection = None
    if config.collection and Path(config.collection).exists():
        logger.info(f"Indexing collection at {config.collection} for docid resolution ...")
        collection = Collection(config.collection)
    dataset = StreamingTriplesDataset(config.triples, collection=collection)
    collator = TriplesCollator(config)

    loader = DataLoader(
        dataset,
        batch_size=config.bsize,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = _estimate_steps_per_epoch(dataset, config)
    logger.info(
        f"Data: ~{dataset.num_lines:,} triples (streaming), batch_size={config.bsize}, "
        f"~{steps_per_epoch:,} steps/epoch, target={config.maxsteps:,} total steps"
    )

    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(config.output_dir, "logs", "phase1"))
        except ImportError:
            writer = None
            logger.warning("TensorBoard not available, skipping logging")

        if config.wandb_enabled:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity or None,
                name=config.wandb_run_name or "phase1",
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

    pbar = tqdm(
        total=config.maxsteps, initial=start_step,
        desc="Phase 1", unit="step",
        disable=not is_main_process(),
    )

    while step < config.maxsteps:
        dataset.set_epoch(epoch)

        for batch in loader:
            if step >= config.maxsteps:
                break

            Q_ids = batch["Q_ids"].to(device)
            Q_mask = batch["Q_mask"].to(device)
            D_pos_ids = batch["D_pos_ids"].to(device)
            D_pos_mask = batch["D_pos_mask"].to(device)
            D_neg_ids = batch["D_neg_ids"].to(device)
            D_neg_mask = batch["D_neg_mask"].to(device)

            with autocast("cuda", dtype=amp_dtype):
                base = model.module if hasattr(model, "module") else model
                Q = base.query(Q_ids, Q_mask)
                D_pos, D_pos_doc_mask = base.doc(D_pos_ids, D_pos_mask)
                D_neg, D_neg_doc_mask = base.doc(D_neg_ids, D_neg_mask)

                scores_pos = base.score(Q, D_pos, D_pos_doc_mask)
                scores_neg = base.score(Q, D_neg, D_neg_doc_mask)

                loss = pairwise_ce_loss(scores_pos, scores_neg)
                loss = loss / config.accumsteps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()
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
            train_loss = accumulated_loss
            accumulated_loss = 0.0
            pbar.update(1)
            pbar.set_postfix(loss=f"{train_loss:.4f}", epoch=epoch)

            if is_main_process() and step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {step}/{config.maxsteps} | Loss: {train_loss:.4f} | LR: {lr:.2e}")
                if writer is not None:
                    writer.add_scalar("train/loss", train_loss, step)
                    writer.add_scalar("train/lr", lr, step)
                if config.wandb_enabled:
                    import wandb
                    wandb.log({"train/loss": train_loss, "train/lr": lr}, step=step)

            if step % config.save_every == 0:
                ckpt_path = Path(config.checkpoint_dir) / f"phase1_step{step}.pt"
                save_checkpoint(
                    model, optimizer, step, ckpt_path,
                    scheduler=scheduler, scaler=scaler, epoch=epoch,
                )

        epoch += 1

    pbar.close()

    final_path = Path(config.checkpoint_dir) / "phase1_final.pt"
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
        logger.info(f"Phase 1 training complete. Final checkpoint: {final_path}")
