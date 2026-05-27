"""Training utilities: checkpointing, logging, distributed helpers."""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> int:
    """Initialize distributed process group. Returns local rank."""
    if "RANK" not in os.environ:
        logger.info("Not running in distributed mode")
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str | Path,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 0,
    config: Any = None,
) -> None:
    """Save full training checkpoint (only on rank 0).

    Saves model, optimizer, scheduler, scaler, step, and epoch for complete resume.
    """
    if not is_main_process():
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint at step {step} (epoch {epoch}) to {path}")


_STEP_CKPT_RE = re.compile(r"_step(\d+)\.pt$")


def prune_step_checkpoints(checkpoint_dir: str | Path, prefix: str, keep: int) -> None:
    """Delete oldest `prefix_step{N}.pt` files, keeping only the `keep` most recent.

    No-op when `keep < 0`. The `prefix_final.pt` companion is never matched (no `_step`),
    so it's never pruned. Only rank 0 acts.
    """
    if keep < 0 or not is_main_process():
        return
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return
    candidates = []
    for p in d.iterdir():
        if not p.is_file() or not p.name.startswith(f"{prefix}_step"):
            continue
        m = _STEP_CKPT_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    candidates.sort(key=lambda x: x[0])
    for _, path in candidates[:-keep] if keep > 0 else candidates:
        try:
            path.unlink()
            logger.info(f"Pruned old checkpoint: {path}")
        except OSError as e:
            logger.warning(f"Could not delete {path}: {e}")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """Load checkpoint. Returns dict with 'step' and 'epoch'.

    Restores model, optimizer, scheduler, and scaler states if provided.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Loaded checkpoint from {path} at step {step}, epoch {epoch}")
    return {"step": step, "epoch": epoch}


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then linear decay schedule."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
