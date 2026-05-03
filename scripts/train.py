#!/usr/bin/env python3
"""Training entry point for ColBERTv2.

Usage:
    Phase 1 (triples):
        torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml

    Phase 2 (distillation):
        torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml \
            --mode distill --init_from experiments/checkpoints/phase1_final.pt \
            --tuples data/tuples/tuples.jsonl
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colbert.config import ColBERTConfig
from colbert.training.trainer import train_phase1
from colbert.training.distill_trainer import train_phase2


def main():
    parser = argparse.ArgumentParser(description="ColBERTv2 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", choices=["triples", "distill"], default="triples")
    parser.add_argument("--init_from", type=str, default=None, help="Checkpoint to initialize from")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--tuples", type=str, default=None, help="Distillation tuples JSONL path")
    parser.add_argument(
        "--dtype", type=str, default=None,
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Override torch_dtype for mixed-precision training (default: from config)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = ColBERTConfig.from_yaml(args.config)

    if args.dtype:
        config = config.override(torch_dtype=args.dtype)

    if args.wandb:
        config = config.override(wandb_enabled=True)
    if args.wandb_project:
        config = config.override(wandb_project=args.wandb_project)
    if args.wandb_entity:
        config = config.override(wandb_entity=args.wandb_entity)
    if args.wandb_run_name:
        config = config.override(wandb_run_name=args.wandb_run_name)

    if args.mode == "triples":
        train_phase1(config, resume_from=args.resume_from)
    elif args.mode == "distill":
        train_phase2(
            config,
            init_from=args.init_from,
            tuples_path=args.tuples,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
