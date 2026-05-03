#!/usr/bin/env python3
"""Indexing entry point for ColBERTv2.

No torchrun / DDP required — the master process spawns encoding workers
on all available GPUs automatically.

Usage:
    python scripts/index.py --config configs/default.yaml \
        --checkpoint experiments/checkpoints/phase2_final.pt

    # Limit to 2 GPUs:
    python scripts/index.py --config configs/default.yaml \
        --checkpoint experiments/checkpoints/phase2_final.pt --num_gpus 2
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.data.collection import Collection
from colbert.indexing.index_builder import build_index
from colbert.training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="ColBERTv2 Indexing")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--doc_maxlen", type=int, default=None)
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs for encoding (default: all visible CUDA devices)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = ColBERTConfig.from_yaml(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ColBERT(config).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    collection_path = args.collection or config.collection
    collection = Collection(collection_path)

    index_path = args.index_path or config.index_path

    build_index(
        model, collection, config,
        index_path=index_path,
        batch_size=args.batch_size,
        doc_maxlen=args.doc_maxlen,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
