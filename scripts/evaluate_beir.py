#!/usr/bin/env python3
"""Zero-shot evaluation on BEIR benchmark.

Usage:
    python scripts/evaluate_beir.py --config configs/default.yaml \
        --checkpoint experiments/checkpoints/phase2_final.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.evaluation.beir_evaluator import evaluate_beir, BEIR_DATASETS
from colbert.training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="ColBERTv2 BEIR Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", nargs="*", default=None, help="BEIR datasets to evaluate")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = ColBERTConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ColBERT(config).to(device)
    load_checkpoint(args.checkpoint, model)

    datasets = args.datasets or BEIR_DATASETS

    results = evaluate_beir(model, config, datasets=datasets, batch_size=args.batch_size)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
