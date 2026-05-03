#!/usr/bin/env python3
"""Evaluation entry point for MS MARCO.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint experiments/checkpoints/phase2_final.pt \
        --index_path experiments/index
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.data.queries import Queries
from colbert.data.ranking import load_qrels, save_ranking
from colbert.evaluation.retriever import ColBERTRetriever
from colbert.evaluation.metrics import evaluate_ranking
from colbert.training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="ColBERTv2 Evaluation on MS MARCO")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Path to save ranking")
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

    queries = Queries(config.queries_dev)
    qrels = load_qrels(config.qrels_dev)

    retriever = ColBERTRetriever(model, args.index_path, config)

    query_list = queries.items()
    qid_list = [qid for qid, _ in query_list]
    query_texts = [text for _, text in query_list]

    raw_results = retriever.retrieve(query_texts, top_k=config.retrieve_top_k)

    ranking = {
        qid_list[q_idx]: raw_results[q_idx]
        for q_idx in raw_results
    }

    metrics = evaluate_ranking(ranking, qrels)

    logger.info("\nMS MARCO Dev Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    if args.output:
        save_ranking(ranking, args.output)
        logger.info(f"Ranking saved to {args.output}")


if __name__ == "__main__":
    main()
