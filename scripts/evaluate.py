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
import torch.multiprocessing as mp

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.dataset.queries import Queries
from colbert.dataset.ranking import load_qrels, save_ranking
from colbert.evaluation.retriever import ColBERTRetriever
from colbert.evaluation.metrics import evaluate_ranking
from colbert.evaluation.doc_evaluator import retrieve_documents
from colbert.training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="ColBERTv2 Evaluation on MS MARCO")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Path to save ranking")
    parser.add_argument(
        "--num_scorer_gpus", type=int, default=None,
        help="GPUs to fan MaxSim scoring across (default: all visible CUDA devices). "
             "1 forces the in-process single-GPU path.",
    )
    args = parser.parse_args()

    # Scoring workers are spawned (not forked) so they can each own a CUDA context;
    # must be set before CUDA is initialized on the master. file_system sharing avoids
    # FD exhaustion when many small tensors cross process boundaries.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    mp.set_sharing_strategy("file_system")

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

    retriever = ColBERTRetriever(
        model, args.index_path, config, num_scorer_gpus=args.num_scorer_gpus
    )

    try:
        query_list = queries.items()
        qid_list = [qid for qid, _ in query_list]
        query_texts = [text for _, text in query_list]

        is_maxp = config.task == "document" and config.doc_segmentation == "maxp"
        if is_maxp:
            logger.info(
                "Document MaxP evaluation: retrieving passages and aggregating to docs."
            )
            raw_results = retrieve_documents(retriever, query_texts, config)
        else:
            raw_results = retriever.retrieve(query_texts, top_k=config.retrieve_top_k)
    finally:
        retriever.close()

    ranking = {
        qid_list[q_idx]: raw_results[q_idx]
        for q_idx in raw_results
    }

    metrics = evaluate_ranking(ranking, qrels)

    label = "MS MARCO Doc" if config.task == "document" else "MS MARCO"
    logger.info(f"\n{label} Dev Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    if args.output:
        save_ranking(ranking, args.output)
        logger.info(f"Ranking saved to {args.output}")


if __name__ == "__main__":
    main()
