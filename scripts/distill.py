#!/usr/bin/env python3
"""Distillation preparation: retrieve + cross-encoder scoring + tuple building.

Usage:
    python scripts/distill.py --config configs/default.yaml \
        --checkpoint experiments/checkpoints/phase1_final.pt
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
from colbert.data.queries import Queries
from colbert.data.ranking import load_qrels
from colbert.indexing.index_builder import build_index
from colbert.evaluation.retriever import ColBERTRetriever
from colbert.distillation.score_passages import score_passages
from colbert.distillation.build_tuples import build_tuples


def main():
    parser = argparse.ArgumentParser(description="ColBERTv2 Distillation Preparation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Output tuples path")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    config = ColBERTConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {args.checkpoint}")
    model = ColBERT(config).to(device)
    from colbert.training.utils import load_checkpoint
    load_checkpoint(args.checkpoint, model)

    collection = Collection(config.collection)
    queries = Queries(config.queries_train)

    # Step 1: Index the training passages
    logger.info("Step 1: Building index for training passages")
    index_dir = str(Path(config.output_dir) / "distill_index")
    build_index(model, collection, config, index_path=index_dir, batch_size=args.batch_size)

    # Step 2: Retrieve top-k passages per training query
    logger.info("Step 2: Retrieving passages for training queries")
    retriever = ColBERTRetriever(model, index_dir, config)

    query_list = queries.items()
    qid_list = [qid for qid, _ in query_list]
    query_texts = [text for _, text in query_list]

    raw_results = retriever.retrieve(query_texts, top_k=config.top_k_distill)

    ranking = {
        qid_list[q_idx]: raw_results[q_idx]
        for q_idx in raw_results
    }

    # Step 3: Score with cross-encoder
    logger.info("Step 3: Scoring with cross-encoder")
    ce_scores = score_passages(config, ranking, queries, collection)

    # Step 4: Build distillation tuples
    logger.info("Step 4: Building distillation tuples")
    output_path = args.output or str(Path(config.tuples_dir) / "tuples.jsonl")
    build_tuples(
        config,
        ce_scores,
        queries,
        qrels_path=config.qrels_train,
        output_path=output_path,
    )

    logger.info(f"Distillation tuples written to {output_path}")


if __name__ == "__main__":
    main()
