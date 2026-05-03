"""LoTTE benchmark zero-shot evaluation orchestrator."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.ranking import load_qrels
from colbert.indexing.index_builder import build_index
from colbert.evaluation.retriever import ColBERTRetriever
from colbert.evaluation.metrics import success_at_k, evaluate_ranking

logger = logging.getLogger(__name__)

LOTTE_TOPICS = ["writing", "recreation", "science", "technology", "lifestyle", "pooled"]
LOTTE_QUERY_TYPES = ["search", "forum"]


def _get_lotte_paths(
    topic: str, query_type: str, lotte_data_dir: str, split: str = "test"
) -> Tuple[str, str, str]:
    """Get paths for LoTTE topic/query_type/split.

    Returns (collection_path, queries_path, qrels_path).
    """
    base = Path(lotte_data_dir) / topic / split

    collection_path = base / "collection.tsv"
    queries_path = base / f"questions.{query_type}.tsv"
    qrels_path = base / f"qrels.{query_type}.tsv"

    return str(collection_path), str(queries_path), str(qrels_path)


def evaluate_lotte(
    model: ColBERT,
    config: ColBERTConfig,
    topics: List[str] | None = None,
    query_types: List[str] | None = None,
    split: str = "test",
    batch_size: int = 128,
) -> Dict[str, Dict[str, float]]:
    """Run zero-shot evaluation on LoTTE benchmark.

    Args:
        model: Trained ColBERT model.
        config: Configuration.
        topics: LoTTE topics to evaluate (default: all).
        query_types: Query types to evaluate (default: search + forum).
        split: Dataset split (test or dev).
        batch_size: Encoding batch size.

    Returns:
        Dict mapping "topic/query_type" -> {metric_name: value}.
    """
    topics = topics or LOTTE_TOPICS
    query_types = query_types or LOTTE_QUERY_TYPES
    results: Dict[str, Dict[str, float]] = {}

    eval_config = config.override(doc_maxlen=config.lotte_doc_maxlen)

    # Index each topic's corpus once, evaluate on both query types
    for topic in topics:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"LoTTE topic: {topic}")
        logger.info(f"{'=' * 60}")

        # Check if corpus exists
        collection_path = Path(config.lotte_data_dir) / topic / split / "collection.tsv"
        if not collection_path.exists():
            logger.warning(f"LoTTE corpus not found: {collection_path}, skipping")
            continue

        collection = Collection(str(collection_path))

        # Build index for this topic
        index_dir = os.path.join(config.output_dir, "lotte_indices", topic, split)
        build_index(
            model, collection, eval_config,
            index_path=index_dir,
            batch_size=batch_size,
            doc_maxlen=config.lotte_doc_maxlen,
        )

        retriever = ColBERTRetriever(model, index_dir, eval_config)

        for query_type in query_types:
            key = f"{topic}/{query_type}"
            logger.info(f"Evaluating: {key}")

            _, queries_path, qrels_path = _get_lotte_paths(
                topic, query_type, config.lotte_data_dir, split
            )

            if not Path(queries_path).exists():
                logger.warning(f"Queries not found: {queries_path}, skipping")
                continue
            if not Path(qrels_path).exists():
                logger.warning(f"Qrels not found: {qrels_path}, skipping")
                continue

            queries = Queries(queries_path)
            qrels = load_qrels(qrels_path)

            query_list = queries.items()
            qid_list = [qid for qid, _ in query_list]
            query_texts = [text for _, text in query_list]

            raw_results = retriever.retrieve(query_texts, top_k=5)

            ranking = {
                qid_list[q_idx]: raw_results[q_idx]
                for q_idx in raw_results
            }

            s_at_5 = success_at_k(ranking, qrels, k=5)
            dataset_metrics = {"Success@5": s_at_5}
            results[key] = dataset_metrics

            logger.info(f"  Success@5: {s_at_5:.4f}")

    # Report summary
    if results:
        for query_type in query_types:
            qt_results = {k: v for k, v in results.items() if k.endswith(f"/{query_type}")}
            if qt_results:
                avg = sum(r["Success@5"] for r in qt_results.values()) / len(qt_results)
                logger.info(f"\nAverage Success@5 ({query_type}): {avg:.4f}")

    return results
