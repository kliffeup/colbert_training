"""Score retrieved passages with a cross-encoder for distillation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from colbert.config import ColBERTConfig
from colbert.dataset.queries import Queries
from colbert.dataset.collection import Collection

logger = logging.getLogger(__name__)


def score_passages(
    config: ColBERTConfig,
    ranking: Dict[int, List[Tuple[int, float]]],
    queries: Queries,
    collection: Collection,
    batch_size: int = 128,
) -> Dict[int, List[Tuple[int, float]]]:
    """Score retrieved passages using a cross-encoder.

    Args:
        config: ColBERT configuration.
        ranking: qid -> [(pid, retriever_score), ...] from retrieval.
        queries: Query reader.
        collection: Passage collection.
        batch_size: Batch size for cross-encoder inference.

    Returns:
        qid -> [(pid, cross_encoder_score), ...] sorted by score descending.
    """
    logger.info(f"Loading cross-encoder: {config.cross_encoder}")
    ce_model = CrossEncoder(config.cross_encoder, max_length=512)

    scored: Dict[int, List[Tuple[int, float]]] = {}

    all_pairs: List[Tuple[int, int, str, str]] = []
    for qid in ranking:
        query_text = queries[qid]
        for pid, _ in ranking[qid][:config.top_k_distill]:
            passage_text = collection[pid]
            all_pairs.append((qid, pid, query_text, passage_text))

    logger.info(f"Scoring {len(all_pairs)} query-passage pairs with cross-encoder")

    all_scores: List[float] = []
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Cross-encoder scoring"):
        batch_pairs = all_pairs[i:i + batch_size]
        texts = [(qp[2], qp[3]) for qp in batch_pairs]
        scores = ce_model.predict(texts, show_progress_bar=False)
        all_scores.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))

    for (qid, pid, _, _), score in zip(all_pairs, all_scores):
        if qid not in scored:
            scored[qid] = []
        scored[qid].append((pid, float(score)))

    for qid in scored:
        scored[qid].sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Scored passages for {len(scored)} queries")
    return scored
