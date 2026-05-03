"""IR evaluation metrics: MRR@k, Recall@k, nDCG@k, Success@k."""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple


def mrr_at_k(
    ranking: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int = 10,
) -> float:
    """Mean Reciprocal Rank at k.

    Args:
        ranking: qid -> [(pid, score), ...] sorted descending.
        qrels: qid -> {pid: relevance}.
        k: Cutoff depth.

    Returns:
        MRR@k averaged over queries present in qrels.
    """
    rr_sum = 0.0
    count = 0

    for qid in qrels:
        if qid not in ranking:
            count += 1
            continue

        positives = {pid for pid, rel in qrels[qid].items() if rel > 0}
        for rank, (pid, _) in enumerate(ranking[qid][:k], 1):
            if pid in positives:
                rr_sum += 1.0 / rank
                break
        count += 1

    return rr_sum / max(count, 1)


def recall_at_k(
    ranking: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int = 1000,
) -> float:
    """Recall at k.

    Args:
        ranking: qid -> [(pid, score), ...] sorted descending.
        qrels: qid -> {pid: relevance}.
        k: Cutoff depth.

    Returns:
        Recall@k averaged over queries present in qrels.
    """
    recall_sum = 0.0
    count = 0

    for qid in qrels:
        positives = {pid for pid, rel in qrels[qid].items() if rel > 0}
        if not positives:
            continue

        if qid not in ranking:
            count += 1
            continue

        retrieved = {pid for pid, _ in ranking[qid][:k]}
        recall_sum += len(positives & retrieved) / len(positives)
        count += 1

    return recall_sum / max(count, 1)


def success_at_k(
    ranking: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int = 5,
) -> float:
    """Success at k: fraction of queries with at least one relevant doc in top-k.

    Used as the primary metric for LoTTE and Wikipedia Open-QA.

    Args:
        ranking: qid -> [(pid, score), ...] sorted descending.
        qrels: qid -> {pid: relevance}.
        k: Cutoff depth.

    Returns:
        Success@k averaged over queries.
    """
    success_sum = 0.0
    count = 0

    for qid in qrels:
        positives = {pid for pid, rel in qrels[qid].items() if rel > 0}
        if not positives:
            continue

        if qid not in ranking:
            count += 1
            continue

        retrieved = {pid for pid, _ in ranking[qid][:k]}
        if positives & retrieved:
            success_sum += 1.0
        count += 1

    return success_sum / max(count, 1)


def ndcg_at_k(
    ranking: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Primary metric for BEIR evaluation.

    Args:
        ranking: qid -> [(pid, score), ...] sorted descending.
        qrels: qid -> {pid: relevance}.
        k: Cutoff depth.

    Returns:
        nDCG@k averaged over queries.
    """
    ndcg_sum = 0.0
    count = 0

    for qid in qrels:
        rels = qrels[qid]
        if not rels:
            continue

        # DCG
        dcg = 0.0
        if qid in ranking:
            for rank, (pid, _) in enumerate(ranking[qid][:k], 1):
                rel = rels.get(pid, 0)
                if rel > 0:
                    dcg += (2 ** rel - 1) / math.log2(rank + 1)

        # Ideal DCG
        ideal_rels = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(rank + 1)
            for rank, rel in enumerate(ideal_rels, 1)
            if rel > 0
        )

        if idcg > 0:
            ndcg_sum += dcg / idcg
        count += 1

    return ndcg_sum / max(count, 1)


def evaluate_ranking(
    ranking: Dict[int, List[Tuple[int, float]]],
    qrels: Dict[int, Dict[int, int]],
) -> Dict[str, float]:
    """Compute all standard metrics for a ranking."""
    return {
        "MRR@10": mrr_at_k(ranking, qrels, k=10),
        "Recall@50": recall_at_k(ranking, qrels, k=50),
        "Recall@1000": recall_at_k(ranking, qrels, k=1000),
        "nDCG@10": ndcg_at_k(ranking, qrels, k=10),
        "Success@5": success_at_k(ranking, qrels, k=5),
    }
