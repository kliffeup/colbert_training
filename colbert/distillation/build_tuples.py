"""Build w-way distillation tuples from cross-encoder scored passages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from colbert.config import ColBERTConfig
from colbert.data.queries import Queries
from colbert.data.ranking import load_qrels, get_positive_pids

logger = logging.getLogger(__name__)


def build_tuples(
    config: ColBERTConfig,
    ce_scores: Dict[int, List[Tuple[int, float]]],
    queries: Queries,
    qrels_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Build nway distillation tuples from cross-encoder scores.

    For each query:
    - Select 1 positive (labeled positive or highest CE-scored passage)
    - Sample nway-1 negatives from remaining passages

    Args:
        config: ColBERT configuration.
        ce_scores: qid -> [(pid, ce_score), ...] sorted descending.
        queries: Query reader.
        qrels_path: Path to qrels file (optional, for labeled positives).
        output_path: Where to write tuples JSONL.

    Returns:
        Path to the written tuples file.
    """
    out = Path(output_path or (config.tuples_dir + "/tuples.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)

    positives: Dict[int, Set[int]] = {}
    if qrels_path:
        qrels = load_qrels(qrels_path)
        positives = get_positive_pids(qrels)

    nway = config.nway
    count = 0

    with open(out, "w", encoding="utf-8") as f:
        for qid in ce_scores:
            if qid not in queries:
                continue

            scored_pids = ce_scores[qid]
            if len(scored_pids) < nway:
                continue

            query_text = queries[qid]
            qid_positives = positives.get(qid, set())

            # Determine positive: labeled positive if available, else top CE-scored
            positive_pid = None
            positive_score = None
            for pid, score in scored_pids:
                if pid in qid_positives:
                    positive_pid = pid
                    positive_score = score
                    break

            if positive_pid is None:
                positive_pid = scored_pids[0][0]
                positive_score = scored_pids[0][1]

            # Build tuple: positive first, then fill with other passages
            pids = [positive_pid]
            scores = [positive_score]

            for pid, score in scored_pids:
                if pid == positive_pid:
                    continue
                pids.append(pid)
                scores.append(score)
                if len(pids) == nway:
                    break

            if len(pids) < nway:
                continue

            example = {
                "qid": qid,
                "query": query_text,
                "pids": pids,
                "scores": scores,
                "positive_idx": 0,
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    logger.info(f"Built {count} distillation tuples -> {out}")
    return str(out)
