"""Qrels readers and ranking I/O.

The qrels format is the 3-column TSV `qid<TAB>docid<TAB>rel` produced by
`scripts/preprocess_msmarco_docs.py`. The original TREC 4-column form
`qid 0 docid rel` is also accepted on read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Return ``{qid: {docid: rel}}`` from a qrels TSV.

    Accepts both 3-column (`qid\\tdocid\\trel`) and 4-column TREC
    (`qid 0 docid rel`, whitespace-separated) layouts. Non-positive
    relevances are preserved (downstream code may want them).
    """
    qrels: Dict[str, Dict[str, int]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Qrels not found: {path}")

    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            else:
                continue
            try:
                rel_i = int(rel)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[docid] = rel_i
    return qrels


def get_positive_pids(qrels: Dict[Any, Dict[Any, int]]) -> Dict[Any, Set[Any]]:
    """Return ``{qid: {docids with rel > 0}}``."""
    return {qid: {pid for pid, rel in rels.items() if rel > 0} for qid, rels in qrels.items()}


def save_ranking(
    ranking: Dict[Any, List[Tuple[Any, float]]],
    path: str,
    run_name: str = "colbert",
) -> None:
    """Write a ranking as TREC-style `qid Q0 docid rank score run` lines."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for qid, hits in ranking.items():
            for rank, (pid, score) in enumerate(hits, start=1):
                f.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score:.6f}\t{run_name}\n")
