"""Passage-id → doc-id mapping and MaxP score aggregation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class PassageDocMap:
    """In-memory passage_id -> doc_id lookup.

    File format (TSV, no header): `passage_id<TAB>doc_id` per line.
    """

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    @classmethod
    def load(cls, path: str | Path) -> "PassageDocMap":
        mapping: Dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                pid, did = line.split("\t", 1)
                mapping[pid] = did
        return cls(mapping)

    def __contains__(self, pid: str) -> bool:
        return pid in self.mapping

    def __getitem__(self, pid: str) -> str:
        return self.mapping[pid]

    def get(self, pid: str, default: str | None = None) -> str | None:
        return self.mapping.get(pid, default)


def aggregate(
    passage_ranking: Dict[str, List[Tuple[str, float]]],
    pmap: PassageDocMap,
    top_k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    """MaxP aggregation: passage scores → doc scores.

    For each query, group passage hits by their doc_id and keep the max score per doc;
    return the top_k docs per query, sorted descending by score.

    Passages whose pid is not in the map are skipped (logged at the call site).
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, hits in passage_ranking.items():
        best: Dict[str, float] = {}
        for pid, score in hits:
            did = pmap.get(pid)
            if did is None:
                continue
            if did not in best or score > best[did]:
                best[did] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out[qid] = ranked
    return out
