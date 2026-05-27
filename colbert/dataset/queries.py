"""Query reader for `qid\\ttext` TSV files. Loaded fully into memory (queries are small)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple


class Queries:
    """Dict-like reader for `qid<TAB>query_text` TSVs."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Queries file not found: {path}")
        self._data: dict[str, str] = {}
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                self._data[parts[0]] = parts[1]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, qid) -> bool:
        return str(qid) in self._data

    def __getitem__(self, qid) -> str:
        return self._data[str(qid)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def items(self) -> List[Tuple[str, str]]:
        return list(self._data.items())

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()
