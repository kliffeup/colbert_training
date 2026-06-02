"""Passage/document collection reader for `pid\\ttext` TSV files.

Builds a one-pass byte-offset index so that random access by pid is O(1) seek+read
without holding the corpus (e.g. MS MARCO Doc v1 ~22 GB) in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

logger = logging.getLogger(__name__)


class Collection:
    """TSV-backed collection of (pid, text) rows with byte-offset random access.

    The TSV is scanned once on init to record `(pid -> (offset, length))`. Lookups
    `collection[pid]` seek to the offset and read just that line. Iteration walks
    the file sequentially. The full text content is never loaded into RAM.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Collection file not found: {path}")

        self._offsets: dict[str, Tuple[int, int]] = {}
        self._pid_order: List[str] = []
        self._build_offset_index()
        logger.info(
            f"Collection: indexed {len(self._pid_order):,} rows from {self.path}"
        )

    def _build_offset_index(self) -> None:
        with open(self.path, "rb") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                length = len(line)
                # Find the first tab byte to extract pid without decoding the body.
                tab = line.find(b"\t")
                if tab > 0:
                    pid = line[:tab].decode("utf-8")
                    self._offsets[pid] = (offset, length)
                    self._pid_order.append(pid)
                offset += length

    def __len__(self) -> int:
        return len(self._pid_order)

    def __contains__(self, pid) -> bool:
        return str(pid) in self._offsets

    def __getitem__(self, pid) -> str:
        key = str(pid)
        loc = self._offsets.get(key)
        if loc is None:
            raise KeyError(f"pid {pid!r} not in collection")
        offset, length = loc
        with open(self.path, "rb") as f:
            f.seek(offset)
            raw = f.read(length).decode("utf-8", errors="replace").rstrip("\n")
        tab = raw.find("\t")
        return raw[tab + 1:] if tab >= 0 else ""

    def pids(self) -> Iterable[str]:
        return iter(self._pid_order)

    def shard_size(self, rank: int, world_size: int) -> int:
        n = len(self._pid_order)
        base, rem = divmod(n, world_size)
        return base + (1 if rank < rem else 0)

    def _shard_pid_slice(self, rank: int, world_size: int) -> List[str]:
        """Contiguous slice of pids belonging to this rank."""
        n = len(self._pid_order)
        base, rem = divmod(n, world_size)
        start = rank * base + min(rank, rem)
        size = base + (1 if rank < rem else 0)
        return self._pid_order[start:start + size]

    def iterate(
        self, batch_size: int, start: int = 0
    ) -> Iterator[List[Tuple[str, str]]]:
        """Yield batches of (pid, text) walking the file sequentially.

        ``start`` skips the first ``start`` documents (in deterministic file order)
        before emitting any batch — used to resume an interrupted encode pass.
        """
        seen = 0
        batch: List[Tuple[str, str]] = []
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                tab = line.find("\t")
                if tab <= 0:
                    continue
                seen += 1
                if seen <= start:
                    continue
                batch.append((line[:tab], line[tab + 1:]))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def iterate_shard(
        self, rank: int, world_size: int, batch_size: int, start: int = 0
    ) -> Iterator[List[Tuple[str, str]]]:
        """Yield (pid, text) batches for this rank's contiguous slice.

        ``start`` skips the first ``start`` documents *within this shard* (in
        deterministic shard order) — used to resume an interrupted encode pass.
        """
        shard_pids = set(self._shard_pid_slice(rank, world_size))
        seen = 0
        batch: List[Tuple[str, str]] = []
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                tab = line.find("\t")
                if tab <= 0:
                    continue
                pid = line[:tab]
                if pid not in shard_pids:
                    continue
                seen += 1
                if seen <= start:
                    continue
                batch.append((pid, line[tab + 1:]))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
