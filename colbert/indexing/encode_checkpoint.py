"""Crash-safe checkpointing for the fused encode→compress pass.

Wraps a :class:`CompressedShardWriter` (the compressed codes) together with two
append-only sidecars that mirror the in-RAM per-doc metadata, plus a JSON file that
acts as the single commit point:

  * ``centroid_ids{tag}.bin`` / ``packed_residuals{tag}.bin`` — the writer's codes.
  * ``doclens{tag}.partial.bin``  — int32 per doc (kept tokens).
  * ``pids{tag}.partial.txt``     — one pid per line.
  * ``_encode_ckpt{tag}.json``    — ``{"docs_done": N, "total_tokens": T}``.

At each checkpoint the three data files are flushed and fsync'd, then the JSON is
written atomically (temp + ``os.replace``).  Because the JSON is written last, a kill
at any instant leaves the JSON pointing at a fully-durable prefix; on resume the data
files are truncated back to those committed counts, discarding any rows from a batch
that was in flight.  Checkpoints only happen on batch boundaries, so the codes,
doclens and pids always agree: ``total_tokens == sum(doclens[:docs_done])``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from colbert.indexing.compressed_writer import CompressedShardWriter

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write ``obj`` as JSON to ``path`` atomically (temp file + ``os.replace``)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_doclens_truncate(path: Path, n: int) -> List[int]:
    """Load the first ``n`` int32 doclens and truncate the file to exactly ``n`` rows."""
    data = np.fromfile(path, dtype=np.int32, count=n)
    with open(path, "r+b") as f:
        f.truncate(n * 4)
    return data.tolist()


def _load_pids_truncate(path: Path, n: int) -> List[str]:
    """Load the first ``n`` pids (one per line) and truncate the file after line ``n``."""
    pids: List[str] = []
    offset = 0
    with open(path, "rb") as f:
        for line in f:
            offset += len(line)
            pids.append(line.rstrip(b"\n").decode("utf-8"))
            if len(pids) >= n:
                break
    with open(path, "r+b") as f:
        f.truncate(offset)
    return pids


class EncodeCheckpointer:
    """Append batches of compressed codes + per-doc metadata, checkpointing durably.

    Construct via :meth:`from_dir`.  After construction, ``docs_done`` is the number of
    already-committed documents (0 for a fresh pass; the resumed count otherwise) and
    ``doclens`` / ``pids`` are the in-RAM accumulators seeded from the sidecars.
    """

    def __init__(
        self,
        writer: CompressedShardWriter,
        work_dir: Path,
        tag: str,
        doclens: List[int],
        pids: List[str],
        doclens_f,
        pids_f,
    ):
        self.writer = writer
        self.work_dir = work_dir
        self.tag = tag
        self.doclens = doclens
        self.pids = pids
        self.docs_done = len(doclens)
        self._doclens_f = doclens_f
        self._pids_f = pids_f
        self._ckpt_path = work_dir / f"_encode_ckpt{tag}.json"

    @classmethod
    def from_dir(
        cls,
        work_dir: str | Path,
        bytes_per_residual: int,
        tag: str = "",
        resume: bool = False,
    ) -> "EncodeCheckpointer":
        """Open (or resume) a checkpointed encode pass under ``work_dir``."""
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = work_dir / f"_encode_ckpt{tag}.json"
        doclens_path = work_dir / f"doclens{tag}.partial.bin"
        pids_path = work_dir / f"pids{tag}.partial.txt"

        if resume and ckpt_path.exists():
            with open(ckpt_path, encoding="utf-8") as f:
                state = json.load(f)
            docs_done = int(state["docs_done"])
            total_tokens = int(state["total_tokens"])
            doclens = _load_doclens_truncate(doclens_path, docs_done)
            pids = _load_pids_truncate(pids_path, docs_done)
            writer = CompressedShardWriter.open_resume(
                work_dir, bytes_per_residual, tag=tag, total_tokens=total_tokens
            )
            doclens_f = open(doclens_path, "ab")
            pids_f = open(pids_path, "a", encoding="utf-8")
            logger.info(
                f"Resuming encode pass{tag or ''}: {docs_done} docs / "
                f"{total_tokens} tokens already committed"
            )
        else:
            writer = CompressedShardWriter(work_dir, bytes_per_residual, tag=tag)
            doclens_f = open(doclens_path, "wb")
            pids_f = open(pids_path, "w", encoding="utf-8")
            doclens, pids = [], []

        return cls(writer, work_dir, tag, doclens, pids, doclens_f, pids_f)

    def append_batch(
        self,
        centroid_ids: np.ndarray,
        packed: np.ndarray,
        batch_doclens: List[int],
        batch_pids: List[str],
    ) -> None:
        """Append one batch's codes + per-doc metadata (not yet durable)."""
        self.writer.append(centroid_ids, packed)
        np.asarray(batch_doclens, dtype=np.int32).tofile(self._doclens_f)
        for pid in batch_pids:
            self._pids_f.write(f"{pid}\n")
        self.doclens.extend(int(x) for x in batch_doclens)
        self.pids.extend(batch_pids)

    def checkpoint(self) -> None:
        """Flush all data durably, then atomically commit the checkpoint JSON."""
        self.writer.flush_sync()
        for f in (self._doclens_f, self._pids_f):
            f.flush()
            os.fsync(f.fileno())
        _atomic_write_json(
            self._ckpt_path,
            {"docs_done": len(self.doclens), "total_tokens": self.writer.total},
        )
        self.docs_done = len(self.doclens)

    @property
    def doclens_array(self) -> np.ndarray:
        return np.array(self.doclens, dtype=np.int32)

    def close_sidecars(self) -> None:
        """Close the metadata sidecar handles (leaves the writer's ``.bin`` open)."""
        for f in (self._doclens_f, self._pids_f):
            if not f.closed:
                f.close()

    def cleanup(self) -> None:
        """Remove the checkpoint JSON and metadata sidecars (after a successful finalize)."""
        self.close_sidecars()
        for name in (
            f"_encode_ckpt{self.tag}.json",
            f"doclens{self.tag}.partial.bin",
            f"pids{self.tag}.partial.txt",
        ):
            (self.work_dir / name).unlink(missing_ok=True)

    def finalize(
        self, cids_path: str | Path, residuals_path: str | Path
    ) -> Tuple[np.ndarray, List[str], int]:
        """Finalize the writer to ``.npy`` and clean up; return (doclens, pids, total).

        For the single-GPU path, which owns the final artifacts directly.  Multi-GPU
        workers instead keep their ``.bin`` for the master to merge (see the encoder).
        """
        total_tokens = self.writer.finalize_npy(cids_path, residuals_path)
        doclens = self.doclens_array
        pids = list(self.pids)
        self.cleanup()
        return doclens, pids, total_tokens
