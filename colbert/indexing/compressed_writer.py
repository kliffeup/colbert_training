"""Stream compressed embeddings to disk during the fused encode→compress pass.

The index build never materializes the raw float32 embeddings (512 B/token).  Each
batch is compressed immediately and its codes appended here, so peak RAM is ~one
batch regardless of collection size.  ``total_tokens`` is unknown until the pass
finishes, so codes are appended to raw temp ``.bin`` files and finalized into proper
``.npy`` artifacts (the formats ``IndexSaver.load_compressed_embeddings`` expects)
once the row count is known.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CompressedShardWriter:
    """Append-only writer for compressed codes, with deferred ``.npy`` finalization.

    Writes two temp binaries under ``work_dir`` (suffixed by ``tag`` so multiple
    writers — e.g. one per GPU shard — can coexist in the same directory):
      * ``centroid_ids{tag}.bin``     — uint32, shape (total,)
      * ``packed_residuals{tag}.bin`` — uint8,  shape (total, bytes_per_residual)
    """

    def __init__(self, work_dir: str | Path, bytes_per_residual: int, tag: str = ""):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.bpr = int(bytes_per_residual)
        self.total = 0

        self.cids_bin = self.work_dir / f"centroid_ids{tag}.bin"
        self.res_bin = self.work_dir / f"packed_residuals{tag}.bin"
        self._cids_f = open(self.cids_bin, "wb")
        self._res_f = open(self.res_bin, "wb")
        self._closed = False

    def append(self, centroid_ids: np.ndarray, packed: np.ndarray) -> None:
        """Append one batch of compressed codes to the temp binaries."""
        n = len(centroid_ids)
        if n == 0:
            return
        if packed.shape != (n, self.bpr):
            raise ValueError(
                f"packed shape {packed.shape} != ({n}, {self.bpr})"
            )
        cids = np.ascontiguousarray(centroid_ids, dtype=np.uint32)
        res = np.ascontiguousarray(packed, dtype=np.uint8)
        self._cids_f.write(cids.tobytes())
        self._res_f.write(res.tobytes())
        self.total += n

    def _close(self) -> None:
        if not self._closed:
            self._cids_f.close()
            self._res_f.close()
            self._closed = True

    def as_memmaps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Open the (closed) temp binaries read-only as memmaps for merging."""
        self._close()
        cids = np.memmap(self.cids_bin, dtype=np.uint32, mode="r", shape=(self.total,))
        res = np.memmap(
            self.res_bin, dtype=np.uint8, mode="r", shape=(self.total, self.bpr)
        )
        return cids, res

    def finalize_npy(
        self,
        cids_path: str | Path,
        residuals_path: str | Path,
        chunk_rows: int = 4_000_000,
        cleanup: bool = True,
    ) -> int:
        """Copy temp binaries into proper ``.npy`` artifacts, then delete the temps.

        The disk-to-disk copy is chunked so peak RAM stays at ``chunk_rows`` rows.
        Returns the total number of rows (token embeddings) written.
        """
        self._close()

        src_cids = np.memmap(self.cids_bin, dtype=np.uint32, mode="r", shape=(self.total,))
        dst_cids = np.lib.format.open_memmap(
            Path(cids_path), mode="w+", dtype=np.uint32, shape=(self.total,)
        )
        for start in range(0, self.total, chunk_rows):
            end = min(start + chunk_rows, self.total)
            dst_cids[start:end] = src_cids[start:end]
        dst_cids.flush()
        del src_cids, dst_cids

        src_res = np.memmap(
            self.res_bin, dtype=np.uint8, mode="r", shape=(self.total, self.bpr)
        )
        dst_res = np.lib.format.open_memmap(
            Path(residuals_path), mode="w+", dtype=np.uint8, shape=(self.total, self.bpr)
        )
        for start in range(0, self.total, chunk_rows):
            end = min(start + chunk_rows, self.total)
            dst_res[start:end] = src_res[start:end]
        dst_res.flush()
        del src_res, dst_res

        if cleanup:
            self.cleanup()
        return self.total

    def cleanup(self) -> None:
        self._close()
        self.cids_bin.unlink(missing_ok=True)
        self.res_bin.unlink(missing_ok=True)


def merge_compressed_shards(
    shard_bins: list[Tuple[Path, Path, int]],
    doc_order: list[Tuple[int, int, int]],
    cids_path: str | Path,
    residuals_path: str | Path,
    bytes_per_residual: int,
    total_tokens: int,
) -> None:
    """Merge per-shard compressed binaries into final ``.npy`` artifacts, doc-by-doc.

    Args:
        shard_bins: one ``(cids_bin, residuals_bin, n_tokens)`` per shard.
        doc_order: the output document order as ``(shard_idx, tok_start, tok_len)``
            tuples — ``tok_start`` is the document's first token offset *within its
            shard*.  Emitting documents in this order defines the final row layout.
        total_tokens: sum of all ``tok_len`` (== rows in the output).

    Copies one document's tokens at a time through memmaps, so peak RAM is a single
    document's codes.  Preserves the doc0-tokens, doc1-tokens, … layout the retriever
    relies on.
    """
    shard_cids = [
        np.memmap(c, dtype=np.uint32, mode="r", shape=(n,)) for c, _, n in shard_bins
    ]
    shard_res = [
        np.memmap(r, dtype=np.uint8, mode="r", shape=(n, bytes_per_residual))
        for _, r, n in shard_bins
    ]

    dst_cids = np.lib.format.open_memmap(
        Path(cids_path), mode="w+", dtype=np.uint32, shape=(total_tokens,)
    )
    dst_res = np.lib.format.open_memmap(
        Path(residuals_path), mode="w+", dtype=np.uint8,
        shape=(total_tokens, bytes_per_residual),
    )

    pos = 0
    for shard_idx, tok_start, tok_len in doc_order:
        if tok_len == 0:
            continue
        end = pos + tok_len
        s_end = tok_start + tok_len
        dst_cids[pos:end] = shard_cids[shard_idx][tok_start:s_end]
        dst_res[pos:end] = shard_res[shard_idx][tok_start:s_end]
        pos = end

    if pos != total_tokens:
        raise RuntimeError(f"merge wrote {pos} tokens, expected {total_tokens}")

    dst_cids.flush()
    dst_res.flush()
    del dst_cids, dst_res, shard_cids, shard_res
