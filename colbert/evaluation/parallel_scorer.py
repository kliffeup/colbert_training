"""Multi-GPU tiled MaxSim scorer for ColBERTv2 retrieval.

The single-GPU retriever (:mod:`colbert.evaluation.retriever`) decodes *all* of a
query's candidate embeddings on one device and computes ``Q @ candidates.t()`` in one
shot. For an end-to-end document index (billions of token embeddings, long docs) that
matrix OOMs a single GPU.

This module fans that scoring out across all GPUs with a persistent pool of workers
(one per device, ``torch.multiprocessing`` spawn — the same idiom as the indexing
encoder). Key properties:

- **Doc-boundary sharding.** Worker ``g`` owns a contiguous embedding-id range that
  starts and ends on document boundaries, so no document ever straddles two workers.
  Each document's MaxSim is therefore fully local — scores from disjoint workers just
  merge, with no cross-worker reduction.
- **Memory-mapped shards.** Workers ``np.load(..., mmap_mode="r")`` the compressed
  arrays and gather only the rows they need. The OS page cache is shared across
  processes, so there is no explicit multi-GB copy and the master need not hold the
  arrays at all.
- **On-GPU tiled decode.** Candidate rows are decoded on the GPU in tiles and folded
  into a running ``(qlen, n_docs)`` max buffer via ``scatter_reduce``, so peak memory
  is bounded by the tile size regardless of how many candidates a query pulls in.

Candidate *generation* (nearest centroids + inverted-list lookup) stays on the master;
only the decode+MaxSim scoring is parallelized here.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.multiprocessing as tmp

from colbert.config import ColBERTConfig
from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.saver import IndexSaver

logger = logging.getLogger(__name__)

# Generous per-message timeout when waiting on worker results; if it elapses we check
# whether the workers are still alive and raise rather than hang forever.
_QUEUE_TIMEOUT_S = 1800.0


# ---------------------------------------------------------------------------
# Doc-boundary sharding (pure, unit-tested)
# ---------------------------------------------------------------------------

def shard_boundaries_by_doc(
    doc_offsets: np.ndarray,
    world_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition the embedding space into ``world_size`` contiguous doc-aligned shards.

    Args:
        doc_offsets: int64 array of shape ``(n_docs + 1,)``; ``doc_offsets[d]`` is the
            first embedding id of doc ``d`` and ``doc_offsets[-1]`` is the total number
            of token embeddings ``N`` (as built in the retriever from ``doclens``).
        world_size: Number of shards / workers.

    Returns:
        ``(emb_bounds, doc_bounds)``, each int64 of shape ``(world_size + 1,)``. Worker
        ``g`` owns embedding ids ``[emb_bounds[g], emb_bounds[g+1])`` == documents
        ``[doc_bounds[g], doc_bounds[g+1])``. Every ``emb_bounds`` value equals some
        ``doc_offsets`` entry, so no document straddles a shard. Bands may be empty when
        ``world_size > n_docs``.
    """
    doc_offsets = np.asarray(doc_offsets, dtype=np.int64)
    n_docs = doc_offsets.shape[0] - 1
    total = int(doc_offsets[-1])

    doc_bounds = np.empty(world_size + 1, dtype=np.int64)
    for g in range(world_size + 1):
        target = (g * total) // world_size if world_size > 0 else total
        # Smallest doc edge whose embedding offset is >= target.
        doc_bounds[g] = int(np.searchsorted(doc_offsets, target, side="left"))
    doc_bounds[0] = 0
    doc_bounds[world_size] = n_docs
    # Non-decreasing (targets are non-decreasing); clamp defensively.
    np.maximum.accumulate(doc_bounds, out=doc_bounds)

    emb_bounds = doc_offsets[doc_bounds]

    # Invariant: every boundary sits on a document edge.
    assert emb_bounds[0] == 0 and emb_bounds[-1] == total
    return emb_bounds, doc_bounds


# ---------------------------------------------------------------------------
# Core scoring (shared by both rounds, runs inside each worker)
# ---------------------------------------------------------------------------

def _score_candidate_ids(
    Q_dev: torch.Tensor,
    cand_ids: np.ndarray,
    codec: ResidualCodec,
    cids_mm: np.ndarray,
    packed_mm: np.ndarray,
    doc_offsets: np.ndarray,
    device: torch.device,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-document MaxSim scores for a set of embedding ids.

    Decodes ``cand_ids`` rows (from the memory-mapped shard) in tiles on ``device`` and
    reduces them into per-document scores. Used for both the approximate round (a
    subset of each doc's tokens) and the exact round (all of each doc's tokens).

    Returns:
        ``(doc_idxs, scores)`` numpy arrays: the distinct global document ids present in
        ``cand_ids`` and their MaxSim scores. Empty arrays if ``cand_ids`` is empty.
    """
    n = cand_ids.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    # Map each candidate embedding id to its global document id, then to a dense slot.
    doc_of_cand = np.searchsorted(doc_offsets, cand_ids, side="right") - 1
    unique_docs, inverse = np.unique(doc_of_cand, return_inverse=True)
    n_docs = unique_docs.shape[0]
    seg = torch.as_tensor(inverse, device=device, dtype=torch.int64)

    qlen = Q_dev.shape[0]
    buf = Q_dev.new_full((qlen, n_docs), torch.finfo(torch.float32).min)

    for start in range(0, n, tile_size):
        sl = slice(start, min(start + tile_size, n))
        ids_chunk = cand_ids[sl]
        # Fancy-indexing the memmap copies just these rows into RAM.
        cids_chunk = np.asarray(cids_mm[ids_chunk])
        packed_chunk = np.asarray(packed_mm[ids_chunk])
        dec = codec.decode_to(cids_chunk, packed_chunk, device)  # (c, dim) fp32
        sims = Q_dev @ dec.t()  # (qlen, c)
        idx = seg[sl].unsqueeze(0).expand(qlen, -1)
        buf.scatter_reduce_(1, idx, sims, reduce="amax", include_self=True)

    scores = buf.sum(dim=0)  # (n_docs,)
    return unique_docs.astype(np.int64), scores.detach().to("cpu", torch.float32).numpy()


def _full_token_ids(doc_idxs: np.ndarray, doc_offsets: np.ndarray) -> np.ndarray:
    """Flatten each document's full contiguous token-embedding id range into one array."""
    if doc_idxs.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    starts = doc_offsets[doc_idxs]
    ends = doc_offsets[doc_idxs + 1]
    return np.concatenate(
        [np.arange(s, e, dtype=np.int64) for s, e in zip(starts.tolist(), ends.tolist())]
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_serve_loop(
    rank: int,
    device_type: str,
    emb_lo: int,
    emb_hi: int,
    index_path: str,
    codec_path: str,
    doc_offsets: np.ndarray,
    config: ColBERTConfig,
    tile_size: int,
    in_q,
    out_q,
) -> None:
    """Persistent worker: memmap this rank's shard, then serve scoring requests.

    Message protocol (in_q -> out_q):
        ("SCORE_A", req_id, Q_np, cand_ids, top_n) -> ("RESULT", req_id, rank, docs, scores)
        ("SCORE_B", req_id, Q_np, doc_idxs)         -> ("RESULT", req_id, rank, docs, scores)
        ("STOP",)                                    -> loop exits
    On any exception: ("ERROR", req_id, rank, traceback_str).
    """
    try:
        if device_type == "cuda":
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        codec = ResidualCodec.load(codec_path)
        saver = IndexSaver(index_path)
        cids_path, packed_path = saver.compressed_embeddings_paths()
        cids_mm = np.load(cids_path, mmap_mode="r")
        packed_mm = np.load(packed_path, mmap_mode="r")
        doc_offsets = np.asarray(doc_offsets, dtype=np.int64)

        # Warm up: populate the codec's per-device caches and pay the cuBLAS init cost
        # on a dummy row so the first real query isn't skewed.
        if emb_hi > emb_lo:
            warm = np.asarray([emb_lo], dtype=np.int64)
            _score_candidate_ids(
                torch.zeros(config.query_maxlen, codec.dim, device=device),
                warm, codec, cids_mm, packed_mm, doc_offsets, device, tile_size,
            )

        out_q.put(("READY", rank))

        while True:
            msg = in_q.get()
            tag = msg[0]
            if tag == "STOP":
                break

            req_id = msg[1]
            try:
                Q_dev = torch.as_tensor(msg[2], device=device, dtype=torch.float32)
                if tag == "SCORE_A":
                    cand_ids, top_n = msg[3], msg[4]
                    docs, scores = _score_candidate_ids(
                        Q_dev, cand_ids, codec, cids_mm, packed_mm,
                        doc_offsets, device, tile_size,
                    )
                    if top_n is not None and docs.shape[0] > top_n:
                        keep = np.argpartition(-scores, top_n)[:top_n]
                        docs, scores = docs[keep], scores[keep]
                elif tag == "SCORE_B":
                    doc_idxs = msg[3]
                    cand_ids = _full_token_ids(doc_idxs, doc_offsets)
                    docs, scores = _score_candidate_ids(
                        Q_dev, cand_ids, codec, cids_mm, packed_mm,
                        doc_offsets, device, tile_size,
                    )
                else:
                    raise ValueError(f"Unknown message tag: {tag!r}")
                out_q.put(("RESULT", req_id, rank, docs, scores))
            except Exception:  # noqa: BLE001 — report and keep serving
                out_q.put(("ERROR", req_id, rank, traceback.format_exc()))
    except Exception:  # noqa: BLE001 — fatal setup error; report so master doesn't hang
        out_q.put(("ERROR", -1, rank, traceback.format_exc()))


# ---------------------------------------------------------------------------
# Pool manager (lives on the master)
# ---------------------------------------------------------------------------

class ParallelMaxSimScorer:
    """Persistent pool of per-device MaxSim scoring workers.

    The retriever generates candidate embedding ids on the master and hands them here;
    :meth:`score` fans the decode+MaxSim work across all workers and returns the final
    ``[(pid, score)]`` ranking. Always :meth:`close` the pool (context-manager or the
    retriever's ``close()``).
    """

    def __init__(
        self,
        index_path: str,
        codec_path: str,
        doc_offsets: np.ndarray,
        pids: List[str],
        config: ColBERTConfig,
        world_size: int,
        device_type: str = "cuda",
        tile_size: int = 200_000,
    ):
        self.pids = pids
        self.config = config
        self.world_size = world_size
        self.tile_size = tile_size
        self._req_counter = 0
        self._closed = False

        self.doc_offsets = np.asarray(doc_offsets, dtype=np.int64)
        self.emb_bounds, self.doc_bounds = shard_boundaries_by_doc(self.doc_offsets, world_size)

        ctx = tmp.get_context("spawn")
        self._in_qs = [ctx.Queue() for _ in range(world_size)]
        self._out_q = ctx.Queue()
        self._procs: List = []
        for g in range(world_size):
            p = ctx.Process(
                target=_worker_serve_loop,
                args=(
                    g, device_type, int(self.emb_bounds[g]), int(self.emb_bounds[g + 1]),
                    str(index_path), str(codec_path), self.doc_offsets, config,
                    tile_size, self._in_qs[g], self._out_q,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

        # Barrier: block until every worker has warmed up and reported READY.
        self._await_ready()
        logger.info(
            f"ParallelMaxSimScorer: {world_size} {device_type} workers ready "
            f"(tile_size={tile_size})"
        )

    # -- lifecycle ----------------------------------------------------------
    def _await_ready(self) -> None:
        pending = set(range(self.world_size))
        while pending:
            msg = self._get_or_raise(expected_req=None)
            if msg[0] == "READY":
                pending.discard(msg[1])
            elif msg[0] == "ERROR":
                self.close()
                raise RuntimeError(f"[worker {msg[2]}] failed during startup:\n{msg[3]}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for q in self._in_qs:
            try:
                q.put(("STOP",))
            except Exception:  # noqa: BLE001
                pass
        for p in self._procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> "ParallelMaxSimScorer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()

    # -- queue helpers ------------------------------------------------------
    def _get_or_raise(self, expected_req):
        """Pull one message; raise on worker ERROR or if a worker has died."""
        try:
            return self._out_q.get(timeout=_QUEUE_TIMEOUT_S)
        except Exception as e:  # queue.Empty (timeout) or transport error
            if any(not p.is_alive() for p in self._procs):
                raise RuntimeError("A scoring worker died unexpectedly.") from e
            raise RuntimeError(f"Timed out waiting for worker results (req={expected_req}).") from e

    def _gather(self, req_id: int, ranks) -> Tuple[np.ndarray, np.ndarray]:
        """Collect RESULT messages for ``req_id`` from the given set of ranks; concat."""
        pending = set(ranks)
        docs_parts, score_parts = [], []
        while pending:
            msg = self._get_or_raise(expected_req=req_id)
            if msg[0] == "ERROR":
                raise RuntimeError(f"[worker {msg[2]}] scoring error (req={msg[1]}):\n{msg[3]}")
            _, r_req, rank, docs, scores = msg
            if r_req != req_id:
                continue  # stale (shouldn't happen: scoring is sequential)
            pending.discard(rank)
            docs_parts.append(docs)
            score_parts.append(scores)
        if not docs_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(docs_parts), np.concatenate(score_parts)

    def _next_req(self) -> int:
        self._req_counter += 1
        return self._req_counter

    # -- public API ---------------------------------------------------------
    @torch.no_grad()
    def score(
        self,
        Q: torch.Tensor,
        candidate_emb_ids: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Score a single query's candidate embeddings and return the top-k ranking.

        Args:
            Q: Query embeddings of shape ``(qlen, dim)``.
            candidate_emb_ids: Sorted int64 array of candidate embedding ids (from the
                master's inverted-list lookup).
            top_k: Number of documents to return.

        Returns:
            ``[(pid, score)]`` sorted by score descending, truncated to ``top_k``.
        """
        candidate_emb_ids = np.asarray(candidate_emb_ids, dtype=np.int64)
        if candidate_emb_ids.shape[0] == 0:
            return []

        Q_np = Q.detach().to("cpu", torch.float32).contiguous().numpy()

        # --- Round A: approximate MaxSim over candidate embeddings ---
        pos = np.searchsorted(candidate_emb_ids, self.emb_bounds)
        top_n = self.config.ncandidates
        req_a = self._next_req()
        ranks_a = []
        for g in range(self.world_size):
            cand_g = candidate_emb_ids[pos[g]:pos[g + 1]]
            if cand_g.shape[0] == 0:
                continue
            self._in_qs[g].put(("SCORE_A", req_a, Q_np, cand_g, top_n))
            ranks_a.append(g)
        docs_a, scores_a = self._gather(req_a, ranks_a)
        if docs_a.shape[0] == 0:
            return []

        # Global top-ncandidates documents for exact re-ranking.
        if docs_a.shape[0] > top_n:
            keep = np.argpartition(-scores_a, top_n)[:top_n]
            docs_a = docs_a[keep]
        candidate_doc_idxs = docs_a

        # --- Round B: exact re-rank of the selected docs on their owning workers ---
        req_b = self._next_req()
        ranks_b = []
        for g in range(self.world_size):
            mask = (candidate_doc_idxs >= self.doc_bounds[g]) & (
                candidate_doc_idxs < self.doc_bounds[g + 1]
            )
            docs_g = candidate_doc_idxs[mask]
            if docs_g.shape[0] == 0:
                continue
            self._in_qs[g].put(("SCORE_B", req_b, Q_np, docs_g))
            ranks_b.append(g)
        docs_b, scores_b = self._gather(req_b, ranks_b)
        if docs_b.shape[0] == 0:
            return []

        order = np.argsort(-scores_b)[:top_k]
        return [(str(self.pids[int(docs_b[i])]), float(scores_b[i])) for i in order]
