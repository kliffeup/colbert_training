"""Tests for the multi-GPU tiled MaxSim scorer.

All tests force ``device_type="cpu"`` with ``world_size=2`` and a spawn context, so
they run on a box with 0 GPUs while still exercising the exact production code path
(spawn re-imports the worker module, the two-round protocol, doc-boundary sharding,
memmap shard reads, and tiled decode). The ground truth is ``reference_score`` — a
standalone reimplementation of ``ColBERTRetriever._retrieve_single_local``.
"""

import numpy as np
import pytest
import torch

from colbert.config import ColBERTConfig
from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.saver import IndexSaver
from colbert.evaluation import parallel_scorer
from colbert.evaluation.parallel_scorer import (
    ParallelMaxSimScorer,
    shard_boundaries_by_doc,
)

DIM = 128


@pytest.fixture(autouse=True)
def _fast_timeout(monkeypatch):
    """Fail fast instead of hanging 30 min if a worker deadlocks."""
    monkeypatch.setattr(parallel_scorer, "_QUEUE_TIMEOUT_S", 60.0)


# ---------------------------------------------------------------------------
# Synthetic tiny index
# ---------------------------------------------------------------------------

def _build_index(tmp_path, n_docs=40, seed=0):
    """Build a tiny on-disk index and return the artifacts needed for scoring."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    centroids = torch.nn.functional.normalize(torch.randn(64, DIM), dim=1)
    codec = ResidualCodec(centroids, nbits=2)
    codec.set_quantization_params(torch.randn(1000, DIM) * 0.1)

    doclens = rng.integers(3, 12, size=n_docs).astype(np.int64)
    n = int(doclens.sum())
    embs = torch.nn.functional.normalize(torch.randn(n, DIM), dim=1)
    centroid_ids, packed = codec.encode(embs)

    saver = IndexSaver(tmp_path)
    saver.save_codec(codec)
    saver.save_compressed_embeddings(centroid_ids, packed)
    saver.save_doclens(doclens)

    doc_offsets = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(doclens, out=doc_offsets[1:])
    emb_to_doc = np.zeros(n, dtype=np.int32)
    off = 0
    for d, dl in enumerate(doclens):
        emb_to_doc[off:off + dl] = d
        off += dl
    pids = [f"D{i}" for i in range(n_docs)]

    return {
        "index_path": str(tmp_path),
        "codec_path": str(saver.index_dir / "codec.pt"),
        "codec": codec,
        "centroid_ids": centroid_ids,
        "packed": packed,
        "emb_to_doc": emb_to_doc,
        "doc_offsets": doc_offsets,
        "pids": pids,
        "n": n,
    }


def reference_score(ix, ncandidates, Q, cand_ids, top_k):
    """In-process reference == ColBERTRetriever._retrieve_single_local (fp32)."""
    codec, cids_all, packed_all = ix["codec"], ix["centroid_ids"], ix["packed"]
    emb_to_doc, doc_offsets, pids = ix["emb_to_doc"], ix["doc_offsets"], ix["pids"]
    Qf = Q.float()

    decompressed = codec.decode(cids_all[cand_ids], packed_all[cand_ids])
    sims = Qf @ decompressed.t()
    doc_indices = emb_to_doc[cand_ids]
    doc_scores = {}
    for d in np.unique(doc_indices):
        mask = doc_indices == d
        doc_scores[int(d)] = sims[:, mask].max(dim=1).values.sum().item()
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    cand_doc_idxs = [d for d, _ in sorted_docs[:ncandidates]]

    final = []
    for d in cand_doc_idxs:
        s, e = int(doc_offsets[d]), int(doc_offsets[d + 1])
        de = codec.decode(cids_all[s:e], packed_all[s:e])
        final.append((str(pids[d]), (Qf @ de.t()).max(dim=1).values.sum().item()))
    final.sort(key=lambda x: x[1], reverse=True)
    return final[:top_k]


def _make_config(ncandidates):
    # nprobe*ncandidates_factor == ncandidates; nprobe=1 keeps it simple.
    return ColBERTConfig(dim=DIM, nbits=2, nprobe=1, ncandidates_factor=ncandidates)


def _assert_ranking_close(got, ref, atol=1e-4):
    assert [p for p, _ in got] == [p for p, _ in ref], f"\ngot={got}\nref={ref}"
    for (_, sg), (_, sr) in zip(got, ref):
        assert abs(sg - sr) <= atol, f"score {sg} vs {sr}"


# ---------------------------------------------------------------------------
# shard_boundaries_by_doc
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [1, 2, 3, 8, 50])
def test_shard_boundaries_on_doc_edges(world_size):
    doclens = np.array([5, 1, 8, 3, 2, 7, 4, 6], dtype=np.int64)
    doc_offsets = np.zeros(len(doclens) + 1, dtype=np.int64)
    np.cumsum(doclens, out=doc_offsets[1:])
    edges = set(doc_offsets.tolist())

    emb_bounds, doc_bounds = shard_boundaries_by_doc(doc_offsets, world_size)

    assert len(emb_bounds) == world_size + 1
    assert emb_bounds[0] == 0 and emb_bounds[-1] == doc_offsets[-1]
    assert doc_bounds[0] == 0 and doc_bounds[-1] == len(doclens)
    # Every boundary sits on a document edge, and bands are non-decreasing (may be empty).
    for b in emb_bounds:
        assert int(b) in edges
    assert np.all(np.diff(emb_bounds) >= 0)
    assert np.all(np.diff(doc_bounds) >= 0)


# ---------------------------------------------------------------------------
# End-to-end equivalence: CPU pool vs reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile_size", [3, 10_000])
@pytest.mark.parametrize("ncandidates", [5, 100])  # 5 exercises the top-N cut; 100 does not
def test_parallel_equals_reference_cpu(tmp_path, tile_size, ncandidates):
    ix = _build_index(tmp_path, n_docs=40, seed=1)
    config = _make_config(ncandidates)

    scorer = ParallelMaxSimScorer(
        index_path=ix["index_path"], codec_path=ix["codec_path"],
        doc_offsets=ix["doc_offsets"], pids=ix["pids"], config=config,
        world_size=2, device_type="cpu", tile_size=tile_size,
    )
    try:
        for seed in range(4):
            torch.manual_seed(100 + seed)
            Q = torch.randn(config.query_maxlen, DIM)
            # ~60% of embeddings as candidates, sorted unique.
            rng = np.random.default_rng(seed)
            k = int(ix["n"] * 0.6)
            cand = np.sort(rng.choice(ix["n"], size=k, replace=False)).astype(np.int64)

            got = scorer.score(Q, cand, top_k=10)
            ref = reference_score(ix, config.ncandidates, Q, cand, top_k=10)
            _assert_ranking_close(got, ref)
    finally:
        scorer.close()


def test_empty_candidates(tmp_path):
    ix = _build_index(tmp_path, n_docs=20, seed=2)
    scorer = ParallelMaxSimScorer(
        index_path=ix["index_path"], codec_path=ix["codec_path"],
        doc_offsets=ix["doc_offsets"], pids=ix["pids"], config=_make_config(50),
        world_size=2, device_type="cpu",
    )
    try:
        Q = torch.randn(32, DIM)
        assert scorer.score(Q, np.array([], dtype=np.int64), top_k=10) == []
    finally:
        scorer.close()


def test_candidates_all_on_one_worker(tmp_path):
    """Candidates confined to worker 0's doc range; worker 1 returns nothing."""
    ix = _build_index(tmp_path, n_docs=40, seed=3)
    config = _make_config(100)
    scorer = ParallelMaxSimScorer(
        index_path=ix["index_path"], codec_path=ix["codec_path"],
        doc_offsets=ix["doc_offsets"], pids=ix["pids"], config=config,
        world_size=2, device_type="cpu",
    )
    try:
        # First worker owns emb ids [0, emb_bounds[1]); take candidates only from there.
        hi = int(scorer.emb_bounds[1])
        cand = np.arange(0, hi, dtype=np.int64)
        Q = torch.randn(config.query_maxlen, DIM)
        got = scorer.score(Q, cand, top_k=10)
        ref = reference_score(ix, config.ncandidates, Q, cand, top_k=10)
        _assert_ranking_close(got, ref)
    finally:
        scorer.close()


def test_close_idempotent(tmp_path):
    ix = _build_index(tmp_path, n_docs=10, seed=4)
    scorer = ParallelMaxSimScorer(
        index_path=ix["index_path"], codec_path=ix["codec_path"],
        doc_offsets=ix["doc_offsets"], pids=ix["pids"], config=_make_config(50),
        world_size=2, device_type="cpu",
    )
    scorer.close()
    scorer.close()  # no error
    assert all(not p.is_alive() for p in scorer._procs)


def test_worker_startup_error_propagates(tmp_path):
    """A bad index path makes workers fail to memmap; the master must raise, not hang."""
    doc_offsets = np.array([0, 3, 7], dtype=np.int64)
    with pytest.raises(RuntimeError):
        ParallelMaxSimScorer(
            index_path=str(tmp_path / "does_not_exist"),
            codec_path=str(tmp_path / "nope.pt"),
            doc_offsets=doc_offsets, pids=["D0", "D1"], config=_make_config(10),
            world_size=2, device_type="cpu",
        )
