"""Integration test: resuming an interrupted index build (single-GPU/CPU path).

Simulates a container stop mid-encode by making the encode checkpointer raise on its
second commit, then resumes the build and asserts the final artifacts are byte-for-byte
identical to an *uninterrupted* encode pass that reuses the same Stage-1 codec.  Using
the same codec sidesteps any k-means nondeterminism between independent builds, so the
test isolates exactly what resume must guarantee: the encode/inversion output is the
same whether or not the pass was interrupted.
"""

import shutil

import numpy as np
import pytest
import torch

from colbert.config import ColBERTConfig
from colbert.indexing.index_builder import build_index
from colbert.indexing.encode_checkpoint import EncodeCheckpointer
from colbert.indexing.saver import IndexSaver
from colbert.dataset.collection import Collection


DIM = 16


class FakeModel:
    """Deterministic stand-in for ColBERT (per-doc embeddings independent of batch)."""

    def eval(self):
        return self

    def state_dict(self):  # only touched on the multi-GPU path, unused here
        return {}

    def encode_docs(self, texts, maxlen=None):
        lengths = [min(len(t.split()), maxlen or 10_000) for t in texts]
        S = max(lengths + [1])
        B = len(texts)
        D = torch.zeros(B, S, DIM)
        mask = torch.zeros(B, S, dtype=torch.bool)
        for i, (t, n) in enumerate(zip(texts, lengths)):
            if n == 0:
                continue
            g = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
            v = torch.randn(n, DIM, generator=g)
            D[i, :n] = torch.nn.functional.normalize(v, dim=1)
            mask[i, :n] = True
        return D, mask


def _write_collection(path, n_docs=64):
    lines = []
    rng = np.random.default_rng(0)
    for i in range(n_docs):
        n_tok = int(rng.integers(20, 45))
        words = " ".join(f"w{rng.integers(0, 50)}" for _ in range(n_tok))
        lines.append(f"D{i}\t{words}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config(index_path):
    return ColBERTConfig(
        dim=DIM, nbits=2, doc_maxlen=64, query_maxlen=8,
        kmeans_niters=2, index_path=str(index_path),
        index_checkpoint_every=16,  # small so 64 docs / batch 8 commit several times
    )


def _load_all(index_path):
    saver = IndexSaver(index_path)
    cids, packed = saver.load_compressed_embeddings()
    return {
        "cids": cids,
        "packed": packed,
        "doclens": saver.load_doclens(),
        "pids": saver.load_pids(),
        "ivl": saver.load_inverted_lists(),
        "num_embeddings": saver.load_metadata()["num_embeddings"],
    }


def test_resume_matches_uninterrupted_encode(tmp_path, monkeypatch):
    coll_path = tmp_path / "collection.tsv"
    _write_collection(coll_path, n_docs=64)
    collection = Collection(str(coll_path))

    interrupted = tmp_path / "index_interrupted"
    config = _config(interrupted)

    # --- Run 1: interrupt the encode pass on its 2nd checkpoint commit. ---
    real_checkpoint = EncodeCheckpointer.checkpoint
    calls = {"n": 0}

    def flaky_checkpoint(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated container stop")
        return real_checkpoint(self)

    monkeypatch.setattr(EncodeCheckpointer, "checkpoint", flaky_checkpoint)
    with pytest.raises(RuntimeError, match="simulated container stop"):
        build_index(FakeModel(), collection, config,
                    index_path=str(interrupted), batch_size=8, num_gpus=1)
    monkeypatch.undo()

    # Stage 1 was committed; the encode pass got at least one checkpoint in.
    saver = IndexSaver(interrupted)
    state = saver.load_build_state()
    assert state["stage"] == "centroids"
    assert (interrupted / "codec.pt").exists()
    assert (interrupted / "_shards" / "_encode_ckpt.json").exists()

    # --- Control: a clean, uninterrupted build that REUSES the same Stage-1 codec. ---
    control = tmp_path / "index_control"
    control.mkdir()
    shutil.copy(interrupted / "codec.pt", control / "codec.pt")
    shutil.copy(interrupted / "_build_state.json", control / "_build_state.json")
    build_index(FakeModel(), collection, _config(control),
                index_path=str(control), batch_size=8, num_gpus=1, resume=True)

    # --- Run 2: resume the interrupted build; it continues from the last checkpoint. ---
    build_index(FakeModel(), collection, _config(interrupted),
                index_path=str(interrupted), batch_size=8, num_gpus=1, resume=True)

    a = _load_all(interrupted)
    b = _load_all(control)
    assert a["pids"] == b["pids"]
    assert np.array_equal(a["doclens"], b["doclens"])
    assert np.array_equal(a["cids"], b["cids"])
    assert np.array_equal(a["packed"], b["packed"])
    assert a["ivl"] == b["ivl"]
    assert a["num_embeddings"] == b["num_embeddings"]

    # The resumed index is internally consistent (every embedding inverted once).
    all_ids = sorted(i for lst in a["ivl"].values() for i in lst)
    assert all_ids == list(range(a["num_embeddings"]))
    # Temp shard dir cleaned up on completion.
    assert not (interrupted / "_shards").exists()
    assert saver.load_build_state()["stage"] == "complete"


def test_resume_rejects_mismatched_fingerprint(tmp_path):
    coll_path = tmp_path / "collection.tsv"
    _write_collection(coll_path, n_docs=64)
    index_path = tmp_path / "index"

    build_index(FakeModel(), Collection(str(coll_path)), _config(index_path),
                index_path=str(index_path), batch_size=8, num_gpus=1)

    # A different collection (fewer docs) → fingerprint mismatch → refuse to resume.
    smaller = tmp_path / "collection_small.tsv"
    _write_collection(smaller, n_docs=32)
    with pytest.raises(ValueError, match="Cannot resume"):
        build_index(FakeModel(), Collection(str(smaller)), _config(index_path),
                    index_path=str(index_path), batch_size=8, num_gpus=1, resume=True)


def test_resume_after_complete_is_noop(tmp_path):
    coll_path = tmp_path / "collection.tsv"
    _write_collection(coll_path, n_docs=48)
    index_path = tmp_path / "index"
    collection = Collection(str(coll_path))

    build_index(FakeModel(), collection, _config(index_path),
                index_path=str(index_path), batch_size=8, num_gpus=1)
    before = _load_all(index_path)

    # Resuming a completed build returns immediately and leaves artifacts untouched.
    build_index(FakeModel(), collection, _config(index_path),
                index_path=str(index_path), batch_size=8, num_gpus=1, resume=True)
    after = _load_all(index_path)
    assert np.array_equal(before["cids"], after["cids"])
    assert before["pids"] == after["pids"]
