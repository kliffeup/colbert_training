"""Tests for the crash-safe encode checkpointer.

Exercises the resume path without a GPU/model: codes come from the real
ResidualCodec, streamed through EncodeCheckpointer.  The key invariant is that a
checkpoint is the single commit point — rows written after the last committed
checkpoint are discarded on resume, and finishing then yields the same artifacts as
an uninterrupted one-shot run.
"""

import numpy as np
import torch

from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.encode_checkpoint import EncodeCheckpointer


def _make_codec(nbits=2):
    centroids = torch.nn.functional.normalize(torch.randn(64, 128), dim=1)
    codec = ResidualCodec(centroids, nbits=nbits)
    codec.set_quantization_params(torch.randn(1000, 128) * 0.1)
    return codec


def _make_docs(n_docs=10, seed=0):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n_docs):
        n = int(rng.integers(1, 6))
        v = torch.nn.functional.normalize(
            torch.tensor(rng.standard_normal((n, 128)), dtype=torch.float32), dim=1
        )
        docs.append((f"D{i}", v))
    return docs


def _append(cp, codec, batch):
    flat = torch.cat([v for _, v in batch], dim=0)
    c, p = codec.encode(flat)
    cp.append_batch(c, p, [v.shape[0] for _, v in batch], [pid for pid, _ in batch])


def test_resume_discards_uncommitted_and_matches_oneshot(tmp_path):
    codec = _make_codec()
    bpr = codec.bytes_per_vector - 4
    docs = _make_docs(10)

    # One-shot reference: all docs in order.
    ref_cids = np.concatenate([codec.encode(v)[0] for _, v in docs])
    ref_packed = np.concatenate([codec.encode(v)[1] for _, v in docs])
    ref_doclens = [v.shape[0] for _, v in docs]
    ref_pids = [pid for pid, _ in docs]

    work = tmp_path / "work"

    # Pass 1: commit 6 docs, then append 2 MORE uncommitted, flush them to disk,
    # and "crash" (drop the object without finalizing / committing).
    cp = EncodeCheckpointer.from_dir(work, bpr, resume=False)
    _append(cp, codec, docs[0:3])
    _append(cp, codec, docs[3:6])
    cp.checkpoint()                      # commit point: docs_done=6
    _append(cp, codec, docs[6:8])        # uncommitted batch
    cp.writer.flush_sync()               # force codes to disk, but no new checkpoint json
    cp.close_sidecars()                  # flushes doclens/pids sidecars to disk too

    # Pass 2: resume — must truncate the uncommitted rows back to the committed 6.
    cp2 = EncodeCheckpointer.from_dir(work, bpr, resume=True)
    assert cp2.docs_done == 6
    assert cp2.pids == ref_pids[:6]
    assert cp2.doclens == ref_doclens[:6]

    _append(cp2, codec, docs[6:9])
    _append(cp2, codec, docs[9:10])
    cids_path = tmp_path / "centroid_ids.npy"
    res_path = tmp_path / "packed_residuals.npy"
    doclens, pids, total = cp2.finalize(cids_path, res_path)

    assert pids == ref_pids
    assert doclens.tolist() == ref_doclens
    assert total == sum(ref_doclens)
    assert np.array_equal(np.load(cids_path), ref_cids)
    assert np.array_equal(np.load(res_path), ref_packed)

    # finalize() cleans up the checkpoint JSON and partial sidecars.
    assert not (work / "_encode_ckpt.json").exists()
    assert not (work / "doclens.partial.bin").exists()
    assert not (work / "pids.partial.txt").exists()


def test_fresh_then_finalize_matches_oneshot(tmp_path):
    """No interruption: streaming with checkpoints still matches one-shot encoding."""
    codec = _make_codec()
    bpr = codec.bytes_per_vector - 4
    docs = _make_docs(7, seed=1)

    ref_cids = np.concatenate([codec.encode(v)[0] for _, v in docs])
    ref_packed = np.concatenate([codec.encode(v)[1] for _, v in docs])

    cp = EncodeCheckpointer.from_dir(tmp_path / "work", bpr, resume=False)
    for s in range(0, len(docs), 2):
        _append(cp, codec, docs[s:s + 2])
        cp.checkpoint()
    doclens, pids, total = cp.finalize(
        tmp_path / "centroid_ids.npy", tmp_path / "packed_residuals.npy"
    )

    assert pids == [pid for pid, _ in docs]
    assert total == sum(v.shape[0] for _, v in docs)
    assert np.array_equal(np.load(tmp_path / "centroid_ids.npy"), ref_cids)
    assert np.array_equal(np.load(tmp_path / "packed_residuals.npy"), ref_packed)
