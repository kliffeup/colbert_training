"""Tests for the streaming compressed-embedding writer and shard merge.

These exercise the disk-backed Stage 2+3 path without needing a GPU/model: codes
are produced by the real ResidualCodec, streamed through CompressedShardWriter, and
read back via the same ``.npy`` format the retriever/saver expect.
"""

import numpy as np
import torch

from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.compressed_writer import (
    CompressedShardWriter,
    merge_compressed_shards,
)


def _make_codec(nbits=2):
    centroids = torch.nn.functional.normalize(torch.randn(64, 128), dim=1)
    codec = ResidualCodec(centroids, nbits=nbits)
    codec.set_quantization_params(torch.randn(1000, 128) * 0.1)
    return codec


def test_empty_encode():
    """Empty / all-masked docs compress to empty arrays of the right shape."""
    codec = _make_codec()
    cids, packed = codec.encode(torch.empty(0, 128))
    assert cids.shape == (0,)
    assert cids.dtype == np.uint32
    assert packed.shape == (0, codec.bytes_per_vector - 4)
    assert packed.dtype == np.uint8


def test_stream_finalize_roundtrip(tmp_path):
    """Streaming codes batch-by-batch yields the same .npy as one-shot encoding."""
    codec = _make_codec()
    bpr = codec.bytes_per_vector - 4

    vecs = torch.nn.functional.normalize(torch.randn(2500, 128), dim=1)
    ref_cids, ref_packed = codec.encode(vecs)  # one-shot reference

    writer = CompressedShardWriter(tmp_path / "work", bpr)
    # Stream in uneven batches, including an empty one.
    for start, end in [(0, 1000), (1000, 1000), (1000, 2300), (2300, 2500)]:
        c, p = codec.encode(vecs[start:end])
        writer.append(c, p)

    cids_path = tmp_path / "centroid_ids.npy"
    res_path = tmp_path / "packed_residuals.npy"
    total = writer.finalize_npy(cids_path, res_path)

    assert total == 2500
    assert not writer.cids_bin.exists()  # temps cleaned up

    got_cids = np.load(cids_path)
    got_packed = np.load(res_path)
    assert np.array_equal(got_cids, ref_cids)
    assert np.array_equal(got_packed, ref_packed)
    assert got_cids.dtype == np.uint32 and got_packed.dtype == np.uint8


def test_merge_preserves_doc_order(tmp_path):
    """merge_compressed_shards lays out rows per the requested doc order."""
    codec = _make_codec()
    bpr = codec.bytes_per_vector - 4

    # Two shards, each with a few docs of varying length.
    shard_docs = {
        0: [("D2", 3), ("D0", 2)],   # rank 0: pids D2, D0
        1: [("D1", 4), ("D3", 1)],   # rank 1: pids D1, D3
    }

    rng = np.random.default_rng(0)
    all_vecs = {}  # pid -> (n,128) source vectors
    shard_bins = []
    for rank in (0, 1):
        writer = CompressedShardWriter(tmp_path, bpr, tag=f"_{rank}")
        ntok = 0
        for pid, n in shard_docs[rank]:
            v = torch.tensor(rng.standard_normal((n, 128)), dtype=torch.float32)
            v = torch.nn.functional.normalize(v, dim=1)
            all_vecs[pid] = v
            c, p = codec.encode(v)
            writer.append(c, p)
            ntok += n
        writer._close()
        shard_bins.append((writer.cids_bin, writer.res_bin, ntok))

    # Desired output order: sorted by pid -> D0, D1, D2, D3.
    # doc_order entries are (shard_idx, tok_start_in_shard, tok_len).
    doc_order = [
        (0, 3, 2),  # D0: shard 0, after D2's 3 tokens
        (1, 0, 4),  # D1: shard 1, first
        (0, 0, 3),  # D2: shard 0, first
        (1, 4, 1),  # D3: shard 1, after D1's 4 tokens
    ]
    total = sum(t for _, _, t in doc_order)

    cids_path = tmp_path / "centroid_ids.npy"
    res_path = tmp_path / "packed_residuals.npy"
    merge_compressed_shards(shard_bins, doc_order, cids_path, res_path, bpr, total)

    got_cids = np.load(cids_path)
    got_packed = np.load(res_path)

    # Reconstruct the expected concatenation in pid-sorted order and compare codes.
    expected_cids = []
    expected_packed = []
    for pid in ["D0", "D1", "D2", "D3"]:
        c, p = codec.encode(all_vecs[pid])
        expected_cids.append(c)
        expected_packed.append(p)
    expected_cids = np.concatenate(expected_cids)
    expected_packed = np.concatenate(expected_packed)

    assert np.array_equal(got_cids, expected_cids)
    assert np.array_equal(got_packed, expected_packed)
