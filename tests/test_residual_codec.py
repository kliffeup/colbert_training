"""Tests for residual compression codec."""

import numpy as np
import torch
import pytest
from colbert.indexing.residual_codec import ResidualCodec


@pytest.fixture
def codec():
    centroids = torch.randn(64, 128)
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    c = ResidualCodec(centroids, nbits=2)
    # Set quantization params from random residuals
    sample_residuals = torch.randn(1000, 128) * 0.1
    c.set_quantization_params(sample_residuals)
    return c


def test_encode_decode_shape(codec):
    vectors = torch.randn(10, 128)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    cids, packed = codec.encode(vectors)
    assert cids.shape == (10,)
    assert cids.dtype == np.uint32
    assert packed.shape[0] == 10

    reconstructed = codec.decode(cids, packed)
    assert reconstructed.shape == (10, 128)


def test_encode_decode_quality(codec):
    """Reconstructed vectors should be close to originals."""
    vectors = torch.randn(100, 128)
    vectors = torch.nn.functional.normalize(vectors, dim=1)

    cids, packed = codec.encode(vectors)
    reconstructed = codec.decode(cids, packed)

    # Cosine similarity should be high
    cos_sim = torch.nn.functional.cosine_similarity(vectors, reconstructed, dim=1)
    assert cos_sim.mean() > 0.5


def test_bytes_per_vector(codec):
    assert codec.bytes_per_vector == 4 + 32  # 4 bytes centroid + 32 bytes for 2-bit residual


def test_1bit_codec():
    centroids = torch.randn(32, 128)
    codec = ResidualCodec(centroids, nbits=1)
    sample_residuals = torch.randn(500, 128) * 0.1
    codec.set_quantization_params(sample_residuals)

    vectors = torch.randn(10, 128)
    cids, packed = codec.encode(vectors)
    assert packed.shape == (10, 16)  # 128 / 8 = 16 bytes for 1-bit
    assert codec.bytes_per_vector == 4 + 16


def test_save_load(codec, tmp_path):
    path = str(tmp_path / "codec.pt")
    codec.save(path)
    loaded = ResidualCodec.load(path)

    assert loaded.nbits == codec.nbits
    assert loaded.centroids.shape == codec.centroids.shape
    assert torch.allclose(loaded.centroids, codec.centroids)
