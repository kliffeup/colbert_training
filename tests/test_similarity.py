"""Tests for MaxSim scoring."""

import torch
import pytest
from colbert.modeling.similarity import (
    colbert_score,
    colbert_score_packed,
    colbert_score_grouped,
)


def test_colbert_score_shape():
    Q = torch.randn(4, 32, 128)
    D = torch.randn(4, 64, 128)
    scores = colbert_score(Q, D)
    assert scores.shape == (4,)


def test_colbert_score_with_mask():
    Q = torch.randn(2, 8, 16)
    D = torch.randn(2, 10, 16)
    D_mask = torch.ones(2, 10, dtype=torch.bool)
    D_mask[0, 8:] = False
    D_mask[1, 6:] = False

    scores = colbert_score(Q, D, D_mask)
    assert scores.shape == (2,)


def test_colbert_score_positive():
    """Identical Q and D should give high score."""
    Q = torch.randn(1, 4, 16)
    Q = torch.nn.functional.normalize(Q, dim=2)
    D = Q.clone()
    scores = colbert_score(Q, D)
    assert scores.item() > 0


def test_colbert_score_packed_basic():
    Q = torch.randn(8, 16)
    D1 = torch.randn(5, 16)
    D2 = torch.randn(3, 16)
    D_packed = torch.cat([D1, D2], dim=0)
    D_lengths = torch.tensor([5, 3])

    scores = colbert_score_packed(Q, D_packed, D_lengths)
    assert scores.shape == (2,)


def test_colbert_score_grouped_matches_loop():
    """Grouped scatter reduction equals the per-doc max-then-sum loop."""
    torch.manual_seed(0)
    qlen, dim = 6, 16
    lengths = [3, 1, 4, 2]  # includes a single-token doc
    seg = torch.tensor([i for i, L in enumerate(lengths) for _ in range(L)])
    Q = torch.randn(qlen, dim)
    D = torch.randn(seg.numel(), dim)
    sims = Q @ D.t()

    ref, off = [], 0
    for L in lengths:
        ref.append(sims[:, off:off + L].max(dim=1).values.sum())
        off += L
    ref = torch.stack(ref)

    got = colbert_score_grouped(sims, seg, n_docs=len(lengths))
    assert torch.allclose(got, ref, atol=1e-5)


def test_colbert_score_grouped_all_negative():
    """A query token whose similarities are all negative must not clamp to 0."""
    sims = -torch.rand(4, 5) - 0.5  # strictly negative
    seg = torch.tensor([0, 0, 1, 1, 1])
    ref, off, lengths = [], 0, [2, 3]
    for L in lengths:
        ref.append(sims[:, off:off + L].max(dim=1).values.sum())
        off += L
    got = colbert_score_grouped(sims, seg, n_docs=2)
    assert torch.allclose(got, torch.stack(ref), atol=1e-5)
    assert (got < 0).all()
