"""Tests for MaxSim scoring."""

import torch
import pytest
from colbert.modeling.similarity import colbert_score, colbert_score_packed


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
