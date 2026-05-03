"""Tests for evaluation metrics."""

import pytest
from colbert.evaluation.metrics import (
    mrr_at_k,
    recall_at_k,
    success_at_k,
    ndcg_at_k,
    evaluate_ranking,
)


@pytest.fixture
def sample_data():
    ranking = {
        1: [(10, 5.0), (20, 4.0), (30, 3.0), (40, 2.0)],
        2: [(50, 5.0), (60, 4.0), (70, 3.0)],
        3: [(80, 5.0), (90, 4.0)],
    }
    qrels = {
        1: {10: 1, 30: 1},  # relevant at rank 1 and 3
        2: {60: 1},          # relevant at rank 2
        3: {100: 1},         # relevant doc not in ranking
    }
    return ranking, qrels


def test_mrr_at_10(sample_data):
    ranking, qrels = sample_data
    mrr = mrr_at_k(ranking, qrels, k=10)
    # q1: 1/1, q2: 1/2, q3: 0 -> (1 + 0.5 + 0) / 3 = 0.5
    assert abs(mrr - 0.5) < 1e-6


def test_recall_at_k(sample_data):
    ranking, qrels = sample_data
    recall = recall_at_k(ranking, qrels, k=4)
    # q1: 2/2=1.0, q2: 1/1=1.0, q3: 0/1=0.0 -> (1+1+0)/3
    assert abs(recall - 2.0 / 3) < 1e-6


def test_success_at_5(sample_data):
    ranking, qrels = sample_data
    success = success_at_k(ranking, qrels, k=5)
    # q1: yes, q2: yes, q3: no -> 2/3
    assert abs(success - 2.0 / 3) < 1e-6


def test_ndcg_at_k_perfect():
    ranking = {1: [(10, 5.0), (20, 4.0)]}
    qrels = {1: {10: 2, 20: 1}}
    ndcg = ndcg_at_k(ranking, qrels, k=10)
    assert abs(ndcg - 1.0) < 1e-6


def test_ndcg_at_k_imperfect():
    ranking = {1: [(20, 5.0), (10, 4.0)]}
    qrels = {1: {10: 2, 20: 1}}
    ndcg = ndcg_at_k(ranking, qrels, k=10)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_evaluate_ranking(sample_data):
    ranking, qrels = sample_data
    metrics = evaluate_ranking(ranking, qrels)
    assert "MRR@10" in metrics
    assert "Recall@50" in metrics
    assert "Recall@1000" in metrics
    assert "nDCG@10" in metrics
    assert "Success@5" in metrics
