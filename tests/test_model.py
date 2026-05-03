"""Tests for ColBERT model."""

import torch
import pytest
from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT


@pytest.fixture
def config():
    return ColBERTConfig(checkpoint="bert-base-uncased", dim=128, query_maxlen=32, doc_maxlen=64)


@pytest.fixture
def model(config):
    return ColBERT(config)


def test_model_init(model, config):
    assert model.linear.out_features == config.dim
    assert model.linear.in_features == 768


def test_query_encoding(model):
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)
    Q = model.query(ids, mask)
    assert Q.shape == (2, 32, 128)
    # Check L2 normalization
    norms = Q.norm(dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_doc_encoding(model):
    ids = torch.randint(0, 1000, (2, 64))
    mask = torch.ones(2, 64, dtype=torch.long)
    D, D_mask = model.doc(ids, mask)
    assert D.shape == (2, 64, 128)
    assert D_mask.shape == (2, 64)


def test_encode_queries(model):
    queries = ["what is machine learning", "how does retrieval work"]
    Q = model.encode_queries(queries)
    assert Q.shape[0] == 2
    assert Q.shape[2] == 128


def test_encode_docs(model):
    docs = ["Machine learning is a subfield of AI.", "Retrieval systems find relevant documents."]
    D, D_mask = model.encode_docs(docs)
    assert D.shape[0] == 2
    assert D.shape[2] == 128


def test_score(model):
    Q = torch.randn(2, 32, 128)
    D = torch.randn(2, 64, 128)
    D_mask = torch.ones(2, 64, dtype=torch.bool)
    scores = model.score(Q, D, D_mask)
    assert scores.shape == (2,)
