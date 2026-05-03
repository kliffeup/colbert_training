"""Tests for tokenization."""

import torch
import pytest
from colbert.config import ColBERTConfig
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer


@pytest.fixture
def config():
    return ColBERTConfig(query_maxlen=16, doc_maxlen=32)


def test_query_tokenizer_shape(config):
    tok = QueryTokenizer(config)
    ids, mask = tok.tokenize(["hello world", "test query"])
    assert ids.shape == (2, 16)
    assert mask.shape == (2, 16)


def test_query_tokenizer_mask_padding(config):
    tok = QueryTokenizer(config)
    ids, mask = tok.tokenize(["hi"])
    # All positions should be attended (MASK tokens replace PAD)
    assert mask.sum() == config.query_maxlen


def test_query_tokenizer_q_marker(config):
    tok = QueryTokenizer(config)
    ids, _ = tok.tokenize(["test"])
    assert ids[0, 0].item() == tok.cls_token_id
    assert ids[0, 1].item() == tok.Q_marker_token_id


def test_doc_tokenizer_shape(config):
    tok = DocTokenizer(config)
    ids, mask = tok.tokenize(["hello world document text"])
    assert ids.shape[1] <= 32
    assert mask.shape == ids.shape


def test_doc_tokenizer_d_marker(config):
    tok = DocTokenizer(config)
    ids, _ = tok.tokenize(["test"])
    assert ids[0, 0].item() == tok.cls_token_id
    assert ids[0, 1].item() == tok.D_marker_token_id


def test_doc_tokenizer_punctuation_mask(config):
    tok = DocTokenizer(config)
    ids, _ = tok.tokenize(["hello, world! test."])
    pmask = tok.punctuation_mask(ids)
    assert pmask.dtype == torch.bool
    assert pmask.shape == ids.shape
