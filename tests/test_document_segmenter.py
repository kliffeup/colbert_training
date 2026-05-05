"""Tests for colbert.documents.segmenter."""

import pytest

from colbert.documents.segmenter import segment


class _FakeTokenizer:
    """Minimal tokenizer that splits on whitespace and rejoins via spaces.

    Lets us assert window/stride math without pulling in HF.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


@pytest.fixture
def tok():
    return _FakeTokenizer()


def test_short_doc_returns_one_passage(tok):
    out = segment("a b c", tok, window=10, stride=5)
    assert out == ["a b c"]


def test_empty_doc_returns_empty_passage(tok):
    assert segment("", tok, window=4, stride=2) == [""]


def test_exact_window_no_overlap(tok):
    text = " ".join(str(i) for i in range(10))
    out = segment(text, tok, window=5, stride=5)
    assert out == ["0 1 2 3 4", "5 6 7 8 9"]


def test_overlapping_windows(tok):
    text = " ".join(str(i) for i in range(10))
    out = segment(text, tok, window=4, stride=2)
    # starts at 0, 2, 4, 6, 8
    assert out[0] == "0 1 2 3"
    assert out[1] == "2 3 4 5"
    assert out[2] == "4 5 6 7"
    assert out[3] == "6 7 8 9"
    # final window starting at 8 contains only "8 9" — but the loop terminates after
    # the start+window>=len step, which already included tokens 6-9
    assert len(out) == 4


def test_invalid_stride(tok):
    with pytest.raises(ValueError):
        segment("a b c", tok, window=4, stride=0)
    with pytest.raises(ValueError):
        segment("a b c", tok, window=4, stride=5)


def test_invalid_window(tok):
    with pytest.raises(ValueError):
        segment("a b c", tok, window=0, stride=1)
