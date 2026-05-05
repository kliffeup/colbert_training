"""Tests for colbert.documents.formatter."""

import pytest

from colbert.documents.formatter import format_doc, SUPPORTED_STRATEGIES


@pytest.fixture
def row():
    return {
        "docid": "D42",
        "url": "https://example.com/x",
        "title": "Example title",
        "body": "Body text goes here.",
    }


def test_body_only(row):
    assert format_doc(row, "body_only") == "Body text goes here."


def test_title_body(row):
    assert format_doc(row, "title_body") == "Example title Body text goes here."


def test_url_title_body(row):
    out = format_doc(row, "url_title_body")
    assert out.startswith("https://example.com/x")
    assert "Example title" in out
    assert "Body text goes here." in out


def test_tagged_default_template(row):
    out = format_doc(row, "tagged", "<title>{title}</title><body>{body}</body>")
    assert out == "<title>Example title</title><body>Body text goes here.</body>"


def test_tagged_custom_template_with_url(row):
    out = format_doc(
        row,
        "tagged",
        "<u>{url}</u><t>{title}</t><b>{body}</b>",
    )
    assert "<u>https://example.com/x</u>" in out
    assert "<t>Example title</t>" in out
    assert "<b>Body text goes here.</b>" in out


def test_missing_fields_collapse():
    row = {"docid": "D1", "title": "", "body": "only body", "url": ""}
    assert format_doc(row, "title_body") == "only body"
    assert format_doc(row, "url_title_body") == "only body"


def test_strips_whitespace():
    row = {"title": "  hello  ", "body": "\n\nworld\n"}
    assert format_doc(row, "title_body") == "hello world"


def test_unknown_strategy_raises(row):
    with pytest.raises(ValueError):
        format_doc(row, "not_a_strategy")


def test_tagged_requires_template(row):
    with pytest.raises(ValueError):
        format_doc(row, "tagged", template=None)


def test_supported_strategies_listing():
    assert set(SUPPORTED_STRATEGIES) == {
        "body_only", "title_body", "url_title_body", "tagged",
    }
