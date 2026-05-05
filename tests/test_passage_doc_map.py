"""Tests for colbert.documents.passage_doc_map."""

import textwrap

import pytest

from colbert.documents.passage_doc_map import PassageDocMap, aggregate


@pytest.fixture
def pmap_file(tmp_path):
    p = tmp_path / "p2d.tsv"
    p.write_text(textwrap.dedent("""\
        D1_p0\tD1
        D1_p1\tD1
        D2_p0\tD2
        D3_p0\tD3
        D3_p1\tD3
        D3_p2\tD3
    """))
    return p


def test_load_and_lookup(pmap_file):
    pm = PassageDocMap.load(pmap_file)
    assert pm["D1_p0"] == "D1"
    assert pm["D3_p2"] == "D3"
    assert pm.get("missing") is None
    assert "D2_p0" in pm


def test_aggregate_keeps_max_per_doc(pmap_file):
    pm = PassageDocMap.load(pmap_file)
    ranking = {
        0: [
            ("D1_p0", 0.9),
            ("D1_p1", 0.7),
            ("D2_p0", 0.5),
            ("D3_p0", 0.6),
            ("D3_p1", 0.8),
            ("D3_p2", 0.1),
        ],
    }
    out = aggregate(ranking, pm, top_k=10)
    assert out[0][0] == ("D1", pytest.approx(0.9))   # max(0.9, 0.7)
    assert out[0][1] == ("D3", pytest.approx(0.8))   # max(0.6, 0.8, 0.1)
    assert out[0][2] == ("D2", pytest.approx(0.5))


def test_aggregate_truncates_to_top_k(pmap_file):
    pm = PassageDocMap.load(pmap_file)
    ranking = {0: [("D1_p0", 0.9), ("D2_p0", 0.5), ("D3_p0", 0.7)]}
    out = aggregate(ranking, pm, top_k=2)
    assert len(out[0]) == 2
    assert out[0][0][0] == "D1"
    assert out[0][1][0] == "D3"


def test_aggregate_skips_unknown_pids(pmap_file):
    pm = PassageDocMap.load(pmap_file)
    ranking = {0: [("D1_p0", 0.9), ("UNKNOWN_p0", 1.0), ("D2_p0", 0.5)]}
    out = aggregate(ranking, pm, top_k=10)
    docs = [d for d, _ in out[0]]
    assert "D1" in docs and "D2" in docs
    assert "UNKNOWN" not in docs


def test_aggregate_empty_query(pmap_file):
    pm = PassageDocMap.load(pmap_file)
    ranking = {0: []}
    out = aggregate(ranking, pm, top_k=10)
    assert out[0] == []
