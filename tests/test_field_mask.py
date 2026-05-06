"""Tests for colbert.documents.field_mask.compute_indexed_mask."""

import pytest
import torch

from colbert.documents.field_mask import compute_indexed_mask


# Fake token IDs for the test cases. Numbers are arbitrary but distinct.
T_BEG, T_END = 100, 101   # title markers
B_BEG, B_END = 200, 201   # body markers
CLS, SEP, D_MARK = 1, 2, 3
PAD = 0


def _ids(seq):
    return torch.tensor(seq, dtype=torch.long)


@pytest.fixture
def markers():
    return {"title": (T_BEG, T_END), "body": (B_BEG, B_END)}


def test_empty_indexed_fields_keeps_everything(markers):
    ids = _ids([CLS, D_MARK, T_BEG, 10, 11, T_END, B_BEG, 20, B_END, SEP])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=[],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    assert out.tolist() == [True] * len(ids)


def test_title_only_drops_body_and_outside(markers):
    # [CLS] [D] [T_BEG] 10 11 [T_END] [B_BEG] 20 [B_END] [SEP]
    ids = _ids([CLS, D_MARK, T_BEG, 10, 11, T_END, B_BEG, 20, B_END, SEP])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[CLS, SEP, D_MARK],
    )
    # Only the two content tokens of the title (positions 3, 4) are kept.
    assert out.tolist() == [
        False, False, False, True, True, False, False, False, False, False
    ]


def test_keep_special_tokens(markers):
    ids = _ids([CLS, D_MARK, T_BEG, 10, 11, T_END, B_BEG, 20, B_END, SEP])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=False,
        keep_special_tokens=True,
        special_token_ids=[CLS, SEP, D_MARK],
    )
    # CLS, [D], title content (10, 11), SEP all True; markers and body False.
    assert out.tolist() == [
        True, True, False, True, True, False, False, False, False, True
    ]


def test_keep_marker_tokens(markers):
    ids = _ids([CLS, D_MARK, T_BEG, 10, 11, T_END, B_BEG, 20, B_END, SEP])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=True,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    # Title's begin and end and content kept; everything else dropped.
    assert out.tolist() == [
        False, False, True, True, True, True, False, False, False, False
    ]


def test_multiple_indexed_fields(markers):
    ids = _ids([CLS, T_BEG, 10, T_END, B_BEG, 20, 21, B_END, SEP])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title", "body"],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    # Both title content and body content kept.
    assert out.tolist() == [
        False, False, True, False, False, True, True, False, False
    ]


def test_truncated_open_field_extends_to_end(markers):
    # [T_BEG] is present but [T_END] was truncated off.
    ids = _ids([CLS, T_BEG, 10, 11, 12])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    # Everything after T_BEG counts as inside the title (graceful truncation handling).
    assert out.tolist() == [False, False, True, True, True]


def test_unmatched_end_is_ignored(markers):
    # Stray [T_END] without preceding [T_BEG] -> contributes nothing.
    ids = _ids([CLS, T_END, 10, T_BEG, 20, T_END])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    # Only the 20 between the proper [T_BEG]/[T_END] pair is kept.
    assert out.tolist() == [False, False, False, False, True, False]


def test_batch_shape(markers):
    ids = _ids([
        [CLS, T_BEG, 10, T_END, SEP, PAD],
        [CLS, B_BEG, 20, 21, B_END, SEP],
    ])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    assert out.shape == ids.shape
    assert out.tolist() == [
        [False, False, True, False, False, False],
        [False, False, False, False, False, False],
    ]


def test_unknown_indexed_field_raises(markers):
    ids = _ids([CLS, 10])
    with pytest.raises(ValueError, match="not declared"):
        compute_indexed_mask(
            input_ids=ids,
            field_marker_ids=markers,
            indexed_fields=["nope"],
            keep_marker_tokens=False,
            keep_special_tokens=False,
            special_token_ids=[],
        )


def test_collision_begin_end_raises():
    ids = _ids([CLS, 10])
    with pytest.raises(ValueError, match="collide"):
        compute_indexed_mask(
            input_ids=ids,
            field_marker_ids={"title": (50, 50)},
            indexed_fields=["title"],
            keep_marker_tokens=False,
            keep_special_tokens=False,
            special_token_ids=[],
        )


def test_1d_input_returns_1d(markers):
    ids = _ids([CLS, T_BEG, 10, T_END])
    out = compute_indexed_mask(
        input_ids=ids,
        field_marker_ids=markers,
        indexed_fields=["title"],
        keep_marker_tokens=False,
        keep_special_tokens=False,
        special_token_ids=[],
    )
    assert out.dim() == 1
    assert out.shape == ids.shape
