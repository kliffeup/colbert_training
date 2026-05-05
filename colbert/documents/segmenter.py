"""Sliding-window document segmenter for MaxP-style training/indexing."""

from __future__ import annotations

from typing import List

from transformers import PreTrainedTokenizerBase


def segment(
    formatted_text: str,
    tokenizer: PreTrainedTokenizerBase,
    window: int,
    stride: int,
) -> List[str]:
    """Split text into overlapping passages.

    Tokenizes once, slides a window of `window` tokens with step `stride`, and decodes each
    slice back to text. The decoded passages are what downstream tokenization consumes — a
    second tokenization pass is fine because MaxP windows are well below model_max_length.

    Args:
        formatted_text: Already-formatted text (apply document_formatter.format_doc first).
        tokenizer: Any HF tokenizer; only its encode/decode methods are used.
        window: Tokens per passage. Must be > 0.
        stride: Step between window starts. Must be > 0 and <= window.

    Returns:
        List of passage texts; always non-empty (a doc shorter than `window` returns one
        passage equal to the full text).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if stride <= 0 or stride > window:
        raise ValueError(f"stride must be in (0, window]; got stride={stride}, window={window}")

    ids = tokenizer.encode(formatted_text, add_special_tokens=False)
    if not ids:
        return [""]

    passages: List[str] = []
    for start in range(0, len(ids), stride):
        slice_ids = ids[start : start + window]
        if not slice_ids:
            break
        passages.append(tokenizer.decode(slice_ids, skip_special_tokens=True))
        if start + window >= len(ids):
            break

    return passages or [""]
