from __future__ import annotations

import string
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from colbert.config import ColBERTConfig


def _resolve_marker(tok, fallback_token: str, fallback_marker: str) -> Tuple[int, int]:
    """Return (marker_token_id, num_tokens_added).

    Prefers an unused-vocab slot (e.g. `[unused0]` in BERT) when available; otherwise adds
    the fallback as a new special token. Returns the marker id and the number of *new*
    tokens added to the tokenizer (0 when the fallback existed).
    """
    fb_id = tok.convert_tokens_to_ids(fallback_token)
    unk = tok.unk_token_id
    if fb_id is not None and fb_id != unk:
        return fb_id, 0

    added = tok.add_special_tokens({"additional_special_tokens": [fallback_marker]})
    return tok.convert_tokens_to_ids(fallback_marker), added


class _SharedTokenizerState:
    """Lazily-built tokenizer + marker IDs shared between QueryTokenizer and DocTokenizer.

    Markers are resolved once per (checkpoint) so query/doc tokenizers agree on IDs even if
    we had to add tokens (which mutates the tokenizer state).
    """

    _cache: dict = {}

    @classmethod
    def get(cls, checkpoint: str):
        if checkpoint in cls._cache:
            return cls._cache[checkpoint]

        tok = AutoTokenizer.from_pretrained(checkpoint)
        q_id, q_added = _resolve_marker(tok, "[unused0]", "[Q]")
        d_id, d_added = _resolve_marker(tok, "[unused1]", "[D]")
        state = {
            "tokenizer": tok,
            "Q_marker_token_id": q_id,
            "D_marker_token_id": d_id,
            "num_added_tokens": q_added + d_added,
        }
        cls._cache[checkpoint] = state
        return state


class QueryTokenizer:
    """Tokenizes queries with [Q] marker and [MASK] padding to fixed length."""

    def __init__(self, config: ColBERTConfig):
        self.config = config
        state = _SharedTokenizerState.get(config.checkpoint)
        self.tok = state["tokenizer"]
        self.Q_marker_token_id = state["Q_marker_token_id"]
        self.num_added_tokens = state["num_added_tokens"]
        self.mask_token_id = self.tok.mask_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

    def tokenize(
        self, queries: List[str], maxlen: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        maxlen = maxlen or self.config.query_maxlen
        self._assert_within_model_max(maxlen)

        encoded = self.tok(
            queries,
            padding="max_length",
            truncation=True,
            max_length=maxlen,
            return_tensors="pt",
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]

        # Insert [Q] marker after [CLS]: shift tokens right by 1
        # [CLS] query_tokens... [SEP] [PAD]... -> [CLS] [Q] query_tokens... [SEP] [PAD]...
        ids[:, 1:] = ids[:, :-1].clone()
        ids[:, 1] = self.Q_marker_token_id
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 1] = 1

        # Replace [PAD] tokens with [MASK] for query augmentation
        pad_mask = ids == self.tok.pad_token_id
        ids[pad_mask] = self.mask_token_id
        mask[pad_mask] = 1

        return ids, mask

    def _assert_within_model_max(self, maxlen: int) -> None:
        cap = getattr(self.tok, "model_max_length", None)
        if cap and cap < 1_000_000 and maxlen > cap:
            raise ValueError(
                f"query maxlen={maxlen} exceeds tokenizer model_max_length={cap}."
            )


class DocTokenizer:
    """Tokenizes documents with [D] marker and optional punctuation masking."""

    def __init__(self, config: ColBERTConfig):
        self.config = config
        state = _SharedTokenizerState.get(config.checkpoint)
        self.tok = state["tokenizer"]
        self.D_marker_token_id = state["D_marker_token_id"]
        self.num_added_tokens = state["num_added_tokens"]
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

        if config.mask_punctuation:
            punct_ids = [
                self.tok.convert_tokens_to_ids(c) for c in list(string.punctuation)
            ]
            self.skiplist = {
                tid for tid in punct_ids if tid is not None and tid != self.tok.unk_token_id
            }
        else:
            self.skiplist = set()

        config.validate_doc_maxlen()

    def tokenize(
        self, docs: List[str], maxlen: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        maxlen = maxlen or self.config.doc_maxlen
        self._assert_within_model_max(maxlen)

        encoded = self.tok(
            docs,
            padding="longest",
            truncation=True,
            max_length=maxlen,
            return_tensors="pt",
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]

        # Insert [D] marker after [CLS]
        ids[:, 1:] = ids[:, :-1].clone()
        ids[:, 1] = self.D_marker_token_id
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 1] = 1

        return ids, mask

    def punctuation_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask that is True for non-punctuation, non-pad tokens."""
        mask = torch.ones(ids.shape, dtype=torch.bool).to(ids.device)
        for token_id in self.skiplist:
            mask &= ids != token_id
        if self.tok.pad_token_id is not None:
            mask &= ids != self.tok.pad_token_id
        return mask

    def _assert_within_model_max(self, maxlen: int) -> None:
        cap = getattr(self.tok, "model_max_length", None)
        if cap and cap < 1_000_000 and maxlen > cap:
            raise ValueError(
                f"doc maxlen={maxlen} exceeds tokenizer model_max_length={cap} for "
                f"checkpoint '{self.config.checkpoint}'. Use a longer-context encoder "
                f"or lower doc_maxlen."
            )
