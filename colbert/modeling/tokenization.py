from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from colbert.config import ColBERTConfig


def _resolve_marker(tok: PreTrainedTokenizerBase, fallback_token: str, fallback_marker: str) -> Tuple[int, int]:
    """Return ``(marker_token_id, num_tokens_added)``.

    Prefers an unused-vocab slot (e.g. ``[unused0]`` in BERT) when available; otherwise
    adds the fallback as a new special token.
    """
    fb_id = tok.convert_tokens_to_ids(fallback_token)
    unk = tok.unk_token_id
    if fb_id is not None and fb_id != unk:
        return fb_id, 0

    added = tok.add_special_tokens({"additional_special_tokens": [fallback_marker]})
    return tok.convert_tokens_to_ids(fallback_marker), added


@dataclass
class TokenizerSetup:
    """Result of registering [Q] / [D] / field-marker tokens on a tokenizer."""
    tokenizer: PreTrainedTokenizerBase
    Q_marker_id: int
    D_marker_id: int
    field_marker_ids: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    num_added_tokens: int = 0


def setup_tokenizer(config: ColBERTConfig) -> TokenizerSetup:
    """Build a fresh tokenizer for ``config.checkpoint`` and register all required markers.

    Single source of truth for [Q]/[D] markers and field-boundary markers — both Query
    and Doc tokenizers share the resulting tokenizer instance so token IDs stay aligned
    and the encoder only needs to be resized once.
    """
    tok = AutoTokenizer.from_pretrained(config.checkpoint)

    Q_id, q_added = _resolve_marker(tok, "[unused0]", "[Q]")
    D_id, d_added = _resolve_marker(tok, "[unused1]", "[D]")

    field_marker_ids: Dict[str, Tuple[int, int]] = {}
    field_added = 0
    if config.field_markers:
        flat: List[str] = []
        for fname, markers in config.field_markers.items():
            if not isinstance(markers, (list, tuple)) or len(markers) != 2:
                raise ValueError(
                    f"field_markers['{fname}'] must be [begin, end]; got {markers!r}"
                )
            flat.extend(markers)
        if len(set(flat)) != len(flat):
            raise ValueError(f"field_markers contains duplicate marker strings: {flat}")
        field_added = tok.add_special_tokens({"additional_special_tokens": flat})
        for fname, (begin, end) in config.field_markers.items():
            field_marker_ids[fname] = (
                tok.convert_tokens_to_ids(begin),
                tok.convert_tokens_to_ids(end),
            )

    return TokenizerSetup(
        tokenizer=tok,
        Q_marker_id=Q_id,
        D_marker_id=D_id,
        field_marker_ids=field_marker_ids,
        num_added_tokens=q_added + d_added + field_added,
    )


class QueryTokenizer:
    """Tokenizes queries with [Q] marker and [MASK] padding to fixed length."""

    def __init__(self, config: ColBERTConfig, setup: TokenizerSetup | None = None):
        self.config = config
        if setup is None:
            setup = setup_tokenizer(config)
        self.tok = setup.tokenizer
        self.Q_marker_token_id = setup.Q_marker_id
        self.num_added_tokens = setup.num_added_tokens
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
    """Tokenizes documents with [D] marker, optional punctuation masking, and optional
    field-level masking via configured begin/end marker tokens."""

    def __init__(self, config: ColBERTConfig, setup: TokenizerSetup | None = None):
        self.config = config
        if setup is None:
            setup = setup_tokenizer(config)
        self.tok = setup.tokenizer
        self.D_marker_token_id = setup.D_marker_id
        self.num_added_tokens = setup.num_added_tokens
        self.field_marker_ids: Dict[str, Tuple[int, int]] = setup.field_marker_ids
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

        if config.mask_punctuation:
            punct_ids = [self.tok.convert_tokens_to_ids(c) for c in list(string.punctuation)]
            self.skiplist = {
                tid for tid in punct_ids if tid is not None and tid != self.tok.unk_token_id
            }
        else:
            self.skiplist = set()

        # Validate indexed_fields against declared markers
        for fname in config.indexed_fields or []:
            if fname not in self.field_marker_ids:
                raise ValueError(
                    f"config.indexed_fields contains '{fname}' but it is not in "
                    f"config.field_markers (declared: {sorted(self.field_marker_ids)})"
                )

        # Tokens always kept when index_special_tokens=True
        specials: List[int] = [self.D_marker_token_id]
        if self.cls_token_id is not None:
            specials.append(self.cls_token_id)
        if self.sep_token_id is not None:
            specials.append(self.sep_token_id)
        self._special_keep_ids: List[int] = sorted(set(specials))

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

    def indexed_token_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Per-token boolean mask of positions kept under the configured indexed_fields.

        When ``config.indexed_fields`` is empty, this is all-True (no filtering). When
        non-empty, only tokens inside the declared field spans (plus optional special
        tokens) are True; tokens outside (e.g. body when ``indexed_fields=['title']``)
        are False so their post-encoder embeddings get zeroed and contribute nothing
        to MaxSim/loss/index.
        """
        # Local import to avoid coupling tokenizer module to a runtime helper.
        from colbert.documents.field_mask import compute_indexed_mask

        return compute_indexed_mask(
            input_ids=ids,
            field_marker_ids=self.field_marker_ids,
            indexed_fields=list(self.config.indexed_fields or []),
            keep_marker_tokens=bool(self.config.index_field_markers),
            keep_special_tokens=bool(self.config.index_special_tokens),
            special_token_ids=self._special_keep_ids,
        )

    def _assert_within_model_max(self, maxlen: int) -> None:
        cap = getattr(self.tok, "model_max_length", None)
        if cap and cap < 1_000_000 and maxlen > cap:
            raise ValueError(
                f"doc maxlen={maxlen} exceeds tokenizer model_max_length={cap} for "
                f"checkpoint '{self.config.checkpoint}'. Use a longer-context encoder "
                f"or lower doc_maxlen."
            )
