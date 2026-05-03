from __future__ import annotations

import string
from typing import List, Tuple

import torch
from transformers import BertTokenizerFast

from colbert.config import ColBERTConfig


class QueryTokenizer:
    """Tokenizes queries with [Q] marker and [MASK] padding to fixed length."""

    def __init__(self, config: ColBERTConfig):
        self.config = config
        self.tok = BertTokenizerFast.from_pretrained(config.checkpoint)
        self.Q_marker_token = "[unused0]"
        self.Q_marker_token_id = self.tok.convert_tokens_to_ids(self.Q_marker_token)
        self.mask_token_id = self.tok.mask_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

    def tokenize(
        self, queries: List[str], maxlen: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        maxlen = maxlen or self.config.query_maxlen

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


class DocTokenizer:
    """Tokenizes documents with [D] marker and optional punctuation masking."""

    def __init__(self, config: ColBERTConfig):
        self.config = config
        self.tok = BertTokenizerFast.from_pretrained(config.checkpoint)
        self.D_marker_token = "[unused1]"
        self.D_marker_token_id = self.tok.convert_tokens_to_ids(self.D_marker_token)
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id

        if config.mask_punctuation:
            self.skiplist = set(
                self.tok.convert_tokens_to_ids(list(string.punctuation))
            )
        else:
            self.skiplist = set()

    def tokenize(
        self, docs: List[str], maxlen: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        maxlen = maxlen or self.config.doc_maxlen

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
        mask &= ids != self.tok.pad_token_id
        return mask
