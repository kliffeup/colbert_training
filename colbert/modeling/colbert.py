from __future__ import annotations

import logging
import string
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

from colbert.config import ColBERTConfig
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.modeling.similarity import colbert_score


class ColBERT(nn.Module):
    """ColBERTv2 model: shared encoder + linear projection + L2 normalization.

    Encoder is loaded via AutoModel so any HF backbone (BERT, ModernBERT, DeBERTa, ...)
    can be plugged in via config.checkpoint.
    """

    def __init__(self, config: ColBERTConfig):
        super().__init__()
        self.config = config

        attn_impl = config.attn_implementation
        model_dtype = config.resolved_torch_dtype
        try:
            self.bert = AutoModel.from_pretrained(
                config.checkpoint,
                attn_implementation=attn_impl,
                torch_dtype=model_dtype,
            )
            logger.info(
                f"Loaded encoder '{config.checkpoint}' with attn_implementation='{attn_impl}', "
                f"torch_dtype={model_dtype}"
            )
        except (ValueError, ImportError) as e:
            logger.warning(
                f"Failed to load with attn_implementation='{attn_impl}': {e}. "
                f"Falling back to default attention."
            )
            self.bert = AutoModel.from_pretrained(
                config.checkpoint, torch_dtype=model_dtype
            )

        self.linear = nn.Linear(self.bert.config.hidden_size, config.dim, bias=False)

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)

        # If the tokenizer had to add new special tokens for [Q]/[D] markers (e.g. on
        # encoders without [unused0]/[unused1] slots), grow the embedding matrix to match.
        added = self.query_tokenizer.num_added_tokens
        if added > 0:
            self.bert.resize_token_embeddings(len(self.query_tokenizer.tok))
            logger.info(f"Resized encoder embeddings to fit {added} new marker token(s).")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        projected = self.linear(hidden)  # (batch, seq_len, dim)
        normalized = nn.functional.normalize(projected, p=2, dim=2)
        return normalized

    def query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode queries. All tokens (including [MASK]) are kept."""
        return self.forward(input_ids, attention_mask)

    def doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode documents. Returns embeddings and a mask for valid (non-pad, non-punct) tokens."""
        D = self.forward(input_ids, attention_mask)
        doc_mask = self.doc_tokenizer.punctuation_mask(input_ids).to(D.device)
        doc_mask = doc_mask & attention_mask.bool()

        # Zero out punctuation/padding embeddings
        D = D * doc_mask.unsqueeze(2).float()
        return D, doc_mask

    def score(
        self,
        Q: torch.Tensor,
        D: torch.Tensor,
        D_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ColBERT MaxSim scores."""
        return colbert_score(Q, D, D_mask)

    def encode_queries(self, queries: List[str], maxlen: int | None = None) -> torch.Tensor:
        """Tokenize and encode a batch of queries."""
        ids, mask = self.query_tokenizer.tokenize(queries, maxlen=maxlen)
        ids, mask = ids.to(self.device), mask.to(self.device)
        return self.query(ids, mask)

    def encode_docs(
        self, docs: List[str], maxlen: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode a batch of documents."""
        ids, mask = self.doc_tokenizer.tokenize(docs, maxlen=maxlen)
        ids, mask = ids.to(self.device), mask.to(self.device)
        return self.doc(ids, mask)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path: str) -> None:
        state = {
            "model_state_dict": self.state_dict(),
            "config": {
                "checkpoint": self.config.checkpoint,
                "dim": self.config.dim,
                "similarity": self.config.similarity,
                "mask_punctuation": self.config.mask_punctuation,
                "query_maxlen": self.config.query_maxlen,
                "doc_maxlen": self.config.doc_maxlen,
            },
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, config: ColBERTConfig | None = None) -> "ColBERT":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if config is None:
            config = ColBERTConfig(**state["config"])
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        return model
