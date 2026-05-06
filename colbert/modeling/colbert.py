from __future__ import annotations

import logging
import string
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

from colbert.config import ColBERTConfig
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, setup_tokenizer
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

        # One tokenizer instance shared between query and doc sides so [Q]/[D]/field
        # markers are consistent in ID space.
        setup = setup_tokenizer(config)
        self.query_tokenizer = QueryTokenizer(config, setup=setup)
        self.doc_tokenizer = DocTokenizer(config, setup=setup)

        # If [Q]/[D] or field-marker tokens were added to the vocab (encoders without
        # [unused0]/[unused1] slots, or any field_markers), grow the embedding matrix.
        if setup.num_added_tokens > 0:
            self.bert.resize_token_embeddings(len(setup.tokenizer))
            logger.info(
                f"Resized encoder embeddings to fit {setup.num_added_tokens} new "
                f"special token(s) ([Q]/[D] markers + field markers)."
            )

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
        """Encode documents. Returns embeddings and a mask for valid (non-pad, non-punct,
        in-indexed-field) tokens.

        The mask combines:
          * attention_mask (drops padding)
          * punctuation_mask (drops punctuation when config.mask_punctuation)
          * indexed_token_mask (drops tokens outside config.indexed_fields when set)

        The encoder still sees the full sequence, so kept tokens remain context-aware
        of the masked-out body / fields. Masked positions are post-encoder hard-zeroed,
        so they contribute nothing to MaxSim, in-batch CE, or the index.
        """
        D = self.forward(input_ids, attention_mask)
        doc_mask = self.doc_tokenizer.punctuation_mask(input_ids).to(D.device)
        doc_mask = doc_mask & attention_mask.bool()
        doc_mask = doc_mask & self.doc_tokenizer.indexed_token_mask(input_ids).to(D.device)

        # Zero out punctuation/padding/out-of-field embeddings
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
