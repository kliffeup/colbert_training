from __future__ import annotations

import os
import yaml
import torch
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass
class ColBERTConfig:
    # Model
    checkpoint: str = "bert-base-uncased"
    dim: int = 128
    similarity: str = "cosine"
    mask_punctuation: bool = True
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"

    # Tokenization
    query_maxlen: int = 32
    doc_maxlen: int = 180

    # Document training
    task: str = "passage"                    # "passage" | "document"
    doc_segmentation: str = "none"           # "none" (end-to-end) | "maxp" (sliding-window passages)
    passage_window: int = 180                # tokens per window in maxp mode (independent of encoder maxlen)
    passage_stride: int = 90
    max_passages_per_doc_factor: int = 4     # K' = retrieve_top_k * factor when aggregating maxp passages → docs
    field_format: str = "title_body"         # "body_only" | "title_body" | "url_title_body" | "tagged"
    field_format_template: str = "<title>{title}</title><body>{body}</body>"

    # Field-level masking
    # Map from field name -> [begin_marker, end_marker]. Each marker is registered as an
    # additional special token so it tokenizes to exactly one ID and never merges with
    # adjacent text. Example:
    #   {"title": ["[TITLE_BEGIN]", "[TITLE_END]"], "body": ["[BODY_BEGIN]", "[BODY_END]"]}
    field_markers: dict = field(default_factory=dict)
    # Fields whose token positions should be kept in the doc embeddings. Empty -> no
    # field-based filtering (all non-pad, non-punct tokens contribute, current behavior).
    indexed_fields: list = field(default_factory=list)
    # Whether the marker tokens themselves (begin/end) count as kept positions.
    index_field_markers: bool = False
    # Whether [CLS] / [SEP] / [D] are kept regardless of field membership. Recommended True.
    index_special_tokens: bool = True

    # Training — Phase 1 (triples)
    bsize: int = 32
    accumsteps: int = 1
    lr: float = 1e-5
    warmup: int = 20_000
    maxsteps: int = 150_000

    # Training — Phase 2 (distillation)
    distill_bsize: int = 32
    distill_lr: float = 1e-5
    distill_warmup: int = 20_000
    distill_maxsteps: int = 400_000
    nway: int = 64
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_distill: int = 500
    distill_rounds: int = 2

    # Indexing
    nbits: int = 2
    kmeans_niters: int = 20
    index_checkpoint_every: int = 2000  # passages between encode-pass checkpoints (resume granularity)

    # Retrieval
    nprobe: int = 2
    ncandidates_factor: int = 4096
    retrieve_top_k: int = 1000

    # Data paths
    data_dir: str = "data"
    collection: str = "data/collection.tsv"
    queries_train: str = "data/queries.train.tsv"
    queries_dev: str = "data/queries.dev.small.tsv"
    qrels_train: str = "data/qrels.train.tsv"
    qrels_dev: str = "data/qrels.dev.small.tsv"
    triples: str = "data/triples.train.small.tsv"
    tuples_dir: str = "data/tuples"

    # Document data paths (used when task == "document")
    documents_dir: str = "data/docs"
    passage_to_doc_map: str = "data/docs/passage_to_doc.tsv"

    # Output paths
    output_dir: str = "experiments"
    index_path: str = "experiments/index"
    checkpoint_dir: str = "experiments/checkpoints"

    # Logging
    log_every: int = 500
    save_every: int = 10_000
    save_total_limit: int = -1  # -1 = keep all step-checkpoints; N>0 keeps the N most recent

    # Wandb
    wandb_enabled: bool = False
    wandb_project: str = "colbertv2"
    wandb_entity: str = ""
    wandb_run_name: str = ""

    # BEIR / LoTTE
    beir_data_dir: str = "data/beir"
    lotte_data_dir: str = "data/lotte"
    beir_doc_maxlen: int = 300
    lotte_doc_maxlen: int = 300

    @property
    def ncandidates(self) -> int:
        return self.nprobe * self.ncandidates_factor

    def validate_doc_maxlen(self) -> None:
        """Ensure doc_maxlen does not exceed the tokenizer's model_max_length.

        Called lazily from sites that already construct the tokenizer (e.g. DocTokenizer)
        to avoid pulling transformers into config.py at import time.
        """
        if self.task != "document" or self.doc_segmentation != "none":
            return
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.checkpoint)
        cap = getattr(tok, "model_max_length", None)
        if cap and cap < 1_000_000 and self.doc_maxlen > cap:
            raise ValueError(
                f"doc_maxlen={self.doc_maxlen} exceeds tokenizer model_max_length={cap} "
                f"for checkpoint '{self.checkpoint}'. Lower doc_maxlen or pick a longer-context encoder."
            )

    @property
    def resolved_torch_dtype(self) -> torch.dtype:
        dtype = DTYPE_MAP.get(self.torch_dtype.lower())
        if dtype is None:
            raise ValueError(
                f"Unknown torch_dtype '{self.torch_dtype}'. "
                f"Supported values: {list(DTYPE_MAP.keys())}"
            )
        return dtype

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ColBERTConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def override(self, **kwargs) -> "ColBERTConfig":
        """Return a new config with the given overrides applied."""
        data = {fld.name: getattr(self, fld.name) for fld in fields(self)}
        data.update(kwargs)
        return ColBERTConfig(**data)

    def save(self, path: str | Path) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {fld.name: getattr(self, fld.name) for fld in fields(self)}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
