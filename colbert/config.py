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

    # Output paths
    output_dir: str = "experiments"
    index_path: str = "experiments/index"
    checkpoint_dir: str = "experiments/checkpoints"

    # Logging
    log_every: int = 500
    save_every: int = 10_000

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
