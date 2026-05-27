"""Phase 2 training data: distillation tuples + tokenizing collator.

Tuples JSONL layout (one example per line):
  ``{"qid": ..., "query": str, "pids": [pid, ...], "scores": [float, ...],
     "positive_idx": int}``

Length of ``pids`` and ``scores`` equals ``config.nway`` (set by
``colbert.distillation.build_tuples``). ``positive_idx`` points to which pid in the
list is the (labeled or surrogate) positive — used for in-batch CE loss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from colbert.config import ColBERTConfig
from colbert.dataset.collection import Collection
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, setup_tokenizer

logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """Map-style dataset over a tuples JSONL. Holds raw examples in memory and
    resolves passage text on-the-fly via the given Collection.
    """

    def __init__(self, tuples_path: str, collection: Collection):
        super().__init__()
        path = Path(tuples_path)
        if not path.exists():
            raise FileNotFoundError(f"Tuples file not found: {tuples_path}")
        self.collection = collection
        self.examples: List[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                self.examples.append(json.loads(line))
        if not self.examples:
            raise ValueError(f"No tuples loaded from {tuples_path}")
        self.nway = len(self.examples[0]["pids"])
        logger.info(
            f"DistillationDataset: {len(self.examples):,} tuples, nway={self.nway}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        passages = [self.collection[pid] for pid in ex["pids"]]
        return {
            "query": ex["query"],
            "passages": passages,
            "scores": ex["scores"],
            "positive_idx": ex["positive_idx"],
        }


class DistillationCollator:
    """Collate distillation examples into a batch.

    Each example contributes 1 query + nway passages. Documents from all examples
    are flattened to a single (bsz*nway,) sequence dimension; the distill trainer
    reshapes back to (bsz, nway, ...).

    Output keys: ``Q_ids``, ``Q_mask``, ``D_ids``, ``D_mask`` (flattened),
    ``teacher_scores`` (bsz, nway), ``positive_idxs`` (bsz,), ``nway`` (int).
    """

    def __init__(self, config: ColBERTConfig):
        self.config = config
        setup = setup_tokenizer(config)
        self.query_tokenizer = QueryTokenizer(config, setup=setup)
        self.doc_tokenizer = DocTokenizer(config, setup=setup)

    def __call__(self, batch: List[dict]) -> dict:
        queries = [ex["query"] for ex in batch]
        nway = len(batch[0]["passages"])
        flat_passages: List[str] = []
        for ex in batch:
            if len(ex["passages"]) != nway:
                raise ValueError(
                    f"All examples must have the same nway; got "
                    f"{len(ex['passages'])} vs expected {nway}."
                )
            flat_passages.extend(ex["passages"])

        Q_ids, Q_mask = self.query_tokenizer.tokenize(queries)
        D_ids, D_mask = self.doc_tokenizer.tokenize(flat_passages)

        teacher_scores = torch.tensor(
            [ex["scores"] for ex in batch], dtype=torch.float32
        )
        positive_idxs = torch.tensor(
            [ex["positive_idx"] for ex in batch], dtype=torch.long
        )

        return {
            "Q_ids": Q_ids,
            "Q_mask": Q_mask,
            "D_ids": D_ids,
            "D_mask": D_mask,
            "teacher_scores": teacher_scores,
            "positive_idxs": positive_idxs,
            "nway": nway,
        }
