"""Phase 1 training data: streaming triples + tokenizing collator.

Triples TSV layout (one per line, tab-separated):
  ``query_text<TAB>positive<TAB>negative``

When a `Collection` is provided, the second/third columns are treated as docids and
resolved to text via the collection (this is the format written by
`scripts/preprocess_msmarco_docs.py`). When `collection=None`, the columns are
taken to already contain text (the original MS MARCO passage `triples.train.small.tsv`
format).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from colbert.config import ColBERTConfig
from colbert.dataset.collection import Collection
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, setup_tokenizer

logger = logging.getLogger(__name__)


def _count_lines(path: Path) -> int:
    """Fast line count for `num_lines` reporting (binary read, count `\\n`)."""
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


class StreamingTriplesDataset(IterableDataset):
    """Streaming dataset of `(query, positive_text, negative_text)` triples.

    Shards across DDP ranks and DataLoader workers via a modular line filter:
    each line is consumed by exactly one (rank, worker) pair. Optionally resolves
    docids to text via a `Collection`. Optionally shuffles through a fixed-size
    in-memory buffer.

    Args:
        path: Path to the triples TSV.
        collection: When provided, columns 2/3 are treated as docids and resolved.
        shuffle_buffer_size: 0 disables shuffling (file is assumed pre-shuffled).
        seed: RNG seed mixed with epoch for reproducible shuffling.
    """

    def __init__(
        self,
        path: str,
        collection: Collection | None = None,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Triples file not found: {path}")
        self.collection = collection
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self._epoch = 0
        self.num_lines = _count_lines(self.path)
        logger.info(
            f"StreamingTriplesDataset: {self.num_lines:,} lines in {self.path}"
            + (" (resolving docids via Collection)" if collection is not None else "")
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _shard_info(self) -> Tuple[int, int]:
        """Return ``(my_global_worker_id, total_global_workers)``."""
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        return rank * num_workers + worker_id, world_size * num_workers

    def _resolve(self, q: str, pos: str, neg: str) -> Tuple[str, str, str] | None:
        if self.collection is None:
            return q, pos, neg
        try:
            return q, self.collection[pos], self.collection[neg]
        except KeyError:
            return None

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        my_id, total = self._shard_info()
        rng = random.Random(self.seed + self._epoch * 1_000_003 + my_id)
        buf: List[Tuple[str, str, str]] = []

        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f):
                if line_idx % total != my_id:
                    continue
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                triple = self._resolve(parts[0], parts[1], parts[2])
                if triple is None:
                    continue

                if self.shuffle_buffer_size <= 1:
                    yield triple
                    continue

                if len(buf) < self.shuffle_buffer_size:
                    buf.append(triple)
                else:
                    idx = rng.randrange(self.shuffle_buffer_size)
                    out, buf[idx] = buf[idx], triple
                    yield out

        rng.shuffle(buf)
        for triple in buf:
            yield triple


class TriplesCollator:
    """Tokenize a list of (query, pos, neg) triples into a Phase-1 training batch.

    Output keys: ``Q_ids``, ``Q_mask``, ``D_pos_ids``, ``D_pos_mask``,
    ``D_neg_ids``, ``D_neg_mask`` — all on CPU; trainer moves them to the device.
    """

    def __init__(self, config: ColBERTConfig):
        self.config = config
        setup = setup_tokenizer(config)
        self.query_tokenizer = QueryTokenizer(config, setup=setup)
        self.doc_tokenizer = DocTokenizer(config, setup=setup)

    def __call__(self, batch: List[Tuple[str, str, str]]) -> dict:
        queries = [b[0] for b in batch]
        pos_docs = [b[1] for b in batch]
        neg_docs = [b[2] for b in batch]

        Q_ids, Q_mask = self.query_tokenizer.tokenize(queries)
        D_pos_ids, D_pos_mask = self.doc_tokenizer.tokenize(pos_docs)
        D_neg_ids, D_neg_mask = self.doc_tokenizer.tokenize(neg_docs)

        return {
            "Q_ids": Q_ids,
            "Q_mask": Q_mask,
            "D_pos_ids": D_pos_ids,
            "D_pos_mask": D_pos_mask,
            "D_neg_ids": D_neg_ids,
            "D_neg_mask": D_neg_mask,
        }
