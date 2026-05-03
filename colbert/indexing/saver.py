"""Save/load ColBERTv2 index artifacts to/from disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from colbert.indexing.residual_codec import ResidualCodec

logger = logging.getLogger(__name__)


class IndexSaver:
    """Manages reading and writing of ColBERTv2 index files."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def save_codec(self, codec: ResidualCodec) -> None:
        codec.save(str(self.index_dir / "codec.pt"))

    def load_codec(self) -> ResidualCodec:
        return ResidualCodec.load(str(self.index_dir / "codec.pt"))

    def save_compressed_embeddings(
        self,
        centroid_ids: np.ndarray,
        packed_residuals: np.ndarray,
    ) -> None:
        np.save(self.index_dir / "centroid_ids.npy", centroid_ids)
        np.save(self.index_dir / "packed_residuals.npy", packed_residuals)

    def load_compressed_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        centroid_ids = np.load(self.index_dir / "centroid_ids.npy")
        packed_residuals = np.load(self.index_dir / "packed_residuals.npy")
        return centroid_ids, packed_residuals

    def save_doclens(self, doclens: np.ndarray) -> None:
        np.save(self.index_dir / "doclens.npy", doclens)

    def load_doclens(self) -> np.ndarray:
        return np.load(self.index_dir / "doclens.npy")

    def save_pids(self, pids: np.ndarray) -> None:
        np.save(self.index_dir / "pids.npy", pids)

    def load_pids(self) -> np.ndarray:
        return np.load(self.index_dir / "pids.npy")

    def save_inverted_lists(self, inverted_lists: Dict[int, List[int]]) -> None:
        # Store as dict of int -> list[int], serialized to a compact format
        str_dict = {str(k): v for k, v in inverted_lists.items()}
        with open(self.index_dir / "ivl.json", "w") as f:
            json.dump(str_dict, f)

    def load_inverted_lists(self) -> Dict[int, List[int]]:
        with open(self.index_dir / "ivl.json") as f:
            str_dict = json.load(f)
        return {int(k): v for k, v in str_dict.items()}

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        with open(self.index_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self) -> Dict[str, Any]:
        with open(self.index_dir / "metadata.json") as f:
            return json.load(f)
