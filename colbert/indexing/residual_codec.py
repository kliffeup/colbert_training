"""Residual compression codec: centroid + quantized residual encoding."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


class ResidualCodec:
    """Encodes/decodes vectors as (centroid_id, quantized_residual) pairs.

    Each 128-dim vector is stored as:
    - centroid_id: 4 bytes (uint32) — index of nearest centroid
    - quantized_residual: nbits * dim / 8 bytes — quantized per-dimension residual

    For nbits=2, dim=128: 4 + 32 = 36 bytes per vector (vs 256 bytes at fp16).
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        nbits: int = 2,
    ):
        """
        Args:
            centroids: Cluster centroids of shape (num_centroids, dim).
            nbits: Bits per dimension for residual quantization (1 or 2).
        """
        self.centroids = centroids.float()
        self.num_centroids = centroids.shape[0]
        self.dim = centroids.shape[1]
        self.nbits = nbits

        # Precompute quantization buckets
        # For each centroid, we need to know the range of residuals to set bucket boundaries
        # We use a simple uniform quantization over a fixed range
        self.num_levels = 2 ** nbits  # 2 for 1-bit, 4 for 2-bit
        self._bucket_boundaries: torch.Tensor | None = None
        self._bucket_values: torch.Tensor | None = None

    def set_quantization_params(self, sample_residuals: torch.Tensor) -> None:
        """Compute quantization bucket boundaries from sample residuals.

        Args:
            sample_residuals: Sample residuals of shape (n_samples, dim) to estimate range.
        """
        # Per-dimension quantile-based buckets
        boundaries = torch.zeros(self.dim, self.num_levels - 1)
        values = torch.zeros(self.dim, self.num_levels)

        for d in range(self.dim):
            col = sample_residuals[:, d].sort().values
            n = len(col)
            for b in range(self.num_levels - 1):
                idx = int((b + 1) / self.num_levels * n)
                boundaries[d, b] = col[min(idx, n - 1)]
            # Bucket representative values (midpoints)
            for b in range(self.num_levels):
                lo_idx = 0 if b == 0 else int(b / self.num_levels * n)
                hi_idx = n if b == self.num_levels - 1 else int((b + 1) / self.num_levels * n)
                values[d, b] = col[lo_idx:hi_idx].mean()

        self._bucket_boundaries = boundaries
        self._bucket_values = values

    def _ensure_quantization_params(self) -> None:
        if self._bucket_boundaries is None:
            # Fallback: uniform quantization over [-1, 1]
            levels = self.num_levels
            boundaries = torch.zeros(self.dim, levels - 1)
            values = torch.zeros(self.dim, levels)
            for b in range(levels - 1):
                boundaries[:, b] = -1.0 + 2.0 * (b + 1) / levels
            for b in range(levels):
                lo = -1.0 + 2.0 * b / levels
                hi = -1.0 + 2.0 * (b + 1) / levels
                values[:, b] = (lo + hi) / 2.0
            self._bucket_boundaries = boundaries
            self._bucket_values = values

    def encode(self, vectors: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Compress vectors to (centroid_ids, packed_residuals).

        Args:
            vectors: Float tensor of shape (n, dim), L2-normalized.

        Returns:
            centroid_ids: uint32 array of shape (n,).
            packed_residuals: uint8 array of shape (n, bytes_per_residual).
        """
        self._ensure_quantization_params()
        vectors = vectors.float()

        if vectors.shape[0] == 0:
            # Empty / all-masked document: nothing to compress.
            codes_per_byte = 8 // self.nbits
            num_bytes = math.ceil(self.dim / codes_per_byte)
            return (
                np.empty(0, dtype=np.uint32),
                np.empty((0, num_bytes), dtype=np.uint8),
            )

        # Find nearest centroids
        # (n, num_centroids)
        sims = vectors @ self.centroids.t()
        centroid_ids = sims.argmax(dim=1)  # (n,)

        # Compute residuals
        assigned_centroids = self.centroids[centroid_ids]  # (n, dim)
        residuals = vectors - assigned_centroids  # (n, dim)

        # Quantize residuals
        codes = self._quantize(residuals)  # (n, dim) with values in [0, num_levels)

        # Pack into bytes
        packed = self._pack_codes(codes)

        return centroid_ids.numpy().astype(np.uint32), packed

    def decode(
        self,
        centroid_ids: np.ndarray,
        packed_residuals: np.ndarray,
    ) -> torch.Tensor:
        """Decompress to approximate vectors.

        Args:
            centroid_ids: uint32 array of shape (n,).
            packed_residuals: uint8 array of shape (n, bytes_per_residual).

        Returns:
            Approximate vectors of shape (n, dim).
        """
        self._ensure_quantization_params()

        codes = self._unpack_codes(packed_residuals)  # (n, dim)
        residuals = self._dequantize(codes)  # (n, dim)

        cids = torch.from_numpy(centroid_ids.astype(np.int64))
        centroids = self.centroids[cids]  # (n, dim)

        return centroids + residuals

    def _quantize(self, residuals: torch.Tensor) -> torch.Tensor:
        """Quantize residuals to integer codes.

        ``code[i, d]`` = number of bucket boundaries that ``residuals[i, d]`` exceeds,
        i.e. its bucket index in ``[0, num_levels)``.  Vectorized equivalent of a
        per-dimension, per-boundary comparison loop.
        """
        assert self._bucket_boundaries is not None
        boundaries = self._bucket_boundaries.to(residuals.device)  # (dim, num_levels-1)
        # (n, dim, 1) > (1, dim, num_levels-1) -> (n, dim, num_levels-1)
        gt = residuals.unsqueeze(-1) > boundaries.unsqueeze(0)
        codes = gt.sum(dim=-1).to(torch.uint8)  # (n, dim)
        return codes

    def _dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        """Dequantize integer codes back to approximate residuals."""
        assert self._bucket_values is not None
        n = codes.shape[0]
        residuals = torch.zeros(n, self.dim)

        values = self._bucket_values  # (dim, num_levels)
        for d in range(self.dim):
            residuals[:, d] = values[d, codes[:, d].long()]

        return residuals

    def _pack_codes(self, codes: torch.Tensor) -> np.ndarray:
        """Pack quantization codes into bytes.

        For nbits=1: 8 codes per byte -> 128/8 = 16 bytes
        For nbits=2: 4 codes per byte -> 128/4 = 32 bytes
        """
        n = codes.shape[0]
        codes_np = codes.cpu().numpy().astype(np.uint8)

        codes_per_byte = 8 // self.nbits
        num_bytes = math.ceil(self.dim / codes_per_byte)

        # Group the dim codes into bytes of `codes_per_byte` codes each, packing code
        # j-within-byte at bit offset j*nbits. Pad the final partial byte with zeros.
        pad = num_bytes * codes_per_byte - self.dim
        if pad:
            codes_np = np.concatenate(
                [codes_np, np.zeros((n, pad), dtype=np.uint8)], axis=1
            )
        reshaped = codes_np.reshape(n, num_bytes, codes_per_byte)
        shifts = (np.arange(codes_per_byte, dtype=np.uint8) * self.nbits)
        packed = (reshaped << shifts).sum(axis=2).astype(np.uint8)

        return packed

    def _unpack_codes(self, packed: np.ndarray) -> torch.Tensor:
        """Unpack bytes back to quantization codes."""
        n = packed.shape[0]
        codes_per_byte = 8 // self.nbits
        mask = (1 << self.nbits) - 1

        codes = np.zeros((n, self.dim), dtype=np.uint8)
        for i in range(self.dim):
            byte_idx = i // codes_per_byte
            bit_offset = (i % codes_per_byte) * self.nbits
            codes[:, i] = (packed[:, byte_idx] >> bit_offset) & mask

        return torch.from_numpy(codes)

    @property
    def bytes_per_vector(self) -> int:
        """Total bytes per compressed vector."""
        codes_per_byte = 8 // self.nbits
        residual_bytes = math.ceil(self.dim / codes_per_byte)
        return 4 + residual_bytes  # 4 bytes for centroid_id

    def save(self, path: str) -> None:
        """Save codec state to disk."""
        state = {
            "centroids": self.centroids,
            "nbits": self.nbits,
            "bucket_boundaries": self._bucket_boundaries,
            "bucket_values": self._bucket_values,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "ResidualCodec":
        """Load codec state from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        codec = cls(state["centroids"], nbits=state["nbits"])
        codec._bucket_boundaries = state["bucket_boundaries"]
        codec._bucket_values = state["bucket_values"]
        return codec
