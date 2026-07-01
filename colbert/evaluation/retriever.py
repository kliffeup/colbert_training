"""ColBERTv2 retriever: centroid-based candidate generation + exact MaxSim re-ranking."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.saver import IndexSaver
from colbert.evaluation.parallel_scorer import ParallelMaxSimScorer

logger = logging.getLogger(__name__)


class ColBERTRetriever:
    """ColBERTv2 retriever using centroid-based inverted lists."""

    def __init__(
        self,
        model: ColBERT,
        index_path: str,
        config: ColBERTConfig,
        num_scorer_gpus: int | None = None,
        tile_size: int = 200_000,
    ):
        """
        Args:
            model: Loaded ColBERT model (used to encode queries on the master).
            index_path: Directory holding the built index.
            config: ColBERTConfig with retrieval knobs.
            num_scorer_gpus: GPUs to fan MaxSim scoring across. Defaults to all visible
                CUDA devices. When >1, a :class:`ParallelMaxSimScorer` pool is used and
                the master does NOT load the (multi-hundred-GB) compressed arrays —
                workers memory-map their shards. When <=1, scoring runs in-process.
            tile_size: Candidate embeddings decoded per tile inside each worker.
        """
        self.model = model
        self.config = config
        self.model.eval()

        saver = IndexSaver(index_path)
        self.codec = saver.load_codec()
        self.doclens = saver.load_doclens()
        self.pids = saver.load_pids()
        self.inverted_lists = saver.load_inverted_lists()
        self.metadata = saver.load_metadata()

        # Build doc_idx -> [start_emb, end_emb) mapping (cheap; needed by both paths).
        self._doc_offsets = np.zeros(len(self.doclens) + 1, dtype=np.int64)
        np.cumsum(self.doclens, out=self._doc_offsets[1:])

        if num_scorer_gpus is None:
            num_scorer_gpus = torch.cuda.device_count()

        # Populated only on the in-process (single-device) path.
        self.scorer: ParallelMaxSimScorer | None = None
        self.centroid_ids: np.ndarray | None = None
        self.packed_residuals: np.ndarray | None = None
        self._emb_to_doc: np.ndarray | None = None

        if num_scorer_gpus and num_scorer_gpus > 1:
            # Parallel path: candidate generation (below) needs only the codec +
            # inverted lists; workers memory-map the compressed shards themselves.
            self.scorer = ParallelMaxSimScorer(
                index_path=str(index_path),
                codec_path=str(saver.index_dir / "codec.pt"),
                doc_offsets=self._doc_offsets,
                pids=self.pids,
                config=config,
                world_size=int(num_scorer_gpus),
                device_type="cuda",
                tile_size=tile_size,
            )
            logger.info(
                f"Loaded index: {len(self.pids)} passages, "
                f"{len(self.inverted_lists)} centroids "
                f"(parallel MaxSim across {num_scorer_gpus} GPUs)"
            )
        else:
            # In-process path: load compressed embeddings + embedding->doc map.
            self.centroid_ids, self.packed_residuals = saver.load_compressed_embeddings()
            self._emb_to_doc = np.zeros(len(self.centroid_ids), dtype=np.int32)
            offset = 0
            for doc_idx, doclen in enumerate(self.doclens):
                self._emb_to_doc[offset:offset + doclen] = doc_idx
                offset += doclen
            logger.info(
                f"Loaded index: {len(self.pids)} passages, "
                f"{len(self.centroid_ids)} embeddings, "
                f"{len(self.inverted_lists)} centroids"
            )

    def close(self) -> None:
        """Tear down the scoring worker pool, if any. Idempotent."""
        if self.scorer is not None:
            self.scorer.close()

    def __enter__(self) -> "ColBERTRetriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass

    @torch.no_grad()
    def retrieve(
        self,
        queries: List[str],
        top_k: int | None = None,
        query_maxlen: int | None = None,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Retrieve top-k passages for each query.

        Args:
            queries: List of query texts.
            top_k: Number of passages to return per query.
            query_maxlen: Override for query max length.

        Returns:
            Dict mapping query index -> [(pid, score), ...] sorted descending.
            Pids are strings (e.g. "D123456" or "D123456_p7").
        """
        top_k = top_k or self.config.retrieve_top_k
        amp_dtype = self.config.resolved_torch_dtype
        results: Dict[int, List[Tuple[str, float]]] = {}

        for q_idx, query in enumerate(tqdm(queries, desc="Retrieving")):
            with autocast("cuda", dtype=amp_dtype):
                Q = self.model.encode_queries([query], maxlen=query_maxlen)  # (1, qlen, dim)
            Q = Q.squeeze(0)  # (qlen, dim)

            ranked = self._retrieve_single(Q, top_k)
            results[q_idx] = ranked

        return results

    def _retrieve_single(
        self,
        Q: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Retrieve for a single query.

        Steps 1-2 (candidate generation) run on the master; scoring (Steps 3-6) is
        delegated to the parallel worker pool when active, else to the in-process
        reference path.

        Args:
            Q: Query embeddings of shape (qlen, dim).
            top_k: Number of passages to return.

        Returns:
            [(pid, score), ...] sorted by score descending. Pids are strings.
        """
        nprobe = self.config.nprobe
        device = Q.device

        # Step 1: Find nprobe nearest centroids for each query token
        centroids = self.codec.centroids.to(device)
        # (qlen, num_centroids)
        centroid_sims = Q @ centroids.t()
        # (qlen, nprobe)
        _, top_centroid_ids = centroid_sims.topk(nprobe, dim=1)

        # Step 2: Collect candidate embedding IDs from inverted lists
        candidate_emb_ids = set()
        for q_tok in range(Q.shape[0]):
            for c in range(nprobe):
                cid = int(top_centroid_ids[q_tok, c].item())
                if cid in self.inverted_lists:
                    candidate_emb_ids.update(self.inverted_lists[cid])

        if not candidate_emb_ids:
            return []

        candidate_emb_ids_arr = np.array(sorted(candidate_emb_ids), dtype=np.int64)

        # Steps 3-6: score the candidate set (parallel pool or in-process fallback).
        if self.scorer is not None:
            return self.scorer.score(Q, candidate_emb_ids_arr, top_k)
        return self._retrieve_single_local(Q, top_k, candidate_emb_ids_arr)

    def _retrieve_single_local(
        self,
        Q: torch.Tensor,
        top_k: int,
        candidate_emb_ids_arr: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """In-process MaxSim scoring for a single query (single-device / reference path).

        This is the numerical reference the parallel scorer must match: approximate
        MaxSim over candidate embeddings (Steps 3-4), select top-``ncandidates`` docs
        (Step 5), then exact MaxSim re-rank (Step 6). Scores are computed in fp32.
        """
        assert self.centroid_ids is not None and self.packed_residuals is not None
        assert self._emb_to_doc is not None
        device = Q.device
        ncandidates = self.config.ncandidates
        Qf = Q.float()

        # Step 3: Decompress candidate embeddings and compute approximate scores
        cids = self.centroid_ids[candidate_emb_ids_arr]
        packed = self.packed_residuals[candidate_emb_ids_arr]
        decompressed = self.codec.decode(cids, packed).to(device)  # (n_candidates, dim) fp32

        # (qlen, n_candidates)
        sims = Qf @ decompressed.t()

        # Step 4: Group by passage, approximate MaxSim
        doc_indices = self._emb_to_doc[candidate_emb_ids_arr]
        unique_docs = np.unique(doc_indices)

        doc_scores: Dict[int, float] = {}
        for doc_idx in unique_docs:
            emb_mask = doc_indices == doc_idx
            # (qlen, n_doc_tokens)
            doc_sims = sims[:, emb_mask]
            # MaxSim: max per query token, then sum
            max_sim = doc_sims.max(dim=1).values
            doc_scores[int(doc_idx)] = max_sim.sum().item()

        # Step 5: Select top ncandidates for exact re-ranking
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_doc_idxs = [d[0] for d in sorted_docs[:ncandidates]]

        # Step 6: Exact re-ranking with full passage embeddings
        final_scores: List[Tuple[str, float]] = []
        for doc_idx in candidate_doc_idxs:
            start = int(self._doc_offsets[doc_idx])
            end = int(self._doc_offsets[doc_idx + 1])

            doc_cids = self.centroid_ids[start:end]
            doc_packed = self.packed_residuals[start:end]
            doc_embs = self.codec.decode(doc_cids, doc_packed).to(device)  # (doclen, dim) fp32

            # Exact MaxSim
            sim = Qf @ doc_embs.t()  # (qlen, doclen)
            score = sim.max(dim=1).values.sum().item()

            pid = str(self.pids[doc_idx])
            final_scores.append((pid, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
