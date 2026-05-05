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

logger = logging.getLogger(__name__)


class ColBERTRetriever:
    """ColBERTv2 retriever using centroid-based inverted lists."""

    def __init__(
        self,
        model: ColBERT,
        index_path: str,
        config: ColBERTConfig,
    ):
        self.model = model
        self.config = config
        self.model.eval()

        saver = IndexSaver(index_path)
        self.codec = saver.load_codec()
        self.centroid_ids, self.packed_residuals = saver.load_compressed_embeddings()
        self.doclens = saver.load_doclens()
        self.pids = saver.load_pids()
        self.inverted_lists = saver.load_inverted_lists()
        self.metadata = saver.load_metadata()

        # Build embedding_id -> (doc_idx, token_offset) mapping
        self._emb_to_doc: np.ndarray = np.zeros(len(self.centroid_ids), dtype=np.int32)
        offset = 0
        for doc_idx, doclen in enumerate(self.doclens):
            self._emb_to_doc[offset:offset + doclen] = doc_idx
            offset += doclen

        # Build doc_idx -> (start_emb, end_emb) mapping
        self._doc_offsets = np.zeros(len(self.doclens) + 1, dtype=np.int64)
        np.cumsum(self.doclens, out=self._doc_offsets[1:])

        logger.info(
            f"Loaded index: {len(self.pids)} passages, "
            f"{len(self.centroid_ids)} embeddings, "
            f"{len(self.inverted_lists)} centroids"
        )

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

        Args:
            Q: Query embeddings of shape (qlen, dim).
            top_k: Number of passages to return.

        Returns:
            [(pid, score), ...] sorted by score descending. Pids are strings.
        """
        nprobe = self.config.nprobe
        ncandidates = self.config.ncandidates
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

        # Step 3: Decompress candidate embeddings and compute approximate scores
        cids = self.centroid_ids[candidate_emb_ids_arr]
        packed = self.packed_residuals[candidate_emb_ids_arr]
        decompressed = self.codec.decode(cids, packed).to(device)  # (n_candidates, dim)

        # (qlen, n_candidates)
        sims = Q @ decompressed.t()

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
            doc_embs = self.codec.decode(doc_cids, doc_packed).to(device)  # (doclen, dim)

            # Exact MaxSim
            sim = Q @ doc_embs.t()  # (qlen, doclen)
            score = sim.max(dim=1).values.sum().item()

            pid = str(self.pids[doc_idx])
            final_scores.append((pid, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
