from __future__ import annotations

import torch
import torch.nn.functional as F


def colbert_score(
    Q: torch.Tensor,
    D: torch.Tensor,
    D_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ColBERT MaxSim score between query and document embeddings.

    Args:
        Q: Query embeddings of shape (batch, query_len, dim).
        D: Document embeddings of shape (batch, doc_len, dim).
        D_mask: Optional boolean mask of shape (batch, doc_len) indicating valid
                document tokens. If None, all tokens are considered valid.

    Returns:
        Scores of shape (batch,).
    """
    # (batch, query_len, doc_len)
    sim = torch.bmm(Q, D.transpose(1, 2))

    if D_mask is not None:
        # Expand mask to (batch, 1, doc_len) and zero out invalid positions
        sim = sim * D_mask.unsqueeze(1).float()
        # Set masked positions to large negative so they never win the max
        sim[~D_mask.unsqueeze(1).expand_as(sim)] = -9999.0

    # MaxSim: for each query token, take the max similarity across doc tokens
    max_sim = sim.max(dim=2).values  # (batch, query_len)
    return max_sim.sum(dim=1)  # (batch,)


def colbert_score_packed(
    Q: torch.Tensor,
    D_packed: torch.Tensor,
    D_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute MaxSim scores for a query against multiple documents of varying lengths.

    Used during retrieval re-ranking where documents may have different numbers of tokens.

    Args:
        Q: Single query embeddings of shape (query_len, dim).
        D_packed: All document token embeddings packed into (total_tokens, dim).
        D_lengths: Number of tokens per document, shape (num_docs,).

    Returns:
        Scores of shape (num_docs,).
    """
    scores = []
    offset = 0
    for length in D_lengths.tolist():
        length = int(length)
        d_embs = D_packed[offset:offset + length]  # (doc_len, dim)
        # (query_len, doc_len)
        sim = Q @ d_embs.t()
        score = sim.max(dim=1).values.sum()
        scores.append(score)
        offset += length
    return torch.stack(scores)


def colbert_score_grouped(
    sims: torch.Tensor,
    segment_ids: torch.Tensor,
    n_docs: int,
) -> torch.Tensor:
    """Vectorized grouped MaxSim over a flat similarity matrix.

    Equivalent to :func:`colbert_score_packed` but takes the already-computed
    ``Q @ D.t()`` matrix and a per-column document assignment, so the reduction is a
    single ``scatter_reduce`` instead of a Python loop over documents. Used by the
    parallel retriever to fold each tile of candidate embeddings into per-document
    scores without materializing per-document slices.

    Args:
        sims: Similarity matrix of shape ``(qlen, n_cand)`` (``Q @ candidates.t()``).
        segment_ids: ``(n_cand,)`` int64 tensor mapping each candidate column to a
            dense document slot in ``[0, n_docs)``.
        n_docs: Number of distinct document slots.

    Returns:
        Scores of shape ``(n_docs,)``: for each doc, max over its columns per query
        token, then sum over query tokens.
    """
    qlen = sims.shape[0]
    # Prefill with the dtype's min (NOT 0) — similarities can be negative, and any
    # doc slot with no columns must not contribute a spurious 0 to the max.
    buf = sims.new_full((qlen, n_docs), torch.finfo(sims.dtype).min)
    idx = segment_ids.to(torch.int64).unsqueeze(0).expand(qlen, -1)  # (qlen, n_cand)
    buf.scatter_reduce_(1, idx, sims, reduce="amax", include_self=True)
    return buf.sum(dim=0)  # (n_docs,)
