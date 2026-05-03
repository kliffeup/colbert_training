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
