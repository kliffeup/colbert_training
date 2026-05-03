"""Loss functions for ColBERTv2 training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_ce_loss(
    scores_pos: torch.Tensor,
    scores_neg: torch.Tensor,
) -> torch.Tensor:
    """Phase 1 pairwise softmax cross-entropy loss.

    Args:
        scores_pos: Positive passage scores, shape (batch,).
        scores_neg: Negative passage scores, shape (batch,).

    Returns:
        Scalar loss.
    """
    scores = torch.stack([scores_pos, scores_neg], dim=-1)  # (batch, 2)
    labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, labels)


def distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    """Phase 2 KL-Divergence distillation loss.

    Args:
        student_scores: ColBERT scores for nway passages, shape (batch, nway).
        teacher_scores: Cross-encoder scores for nway passages, shape (batch, nway).

    Returns:
        Scalar KL-Div loss.
    """
    student_log_probs = F.log_softmax(student_scores, dim=-1)
    teacher_probs = F.softmax(teacher_scores, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def in_batch_negative_loss(
    Q: torch.Tensor,
    D_pos: torch.Tensor,
    D_pos_mask: torch.Tensor,
) -> torch.Tensor:
    """In-batch cross-entropy negative loss.

    For each query, its positive passage score is compared against all other
    passages in the batch via cross-entropy.

    Args:
        Q: Query embeddings, shape (batch, query_len, dim).
        D_pos: Positive document embeddings, shape (batch, doc_len, dim).
        D_pos_mask: Positive document masks, shape (batch, doc_len).

    Returns:
        Scalar loss.
    """
    batch_size = Q.size(0)
    scores = torch.zeros(batch_size, batch_size, device=Q.device)

    for i in range(batch_size):
        q_i = Q[i].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, qlen, dim)
        # sim: (batch, qlen, dlen)
        sim = torch.bmm(q_i, D_pos.transpose(1, 2))
        if D_pos_mask is not None:
            sim = sim * D_pos_mask.unsqueeze(1).float()
            sim[~D_pos_mask.unsqueeze(1).expand_as(sim)] = -9999.0
        max_sim = sim.max(dim=2).values  # (batch, qlen)
        scores[i] = max_sim.sum(dim=1)  # (batch,)

    labels = torch.arange(batch_size, device=Q.device)
    return F.cross_entropy(scores, labels)
