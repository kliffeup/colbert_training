"""Token-level field membership mask for document encoding.

Given a tokenized document containing pairs of begin/end marker token IDs (e.g. produced
by formatting a doc as ``[TITLE_BEGIN]title text[TITLE_END][BODY_BEGIN]body text[BODY_END]``
and registering the marker tokens as special tokens in the tokenizer), this returns a bool
mask over token positions: ``True`` for positions that should contribute to the doc's
ColBERT embeddings (i.e. survive into the index and into late-interaction scoring),
``False`` for positions to zero out.

The mask is applied post-encoder, AND'd into the existing punctuation/padding mask. The
encoder still sees the full sequence, so kept tokens remain context-aware.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


def compute_indexed_mask(
    input_ids: torch.Tensor,
    field_marker_ids: Dict[str, Tuple[int, int]],
    indexed_fields: List[str],
    keep_marker_tokens: bool,
    keep_special_tokens: bool,
    special_token_ids: Iterable[int],
) -> torch.Tensor:
    """Compute a per-token mask of positions to keep based on field membership.

    Args:
        input_ids: Tensor of shape ``(batch, seq_len)`` (or ``(seq_len,)``).
        field_marker_ids: ``{field_name: (begin_id, end_id)}``.
        indexed_fields: Names of fields whose content positions are kept. Each name must
            be a key in ``field_marker_ids``. Empty list -> all positions kept.
        keep_marker_tokens: If True, the begin/end marker tokens of indexed fields are
            kept too. Otherwise they are masked out.
        keep_special_tokens: If True, ``special_token_ids`` (typically [CLS]/[SEP]/[D])
            are kept regardless of field membership.
        special_token_ids: IDs to always keep (only used when keep_special_tokens=True).

    Returns:
        Boolean tensor with the same shape as ``input_ids``.

    Behavior with truncated sequences (begin without matching end): everything from the
    begin token to the end of the sequence is treated as inside the field — graceful.
    Begin/end pairs do not need to be balanced; an end without a preceding begin is
    ignored. Multiple non-nested pairs are supported.
    """
    if input_ids.dim() == 1:
        squeeze = True
        input_ids = input_ids.unsqueeze(0)
    else:
        squeeze = False

    if not indexed_fields:
        out = torch.ones_like(input_ids, dtype=torch.bool)
        return out.squeeze(0) if squeeze else out

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for field in indexed_fields:
        if field not in field_marker_ids:
            raise ValueError(
                f"indexed_fields references '{field}' but it is not declared in "
                f"field_markers (declared: {sorted(field_marker_ids)})"
            )
        begin_id, end_id = field_marker_ids[field]
        if begin_id == end_id:
            raise ValueError(
                f"begin and end marker IDs collide for field '{field}'; "
                "use distinct marker tokens."
            )
        is_begin = (input_ids == begin_id)
        is_end = (input_ids == end_id)
        # depth[t] = (# begins seen up to and incl. t) - (# ends seen up to and incl. t).
        # depth > 0 -> position t is "inside" the field (includes begin token, excludes end).
        depth = is_begin.cumsum(dim=-1) - is_end.cumsum(dim=-1)
        in_field = depth > 0
        if keep_marker_tokens:
            in_field = in_field | is_end          # add the end token (begin already inside)
        else:
            in_field = in_field & ~is_begin       # drop the begin token
        mask = mask | in_field

    if keep_special_tokens:
        ids_list = list(special_token_ids)
        if ids_list:
            specials = torch.tensor(ids_list, device=input_ids.device, dtype=input_ids.dtype)
            mask = mask | torch.isin(input_ids, specials)

    return mask.squeeze(0) if squeeze else mask
