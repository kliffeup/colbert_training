#!/usr/bin/env python3
"""Phase 1 batch-size smoke test.

Estimate the maximum Phase-1 batch size that fits on the current GPU for a given
config, by running real forward + backward + optimizer.step iterations on
synthetic input tensors at the configured ``query_maxlen`` / ``doc_maxlen``.

The script mirrors ``train_phase1``'s setup: same encoder, same dtype, same
attention implementation, same optimizer (AdamW), same loss (pairwise CE on a
positive + negative doc), same autocast/GradScaler. It does NOT touch disk
data — token IDs are sampled at random in the encoder's vocab range so we
exercise the full forward+backward graph at the largest possible sequence
length, regardless of whether your training data is downloaded.

The search starts at ``--start``, doubles until the first OOM, then binary
searches between the largest successful size and the first OOM to pin down the
true ceiling.

Usage:
    python scripts/smoke_test_batch_size.py --config configs/document_e2e_modernbert.yaml
    python scripts/smoke_test_batch_size.py --config <cfg> --start 1 --max 128 --steps 3
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.amp import autocast, GradScaler

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.training.loss import pairwise_ce_loss

logger = logging.getLogger("smoke_test_batch_size")


def _make_random_batch(
    bsize: int,
    seqlen: int,
    vocab_size: int,
    cls_id: int,
    sep_id: int,
    marker_id: int,
    device: torch.device,
    field_marker_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, attention_mask) of shape (bsize, seqlen) with realistic layout.

    Layout: ``[CLS] [Q|D] <random tokens> [SEP]``. Attention mask is all-ones to force
    the worst-case full-length encoder pass. Random IDs are sampled below ``vocab_size``
    and below the special-token IDs to avoid colliding with [CLS]/[SEP]/[MASK]/markers.
    Optional ``field_marker_ids`` are sprinkled in so the indexed-field code path in
    ``ColBERT.doc()`` has something to find — without this, the field mask zeros the
    whole doc on configs like document_e2e_modernbert.yaml (cosmetic only; memory
    profile is unchanged either way).
    """
    safe_upper = max(1, vocab_size - max(50, marker_id + 1))
    ids = torch.randint(
        low=10, high=safe_upper, size=(bsize, seqlen), device=device, dtype=torch.long
    )
    ids[:, 0] = cls_id
    ids[:, 1] = marker_id
    ids[:, -1] = sep_id

    if field_marker_ids and seqlen >= 6:
        # Plant one begin marker near the start and the matching end near the end so
        # any field-mask cumsum produces a non-empty "in-field" region.
        for begin_id, end_id in field_marker_ids:
            ids[:, 2] = begin_id
            ids[:, -2] = end_id
            break  # one field is enough to exercise the code path

    mask = torch.ones_like(ids, dtype=torch.long)
    return ids, mask


def _trial(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    amp_dtype: torch.dtype,
    config: ColBERTConfig,
    bsize: int,
    steps: int,
    device: torch.device,
    vocab_size: int,
    cls_id: int,
    sep_id: int,
    Q_marker: int,
    D_marker: int,
    field_marker_ids: list[int] | None,
) -> tuple[bool, float, str]:
    """Run ``steps`` forward+backward+optimizer.step iterations at ``bsize``.

    Returns ``(success, peak_mem_gib, message)``.
    """
    base = model.module if hasattr(model, "module") else model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for _ in range(steps):
            Q_ids, Q_mask = _make_random_batch(
                bsize, config.query_maxlen, vocab_size, cls_id, sep_id, Q_marker, device
            )
            D_pos_ids, D_pos_mask = _make_random_batch(
                bsize, config.doc_maxlen, vocab_size, cls_id, sep_id, D_marker,
                device, field_marker_ids=field_marker_ids,
            )
            D_neg_ids, D_neg_mask = _make_random_batch(
                bsize, config.doc_maxlen, vocab_size, cls_id, sep_id, D_marker,
                device, field_marker_ids=field_marker_ids,
            )

            with autocast("cuda", dtype=amp_dtype):
                Q = base.query(Q_ids, Q_mask)
                D_pos, D_pos_doc_mask = base.doc(D_pos_ids, D_pos_mask)
                D_neg, D_neg_doc_mask = base.doc(D_neg_ids, D_neg_mask)
                scores_pos = base.score(Q, D_pos, D_pos_doc_mask)
                scores_neg = base.score(Q, D_neg, D_neg_doc_mask)
                loss = pairwise_ce_loss(scores_pos, scores_neg) / max(1, config.accumsteps)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            del Q, D_pos, D_neg, D_pos_doc_mask, D_neg_doc_mask
            del scores_pos, scores_neg, loss
            del Q_ids, Q_mask, D_pos_ids, D_pos_mask, D_neg_ids, D_neg_mask

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        return True, peak, "ok"

    except torch.cuda.OutOfMemoryError as e:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return False, 0.0, f"OOM: {str(e).splitlines()[0]}"


def run_smoke_test(
    config: ColBERTConfig,
    start: int,
    max_bsize: int,
    steps: int,
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the batch-size smoke test.")

    device = torch.device("cuda:0")
    amp_dtype = config.resolved_torch_dtype
    use_scaler = amp_dtype == torch.float16

    logger.info(
        f"Building model: checkpoint={config.checkpoint!r} dtype={config.torch_dtype} "
        f"attn={config.attn_implementation!r} dim={config.dim}"
    )
    model = ColBERT(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = GradScaler("cuda") if use_scaler else None

    base = model.module if hasattr(model, "module") else model
    tok = base.doc_tokenizer.tok
    vocab_size = base.bert.get_input_embeddings().weight.size(0)
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else 0
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else 0
    Q_marker = base.query_tokenizer.Q_marker_token_id
    D_marker = base.doc_tokenizer.D_marker_token_id
    field_marker_ids = list(base.doc_tokenizer.field_marker_ids.values()) or None

    total_params = sum(p.numel() for p in model.parameters())
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    logger.info(
        f"Params: {total_params:,} | GPU: {torch.cuda.get_device_name(device)} "
        f"({total_mem:.1f} GiB total) | query_maxlen={config.query_maxlen} "
        f"doc_maxlen={config.doc_maxlen} | dtype={amp_dtype} | accumsteps={config.accumsteps}"
    )

    # Warm up Adam state at bsize=1 so subsequent trials measure steady-state memory.
    logger.info("Warming up optimizer state at bsize=1 ...")
    ok, peak, msg = _trial(
        model, optimizer, scaler, amp_dtype, config,
        bsize=1, steps=1, device=device, vocab_size=vocab_size,
        cls_id=cls_id, sep_id=sep_id, Q_marker=Q_marker, D_marker=D_marker,
        field_marker_ids=field_marker_ids,
    )
    if not ok:
        logger.error(f"Even bsize=1 failed: {msg}")
        return 0
    logger.info(f"Warmup ok | peak={peak:.2f} GiB")

    # Phase A: doubling search to find first OOM.
    logger.info("=" * 70)
    logger.info(f"Phase A: doubling from bsize={start} until OOM (max={max_bsize}) ...")
    last_ok = 0
    last_ok_peak = 0.0
    first_oom = None
    bsize = start
    while bsize <= max_bsize:
        ok, peak, msg = _trial(
            model, optimizer, scaler, amp_dtype, config,
            bsize=bsize, steps=steps, device=device, vocab_size=vocab_size,
            cls_id=cls_id, sep_id=sep_id, Q_marker=Q_marker, D_marker=D_marker,
            field_marker_ids=field_marker_ids,
        )
        if ok:
            logger.info(f"  bsize={bsize:<4} OK  | peak={peak:.2f} GiB")
            last_ok = bsize
            last_ok_peak = peak
            if bsize == max_bsize:
                break
            bsize = min(bsize * 2, max_bsize)
        else:
            logger.info(f"  bsize={bsize:<4} FAIL ({msg})")
            first_oom = bsize
            break

    if first_oom is None:
        logger.info("=" * 70)
        logger.info(
            f"No OOM up to --max={max_bsize}. Re-run with a larger --max "
            f"if you want to push higher. Last OK: bsize={last_ok} (peak={last_ok_peak:.2f} GiB)."
        )
        return last_ok

    # Phase B: binary search between last_ok and first_oom.
    logger.info("=" * 70)
    logger.info(f"Phase B: binary search in ({last_ok}, {first_oom}) ...")
    lo, hi = last_ok, first_oom
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, peak, msg = _trial(
            model, optimizer, scaler, amp_dtype, config,
            bsize=mid, steps=steps, device=device, vocab_size=vocab_size,
            cls_id=cls_id, sep_id=sep_id, Q_marker=Q_marker, D_marker=D_marker,
            field_marker_ids=field_marker_ids,
        )
        if ok:
            logger.info(f"  bsize={mid:<4} OK  | peak={peak:.2f} GiB")
            lo = mid
            last_ok_peak = peak
        else:
            logger.info(f"  bsize={mid:<4} FAIL ({msg})")
            hi = mid

    logger.info("=" * 70)
    eff = lo * max(1, config.accumsteps)
    logger.info(
        f"Maximum bsize that fits: {lo} (peak ~{last_ok_peak:.2f} GiB, "
        f"first OOM at {hi}). Effective batch with accumsteps={config.accumsteps}: {eff}."
    )
    logger.info(
        "Reminder: leave ~1-2 GiB headroom for DDP buckets, NCCL buffers, and "
        "data-loader pinned memory before committing this bsize to your config."
    )
    return lo


def main():
    parser = argparse.ArgumentParser(
        description="Estimate max Phase-1 batch size that fits on this GPU.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--start", type=int, default=1,
        help="Starting batch size for the doubling phase (default: 1).",
    )
    parser.add_argument(
        "--max", dest="max_bsize", type=int, default=256,
        help="Hard upper bound for the doubling phase (default: 256).",
    )
    parser.add_argument(
        "--steps", type=int, default=3,
        help="Forward+backward+step iterations per trial (default: 3).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    config = ColBERTConfig.from_yaml(args.config)
    run_smoke_test(config, start=args.start, max_bsize=args.max_bsize, steps=args.steps)


if __name__ == "__main__":
    main()
