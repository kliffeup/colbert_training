"""Fused encode→compress collection pass (streaming, disk-backed).

Each batch is encoded on the GPU and compressed immediately; only the compressed
codes (~40 B/token) are streamed to disk via :class:`CompressedShardWriter`.  The raw
float32 embeddings (512 B/token) are released per-batch and never accumulate, so peak
RAM is ~one batch regardless of collection size.

Supports single-GPU encoding and multi-GPU encoding via spawned workers
(no DDP / torchrun needed).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.amp import autocast
from tqdm import tqdm

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.dataset.collection import Collection
from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.compressed_writer import (
    CompressedShardWriter,
    merge_compressed_shards,
)
from colbert.indexing.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core fused encode→compress loop
# ---------------------------------------------------------------------------

def _encode_compress_batches(
    model: ColBERT,
    batch_iterator,
    total_batches: int,
    config: ColBERTConfig,
    codec: ResidualCodec,
    writer: CompressedShardWriter,
    doc_maxlen: int | None = None,
    desc: str = "Encoding",
    monitor: ResourceMonitor | None = None,
    monitor_every: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """Shared fused loop: encode each batch, compress it, stream codes to ``writer``.

    Compressed rows are appended in document order (doc0 tokens, doc1 tokens, …),
    matching the returned ``doclens`` / ``pids`` — the layout the retriever relies on.

    When ``monitor`` is given, RAM/disk usage is reported every ``monitor_every``
    batches so progress through a long streaming pass is observable.

    Returns:
        doclens: int32 array of shape (num_docs,) — kept tokens per doc.
        pids: List[str] of length num_docs.
    """
    maxlen = doc_maxlen or config.doc_maxlen

    all_doclens: List[int] = []
    all_pids: List[str] = []

    amp_dtype = config.resolved_torch_dtype
    with torch.no_grad(), autocast("cuda", dtype=amp_dtype):
        for batch_idx, batch in enumerate(
            tqdm(batch_iterator, total=total_batches, desc=desc)
        ):
            pids, texts = zip(*batch)
            D, D_mask = model.encode_docs(list(texts), maxlen=maxlen)

            batch_tokens: List[torch.Tensor] = []
            for i in range(len(pids)):
                embs_i = D[i][D_mask[i]]
                all_doclens.append(int(embs_i.shape[0]))
                all_pids.append(str(pids[i]))
                batch_tokens.append(embs_i)

            # Compress the whole batch at once (codec runs in fp32 on CPU), then
            # drop the raw embeddings before the next batch.
            flat = torch.cat(batch_tokens, dim=0).float().cpu()
            cids, packed = codec.encode(flat)
            writer.append(cids, packed)
            del D, D_mask, batch_tokens, flat

            if monitor is not None and monitor_every > 0 and (batch_idx + 1) % monitor_every == 0:
                monitor.report(f"{desc} batch {batch_idx + 1}/{total_batches}")

    doclens = np.array(all_doclens, dtype=np.int32)
    return doclens, all_pids


# ---------------------------------------------------------------------------
# Single-process fused encode→compress
# ---------------------------------------------------------------------------

def encode_and_compress_collection(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    codec: ResidualCodec,
    work_dir: Path,
    cids_path: Path,
    residuals_path: Path,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
    monitor: ResourceMonitor | None = None,
) -> Tuple[np.ndarray, List[str], int]:
    """Encode + compress the whole collection (single process), streaming to disk.

    Writes ``centroid_ids.npy`` / ``packed_residuals.npy`` at the given paths.

    Returns:
        doclens (int32), pids (List[str]), total_tokens.
    """
    model.eval()
    total_passages = len(collection)
    total_batches = math.ceil(total_passages / batch_size)
    bytes_per_residual = codec.bytes_per_vector - 4
    logger.info(
        f"Encoding+compressing {total_passages} passages with batch_size={batch_size}"
    )

    writer = CompressedShardWriter(work_dir, bytes_per_residual, tag="")
    doclens, pids = _encode_compress_batches(
        model, collection.iterate(batch_size), total_batches, config,
        codec, writer, doc_maxlen=doc_maxlen, desc="Encoding",
        monitor=monitor, monitor_every=max(1, total_batches // 20),
    )
    total_tokens = writer.finalize_npy(cids_path, residuals_path)

    logger.info(
        f"Encoded {len(pids)} passages -> {total_tokens} token embeddings "
        f"(avg {total_tokens / max(len(pids), 1):.1f} tokens/doc)"
    )
    return doclens, pids, total_tokens


# ---------------------------------------------------------------------------
# Multi-GPU fused encode→compress via torch.multiprocessing.spawn
# ---------------------------------------------------------------------------

def _save_shard_meta(
    doclens: np.ndarray,
    pids: List[str],
    shard_dir: Path,
    rank: int,
) -> None:
    """Persist a worker's per-doc metadata (compressed codes are already streamed)."""
    np.save(shard_dir / f"doclens_{rank}.npy", doclens)
    with open(shard_dir / f"pids_{rank}.txt", "w", encoding="utf-8") as f:
        for pid in pids:
            f.write(f"{pid}\n")
    logger.info(f"[GPU {rank}] Saved shard metadata to {shard_dir}")


def _encode_compress_worker_fn(
    rank: int,
    world_size: int,
    config: ColBERTConfig,
    state_dict_path: str,
    codec_path: str,
    collection_path: str,
    shard_dir: str,
    batch_size: int,
    doc_maxlen: int | None,
    bytes_per_residual: int,
) -> None:
    """Worker: load model + codec on this GPU, fused-encode its shard, stream codes."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {rank}] %(name)s %(levelname)s: %(message)s",
    )
    worker_logger = logging.getLogger(__name__)
    worker_logger.info(f"Encode+compress worker started on GPU {rank}")

    model = ColBERT(config).to(device)
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict

    codec = ResidualCodec.load(codec_path)
    collection = Collection(collection_path)

    shard_size = collection.shard_size(rank, world_size)
    total_batches = math.ceil(shard_size / batch_size)
    worker_logger.info(
        f"[GPU {rank}] Encoding shard: {shard_size} passages "
        f"(of {len(collection)} total) with batch_size={batch_size}"
    )

    writer = CompressedShardWriter(Path(shard_dir), bytes_per_residual, tag=f"_{rank}")
    monitor = ResourceMonitor(shard_dir, worker_logger)
    doclens, pids = _encode_compress_batches(
        model,
        collection.iterate_shard(rank, world_size, batch_size),
        total_batches, config, codec, writer,
        doc_maxlen=doc_maxlen, desc=f"Encoding [GPU {rank}]",
        monitor=monitor, monitor_every=max(1, total_batches // 10),
    )
    writer._close()  # keep the temp .bin files for the master to merge
    _save_shard_meta(doclens, pids, Path(shard_dir), rank)

    del model, codec
    torch.cuda.empty_cache()
    worker_logger.info(f"[GPU {rank}] worker finished: {writer.total} token embeddings")


def merge_compressed_shards_from_dir(
    shard_dir: Path,
    world_size: int,
    codec: ResidualCodec,
    cids_path: Path,
    residuals_path: Path,
) -> Tuple[np.ndarray, List[str], int]:
    """Merge per-shard compressed binaries into final ``.npy`` artifacts.

    Documents are emitted in pid-sorted order (matching the previous multi-GPU
    behaviour), copying one document's codes at a time so peak RAM is a single doc.
    """
    bytes_per_residual = codec.bytes_per_vector - 4

    # (pid, shard_idx, tok_start_in_shard, doclen) for every document.
    records: List[Tuple[str, int, int, int]] = []
    shard_token_counts: List[int] = []
    for rank in range(world_size):
        dl = np.load(shard_dir / f"doclens_{rank}.npy")
        with open(shard_dir / f"pids_{rank}.txt", encoding="utf-8") as f:
            pids_r = [line.rstrip("\n") for line in f]
        offsets = np.zeros(len(dl) + 1, dtype=np.int64)
        np.cumsum(dl, out=offsets[1:])
        shard_token_counts.append(int(offsets[-1]))
        for j, pid in enumerate(pids_r):
            records.append((pid, rank, int(offsets[j]), int(dl[j])))

    records.sort(key=lambda r: r[0])  # stable lexicographic pid sort

    doc_order = [(r[1], r[2], r[3]) for r in records]
    doclens = np.array([r[3] for r in records], dtype=np.int32)
    pids = [r[0] for r in records]
    total_tokens = int(doclens.sum())

    shard_bins = [
        (
            shard_dir / f"centroid_ids_{rank}.bin",
            shard_dir / f"packed_residuals_{rank}.bin",
            shard_token_counts[rank],
        )
        for rank in range(world_size)
    ]
    merge_compressed_shards(
        shard_bins, doc_order, cids_path, residuals_path,
        bytes_per_residual, total_tokens,
    )

    logger.info(
        f"Merged {world_size} shards: {len(pids)} passages, "
        f"{total_tokens} token embeddings"
    )
    return doclens, pids, total_tokens


def encode_and_compress_collection_multigpu(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    codec: ResidualCodec,
    num_gpus: int,
    shard_dir: Path,
    cids_path: Path,
    residuals_path: Path,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
) -> Tuple[np.ndarray, List[str], int]:
    """Encode + compress in parallel across GPUs, then merge compressed shards.

    Returns:
        doclens (int32), pids (List[str]), total_tokens — same as the single-GPU path.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)

    state_dict_path = shard_dir / "_model_state.pt"
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict_cpu, state_dict_path)
    del state_dict_cpu

    codec_path = shard_dir / "_codec.pt"
    codec.save(str(codec_path))

    bytes_per_residual = codec.bytes_per_vector - 4
    collection_path = str(collection.path)

    logger.info(f"Spawning {num_gpus} encode+compress workers")
    mp.spawn(
        _encode_compress_worker_fn,
        args=(
            num_gpus, config, str(state_dict_path), str(codec_path),
            collection_path, str(shard_dir), batch_size, doc_maxlen,
            bytes_per_residual,
        ),
        nprocs=num_gpus,
        join=True,
    )

    state_dict_path.unlink(missing_ok=True)
    codec_path.unlink(missing_ok=True)
    logger.info("All workers finished, merging compressed shards")
    return merge_compressed_shards_from_dir(
        shard_dir, num_gpus, codec, cids_path, residuals_path,
    )


# ---------------------------------------------------------------------------
# Sampling for centroid estimation
# ---------------------------------------------------------------------------

def sample_embeddings(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    sample_fraction: float | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode a sample of passages for centroid estimation.

    Samples proportional to sqrt(collection_size) passages.

    Returns:
        Sampled embeddings of shape (n_sampled_tokens, dim).
    """
    total = len(collection)
    if sample_fraction is not None:
        n_sample = max(1, int(total * sample_fraction))
    else:
        n_sample = max(1, int(16 * math.sqrt(65 * total)))
    n_sample = min(n_sample, total)

    logger.info(f"Sampling {n_sample}/{total} passages for centroid estimation")

    pids = list(collection.pids())
    rng = np.random.default_rng(seed=42)
    # rng.choice on object dtype returns python strings; cast to list for downstream indexing
    sampled_idx = rng.choice(len(pids), size=n_sample, replace=False)
    sampled_pids = [pids[i] for i in sampled_idx]

    model.eval()
    all_embeddings: List[np.ndarray] = []

    amp_dtype = config.resolved_torch_dtype
    with torch.no_grad(), autocast("cuda", dtype=amp_dtype):
        for i in tqdm(range(0, len(sampled_pids), batch_size), desc="Sampling"):
            batch_pids = sampled_pids[i:i + batch_size]
            texts = [collection[pid] for pid in batch_pids]

            D, D_mask = model.encode_docs(texts, maxlen=config.doc_maxlen)

            for j in range(len(batch_pids)):
                mask_j = D_mask[j]
                embs_j = D[j][mask_j]
                all_embeddings.append(embs_j.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logger.info(f"Sampled {embeddings.shape[0]} token embeddings from {n_sample} passages")
    return embeddings
