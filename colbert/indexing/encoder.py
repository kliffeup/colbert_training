"""Batch-encode collection passages into per-token embeddings.

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core encoding helpers
# ---------------------------------------------------------------------------

def _encode_batches(
    model: ColBERT,
    batch_iterator,
    total_batches: int,
    config: ColBERTConfig,
    doc_maxlen: int | None = None,
    desc: str = "Encoding",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Shared encoding loop for both full-collection and shard-based encoding.

    Returns pids as a List[str] so that arbitrary string IDs (e.g. MS MARCO
    Doc "D123456") are preserved end-to-end.
    """
    maxlen = doc_maxlen or config.doc_maxlen

    all_embeddings: List[np.ndarray] = []
    all_doclens: List[int] = []
    all_pids: List[str] = []

    amp_dtype = config.resolved_torch_dtype
    with torch.no_grad(), autocast("cuda", dtype=amp_dtype):
        for batch in tqdm(batch_iterator, total=total_batches, desc=desc):
            pids, texts = zip(*batch)
            D, D_mask = model.encode_docs(list(texts), maxlen=maxlen)

            for i in range(len(pids)):
                mask_i = D_mask[i]
                embs_i = D[i][mask_i]
                all_embeddings.append(embs_i.cpu().numpy())
                all_doclens.append(embs_i.shape[0])
                all_pids.append(str(pids[i]))

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    doclens = np.array(all_doclens, dtype=np.int32)
    return embeddings, doclens, all_pids


# ---------------------------------------------------------------------------
# Single-process encoding
# ---------------------------------------------------------------------------

def encode_collection(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode all passages in the collection (single-process).

    Returns:
        all_embeddings: float32 array of shape (total_tokens, dim).
        all_doclens: int32 array of shape (num_docs,) — tokens per doc.
        all_pids: List[str] of length num_docs — passage / document IDs.
    """
    model.eval()
    total_passages = len(collection)
    total_batches = math.ceil(total_passages / batch_size)
    logger.info(f"Encoding {total_passages} passages with batch_size={batch_size}")

    embeddings, doclens, pids = _encode_batches(
        model, collection.iterate(batch_size), total_batches, config,
        doc_maxlen=doc_maxlen, desc="Encoding",
    )
    logger.info(
        f"Encoded {len(pids)} passages -> {embeddings.shape[0]} token embeddings "
        f"(avg {embeddings.shape[0] / len(pids):.1f} tokens/doc)"
    )
    return embeddings, doclens, pids


def encode_collection_shard(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    rank: int,
    world_size: int,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode this rank's shard of the collection.

    Returns:
        embeddings, doclens, pids for this shard only.
    """
    model.eval()
    shard_size = collection.shard_size(rank, world_size)
    total_batches = math.ceil(shard_size / batch_size)
    logger.info(
        f"[GPU {rank}] Encoding shard: {shard_size} passages "
        f"(of {len(collection)} total) with batch_size={batch_size}"
    )

    embeddings, doclens, pids = _encode_batches(
        model,
        collection.iterate_shard(rank, world_size, batch_size),
        total_batches, config,
        doc_maxlen=doc_maxlen,
        desc=f"Encoding [GPU {rank}]",
    )
    logger.info(
        f"[GPU {rank}] Encoded {len(pids)} passages -> "
        f"{embeddings.shape[0]} token embeddings"
    )
    return embeddings, doclens, pids


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def save_shard(
    embeddings: np.ndarray,
    doclens: np.ndarray,
    pids: List[str],
    shard_dir: Path,
    rank: int,
) -> None:
    """Persist a GPU worker's encoded shard to disk.

    Pids are saved as a UTF-8 text file (one id per line) so arbitrary string IDs are
    preserved without needing object-dtype numpy arrays.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / f"embeddings_{rank}.npy", embeddings)
    np.save(shard_dir / f"doclens_{rank}.npy", doclens)
    with open(shard_dir / f"pids_{rank}.txt", "w", encoding="utf-8") as f:
        for pid in pids:
            f.write(f"{pid}\n")
    logger.info(f"[GPU {rank}] Saved shard to {shard_dir}")


def load_and_merge_shards(
    shard_dir: Path,
    world_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all shard files and concatenate, sorted by pid (lexicographic for strings)."""
    all_embeddings = []
    all_doclens = []
    all_pids: List[str] = []

    for rank in range(world_size):
        all_embeddings.append(np.load(shard_dir / f"embeddings_{rank}.npy"))
        all_doclens.append(np.load(shard_dir / f"doclens_{rank}.npy"))
        with open(shard_dir / f"pids_{rank}.txt", encoding="utf-8") as f:
            all_pids.extend(line.rstrip("\n") for line in f)

    doclens = np.concatenate(all_doclens)

    sort_idx = sorted(range(len(all_pids)), key=lambda i: all_pids[i])
    pids = [all_pids[i] for i in sort_idx]
    doclens = doclens[sort_idx]

    all_embs = np.concatenate(all_embeddings, axis=0)

    orig_doclens = np.concatenate(all_doclens)
    orig_offsets = np.zeros(len(orig_doclens) + 1, dtype=np.int64)
    np.cumsum(orig_doclens, out=orig_offsets[1:])

    sorted_emb_chunks = []
    for new_pos in range(len(sort_idx)):
        orig_pos = sort_idx[new_pos]
        start = orig_offsets[orig_pos]
        end = orig_offsets[orig_pos + 1]
        sorted_emb_chunks.append(all_embs[start:end])

    embeddings = np.concatenate(sorted_emb_chunks, axis=0).astype(np.float32)

    logger.info(
        f"Merged {world_size} shards: {len(pids)} passages, "
        f"{embeddings.shape[0]} token embeddings"
    )
    return embeddings, doclens, pids


# ---------------------------------------------------------------------------
# Multi-GPU encoding via torch.multiprocessing.spawn
# ---------------------------------------------------------------------------

def _encode_worker_fn(
    rank: int,
    world_size: int,
    config: ColBERTConfig,
    state_dict_path: str,
    collection_path: str,
    shard_dir: str,
    batch_size: int,
    doc_maxlen: int | None,
) -> None:
    """Worker function spawned on each GPU for parallel encoding.

    Each worker independently loads the model on its assigned GPU,
    encodes its shard of the collection, and writes the result to disk.
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [GPU {rank}] %(name)s %(levelname)s: %(message)s",
    )
    worker_logger = logging.getLogger(__name__)
    worker_logger.info(f"Encoding worker started on GPU {rank}")

    model = ColBERT(config).to(device)
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict

    collection = Collection(collection_path)

    embeddings, doclens, pids = encode_collection_shard(
        model, collection, config,
        rank=rank, world_size=world_size,
        batch_size=batch_size, doc_maxlen=doc_maxlen,
    )
    save_shard(embeddings, doclens, pids, Path(shard_dir), rank)

    del model, embeddings, doclens, pids
    torch.cuda.empty_cache()
    worker_logger.info(f"Encoding worker on GPU {rank} finished")


def encode_collection_multigpu(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    num_gpus: int,
    shard_dir: Path,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode the collection in parallel using multiple GPUs.

    Saves the model state to a temp file, spawns one worker per GPU
    (via ``torch.multiprocessing.spawn``), each encodes its shard, then
    the master process merges all shards.

    Returns:
        all_embeddings, all_doclens, all_pids — same as ``encode_collection``.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)

    state_dict_path = shard_dir / "_model_state.pt"
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict_cpu, state_dict_path)
    del state_dict_cpu

    collection_path = str(collection.path)

    logger.info(f"Spawning {num_gpus} encoding workers")
    mp.spawn(
        _encode_worker_fn,
        args=(
            num_gpus, config, str(state_dict_path), collection_path,
            str(shard_dir), batch_size, doc_maxlen,
        ),
        nprocs=num_gpus,
        join=True,
    )

    state_dict_path.unlink(missing_ok=True)
    logger.info("All encoding workers finished, merging shards")
    return load_and_merge_shards(shard_dir, num_gpus)


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
