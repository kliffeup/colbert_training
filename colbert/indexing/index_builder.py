"""ColBERTv2 indexing pipeline — single-process master with spawned GPU workers.

No DDP / torchrun required.  The master process:
  1. Samples embeddings and trains k-means via fastkmeans (GPU-accelerated).
  2. Spawns one encoding worker per GPU via ``torch.multiprocessing.spawn``.
  3. Merges shards and builds the compressed inverted index.
"""

from __future__ import annotations

import gc
import logging
import math
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from fastkmeans import FastKMeans
from tqdm import tqdm

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.dataset.collection import Collection
from colbert.indexing.encoder import (
    encode_and_compress_collection,
    encode_and_compress_collection_multigpu,
    sample_embeddings,
)
from colbert.indexing.residual_codec import ResidualCodec
from colbert.indexing.resource_monitor import ResourceMonitor
from colbert.indexing.saver import IndexSaver

logger = logging.getLogger(__name__)


def _compute_num_centroids(n_embeddings: int) -> int:
    """Round down to nearest power-of-2 >= 16 * sqrt(n)."""
    target = 16 * math.sqrt(n_embeddings)
    power = int(math.log2(max(target, 1)))
    return 2 ** power


def build_index(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    index_path: str | None = None,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
    num_gpus: int | None = None,
) -> str:
    """Build a ColBERTv2 index.

    Runs entirely from a single master process.  When more than one CUDA device
    is available the encoding stage is parallelised by spawning one worker per
    GPU via ``torch.multiprocessing.spawn``, and centroid training uses
    fastkmeans (GPU-accelerated PyTorch k-means).

    Args:
        model: Trained ColBERT model (on *any* device — workers will reload it).
        collection: Passage collection.
        config: Configuration.
        index_path: Override for index output directory.
        batch_size: Batch size **per GPU** for encoding.
        doc_maxlen: Override for document max length.
        num_gpus: Override number of GPUs (defaults to all visible CUDA devices).

    Returns:
        Path to the built index directory.
    """
    ngpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
    ngpus = max(ngpus, 1)
    logger.info(f"Indexing with {ngpus} GPU(s)")

    output = Path(index_path or config.index_path)
    shard_dir = output / "_shards"
    output.mkdir(parents=True, exist_ok=True)

    saver = IndexSaver(output)

    # Tracks process RAM and the index output dir's on-disk size across stages.
    monitor = ResourceMonitor(output, logger)
    monitor.report("build start")

    # ------------------------------------------------------------------
    # Stage 1: Centroid Selection
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: Centroid Selection ===")

    sampled_embeddings = sample_embeddings(
        model, collection, config, batch_size=batch_size,
    )
    # sample_embeddings returns a CPU array; its forward-pass GPU activations are now
    # out of scope. Reclaim them before k-means so they don't inflate the Stage-1 peak.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_centroids = _compute_num_centroids(len(collection) * 60)
    dim = sampled_embeddings.shape[1]

    use_gpu = torch.cuda.is_available()
    logger.info(
        f"Training k-means with {num_centroids} centroids "
        f"on {sampled_embeddings.shape[0]} embeddings "
        f"(fastkmeans, gpu={use_gpu})"
    )
    kmeans = FastKMeans(
        dim, num_centroids,
        niter=config.kmeans_niters,
        verbose=True,
        gpu=use_gpu,
    )
    kmeans.train(sampled_embeddings)
    centroids = torch.from_numpy(kmeans.centroids).float()

    logger.info(f"Centroids shape: {centroids.shape}")

    # Free Stage-1 GPU memory (k-means state + cached allocations) before encoding,
    # so Stage 2 can use a larger batch size. centroids/sampled_embeddings are on CPU.
    del kmeans
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    monitor.report("after Stage 1 (centroids)")

    # ------------------------------------------------------------------
    # Codec quantization params (trained from the Stage-1 sample)
    # ------------------------------------------------------------------
    # Residual buckets are estimated from the in-distribution Stage-1 sample
    # (already in RAM) against the trained centroids — the canonical ColBERTv2
    # choice — so we never need the full embedding set resident to set them.
    codec = ResidualCodec(centroids, nbits=config.nbits)

    sample_size = min(len(sampled_embeddings), 100_000)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(sampled_embeddings), size=sample_size, replace=False)
    sample_vecs = torch.from_numpy(sampled_embeddings[sample_idx])

    sims = sample_vecs @ centroids.t()
    sample_cids = sims.argmax(dim=1)
    sample_residuals = sample_vecs - centroids[sample_cids]
    codec.set_quantization_params(sample_residuals)
    del sampled_embeddings, sample_vecs, sims, sample_cids, sample_residuals

    # ------------------------------------------------------------------
    # Stage 2+3: Fused Encoding + Compression (streamed to disk)
    # ------------------------------------------------------------------
    # Each batch is encoded then compressed immediately; only the compressed codes
    # (~40 B/token) are streamed to disk. The raw float32 embeddings are never
    # accumulated, so peak RAM stays at ~one batch regardless of collection size.
    logger.info("=== Stage 2+3: Encoding + Compression (streaming) ===")

    cids_path, residuals_path = saver.compressed_embeddings_paths()

    if ngpus > 1:
        all_doclens, all_pids, num_embeddings = encode_and_compress_collection_multigpu(
            model, collection, config, codec,
            num_gpus=ngpus, shard_dir=shard_dir,
            cids_path=cids_path, residuals_path=residuals_path,
            batch_size=batch_size, doc_maxlen=doc_maxlen,
        )
    else:
        all_doclens, all_pids, num_embeddings = encode_and_compress_collection(
            model, collection, config, codec,
            work_dir=shard_dir,
            cids_path=cids_path, residuals_path=residuals_path,
            batch_size=batch_size, doc_maxlen=doc_maxlen,
            monitor=monitor,
        )

    monitor.report("after Stage 2+3 (encode+compress)")

    # ------------------------------------------------------------------
    # Stage 4: Index Inversion
    # ------------------------------------------------------------------
    logger.info("=== Stage 4: Index Inversion ===")

    # Re-read the streamed centroid ids from disk in blocks (never the whole array
    # in a second copy); the inverted-list dict itself scales with total tokens.
    centroid_ids_mm = np.load(cids_path, mmap_mode="r")
    inverted_lists: Dict[int, List[int]] = {}
    block_size = 1_000_000
    for start in tqdm(range(0, num_embeddings, block_size), desc="Inverting"):
        block = np.asarray(centroid_ids_mm[start:start + block_size])
        for off, cid in enumerate(block):
            inverted_lists.setdefault(int(cid), []).append(start + off)
    del centroid_ids_mm

    logger.info(f"Built inverted lists for {len(inverted_lists)} centroids")
    monitor.report("after Stage 4 (inversion)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # centroid_ids.npy / packed_residuals.npy were already written by the streaming
    # pass; only the remaining artifacts need saving here.
    saver.save_codec(codec)
    saver.save_doclens(all_doclens)
    saver.save_pids(all_pids)
    saver.save_inverted_lists(inverted_lists)
    saver.save_metadata({
        "num_passages": len(all_pids),
        "num_embeddings": num_embeddings,
        "num_centroids": int(centroids.shape[0]),
        "dim": int(centroids.shape[1]),
        "nbits": config.nbits,
        "bytes_per_vector": codec.bytes_per_vector,
        "num_gpus_used": ngpus,
    })

    index_size_mb = num_embeddings * codec.bytes_per_vector / (1024 * 1024)
    logger.info(
        f"Index built at {output} — "
        f"{len(all_pids)} passages, {num_embeddings} embeddings, "
        f"~{index_size_mb:.1f} MiB compressed"
    )

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
        logger.info("Cleaned up temporary shard files")

    monitor.report("build complete")
    return str(output)
