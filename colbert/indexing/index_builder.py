"""ColBERTv2 indexing pipeline — single-process master with spawned GPU workers.

No DDP / torchrun required.  The master process:
  1. Samples embeddings and trains k-means via fastkmeans (GPU-accelerated).
  2. Spawns one encoding worker per GPU via ``torch.multiprocessing.spawn``.
  3. Merges shards and builds the compressed inverted index.
"""

from __future__ import annotations

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
    encode_collection,
    encode_collection_multigpu,
    sample_embeddings,
)
from colbert.indexing.residual_codec import ResidualCodec
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

    # ------------------------------------------------------------------
    # Stage 1: Centroid Selection
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: Centroid Selection ===")

    sampled_embeddings = sample_embeddings(
        model, collection, config, batch_size=batch_size,
    )
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

    # ------------------------------------------------------------------
    # Stage 2: Passage Encoding (multi-GPU spawn or single-GPU)
    # ------------------------------------------------------------------
    logger.info("=== Stage 2: Passage Encoding ===")

    if ngpus > 1:
        all_embeddings, all_doclens, all_pids = encode_collection_multigpu(
            model, collection, config,
            num_gpus=ngpus,
            shard_dir=shard_dir,
            batch_size=batch_size,
            doc_maxlen=doc_maxlen,
        )
    else:
        all_embeddings, all_doclens, all_pids = encode_collection(
            model, collection, config,
            batch_size=batch_size,
            doc_maxlen=doc_maxlen,
        )

    # ------------------------------------------------------------------
    # Stage 3: Compression
    # ------------------------------------------------------------------
    logger.info("=== Stage 3: Compression ===")

    codec = ResidualCodec(centroids, nbits=config.nbits)

    sample_size = min(len(all_embeddings), 100_000)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(all_embeddings), size=sample_size, replace=False)
    sample_vecs = torch.from_numpy(all_embeddings[sample_idx])

    sims = sample_vecs @ centroids.t()
    sample_cids = sims.argmax(dim=1)
    sample_residuals = sample_vecs - centroids[sample_cids]
    codec.set_quantization_params(sample_residuals)

    logger.info(
        f"Compressing {len(all_embeddings)} embeddings "
        f"with {config.nbits}-bit residuals"
    )
    chunk_size = 1_000_000
    all_centroid_ids_list = []
    all_packed_list = []

    for start in tqdm(range(0, len(all_embeddings), chunk_size), desc="Compressing"):
        end = min(start + chunk_size, len(all_embeddings))
        chunk = torch.from_numpy(all_embeddings[start:end])
        cids, packed = codec.encode(chunk)
        all_centroid_ids_list.append(cids)
        all_packed_list.append(packed)

    all_centroid_ids = np.concatenate(all_centroid_ids_list)
    all_packed_residuals = np.concatenate(all_packed_list)

    # ------------------------------------------------------------------
    # Stage 4: Index Inversion
    # ------------------------------------------------------------------
    logger.info("=== Stage 4: Index Inversion ===")

    inverted_lists: Dict[int, List[int]] = {}
    for emb_id, cid in enumerate(all_centroid_ids):
        cid = int(cid)
        if cid not in inverted_lists:
            inverted_lists[cid] = []
        inverted_lists[cid].append(emb_id)

    logger.info(f"Built inverted lists for {len(inverted_lists)} centroids")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    saver.save_codec(codec)
    saver.save_compressed_embeddings(all_centroid_ids, all_packed_residuals)
    saver.save_doclens(all_doclens)
    saver.save_pids(all_pids)
    saver.save_inverted_lists(inverted_lists)
    saver.save_metadata({
        "num_passages": len(all_pids),
        "num_embeddings": len(all_centroid_ids),
        "num_centroids": int(centroids.shape[0]),
        "dim": int(centroids.shape[1]),
        "nbits": config.nbits,
        "bytes_per_vector": codec.bytes_per_vector,
        "num_gpus_used": ngpus,
    })

    index_size_mb = (
        all_centroid_ids.nbytes + all_packed_residuals.nbytes
    ) / (1024 * 1024)
    logger.info(
        f"Index built at {output} — "
        f"{len(all_pids)} passages, {len(all_centroid_ids)} embeddings, "
        f"~{index_size_mb:.1f} MiB compressed"
    )

    if shard_dir.exists():
        shutil.rmtree(shard_dir)
        logger.info("Cleaned up temporary shard files")

    return str(output)
