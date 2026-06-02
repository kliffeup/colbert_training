"""ColBERTv2 indexing pipeline — single-process master with spawned GPU workers.

No DDP / torchrun required.  The master process:
  1. Samples embeddings and trains k-means via fastkmeans (GPU-accelerated).
  2. Spawns one encoding worker per GPU via ``torch.multiprocessing.spawn``.
  3. Merges shards and builds the compressed inverted index.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional

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


# Build stages, in completion order; used to gate work on resume.
_STAGE_ORDER = {"centroids": 1, "encoded": 2, "complete": 3}


def _stage_ge(stage: Optional[str], target: str) -> bool:
    """True if completed ``stage`` is at least as far along as ``target``."""
    return stage is not None and _STAGE_ORDER.get(stage, 0) >= _STAGE_ORDER[target]


def _build_fingerprint(
    collection: Collection,
    config: ColBERTConfig,
    ngpus: int,
    doc_maxlen: int | None,
) -> str:
    """Hash the structural inputs a resume must match.

    Excludes ``batch_size`` on purpose: ColBERT encodes each doc independently, so the
    compressed codes are batch-independent and the batch size may be retuned between
    runs.  The model checkpoint is *assumed* identical across a resume (not hashed, to
    avoid hashing weights); a different checkpoint would silently change embeddings.
    """
    payload = {
        "collection": str(collection.path),
        "num_passages": len(collection),
        "doc_maxlen": int(doc_maxlen or config.doc_maxlen),
        "nbits": int(config.nbits),
        "kmeans_niters": int(config.kmeans_niters),
        "dim": int(config.dim),
        "num_gpus": int(ngpus),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_index(
    model: ColBERT,
    collection: Collection,
    config: ColBERTConfig,
    index_path: str | None = None,
    batch_size: int = 128,
    doc_maxlen: int | None = None,
    num_gpus: int | None = None,
    resume: bool = False,
) -> str:
    """Build a ColBERTv2 index.

    Runs entirely from a single master process.  When more than one CUDA device
    is available the encoding stage is parallelised by spawning one worker per
    GPU via ``torch.multiprocessing.spawn``, and centroid training uses
    fastkmeans (GPU-accelerated PyTorch k-means).

    The build is resumable: progress is committed to disk after centroid training and
    periodically during the (dominant) encode pass, so a container stopped mid-build
    can be restarted with ``resume=True`` to continue instead of starting over.

    Args:
        model: Trained ColBERT model (on *any* device — workers will reload it).
        collection: Passage collection.
        config: Configuration.
        index_path: Override for index output directory.
        batch_size: Batch size **per GPU** for encoding.
        doc_maxlen: Override for document max length.
        num_gpus: Override number of GPUs (defaults to all visible CUDA devices).
        resume: Continue a previously-interrupted build in ``index_path`` (must match
            the same collection/config; otherwise raises). When False, any stale
            partial build is cleared first.

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
    checkpoint_every = config.index_checkpoint_every
    fingerprint = _build_fingerprint(collection, config, ngpus, doc_maxlen)

    # ------------------------------------------------------------------
    # Resume gating: decide how far a prior build got (if resuming).
    # ------------------------------------------------------------------
    stage: Optional[str] = None
    if resume:
        state = saver.load_build_state()
        if state is None:
            logger.info("No prior build state found; starting a fresh build")
        elif state.get("fingerprint") != fingerprint:
            raise ValueError(
                "Cannot resume: the existing partial index in "
                f"{output} was built with different inputs (collection size, "
                "doc_maxlen, nbits, dim, or num_gpus changed). "
                "Re-run without --resume to rebuild from scratch."
            )
        else:
            stage = state.get("stage")
            logger.info(f"Resuming build from completed stage '{stage}'")
    else:
        # Fresh build: clear any stale partial state so we never append to it.
        saver._build_state_path.unlink(missing_ok=True)
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

    # Tracks process RAM and the index output dir's on-disk size across stages.
    monitor = ResourceMonitor(output, logger)
    monitor.report("build start")

    if stage == "complete":
        logger.info(f"Index at {output} is already complete; nothing to do")
        return str(output)

    cids_path, residuals_path = saver.compressed_embeddings_paths()

    # ------------------------------------------------------------------
    # Stage 1: Centroid Selection (+ codec quantization params)
    # ------------------------------------------------------------------
    if _stage_ge(stage, "centroids"):
        logger.info("=== Stage 1: reusing existing centroids/codec ===")
        codec = saver.load_codec()
    else:
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

        # Commit Stage 1: a resumed run reloads this codec and skips sampling/k-means.
        saver.save_codec(codec)
        saver.save_build_state({"fingerprint": fingerprint, "stage": "centroids"})

    # ------------------------------------------------------------------
    # Stage 2+3: Fused Encoding + Compression (streamed to disk)
    # ------------------------------------------------------------------
    # Each batch is encoded then compressed immediately; only the compressed codes
    # (~40 B/token) are streamed to disk. The raw float32 embeddings are never
    # accumulated, so peak RAM stays at ~one batch regardless of collection size.
    if _stage_ge(stage, "encoded"):
        logger.info("=== Stage 2+3: reusing existing encoded embeddings ===")
        all_doclens = saver.load_doclens()
        all_pids = saver.load_pids()
        num_embeddings = int(all_doclens.sum())
    else:
        logger.info("=== Stage 2+3: Encoding + Compression (streaming) ===")
        if ngpus > 1:
            all_doclens, all_pids, num_embeddings = encode_and_compress_collection_multigpu(
                model, collection, config, codec,
                num_gpus=ngpus, shard_dir=shard_dir,
                cids_path=cids_path, residuals_path=residuals_path,
                batch_size=batch_size, doc_maxlen=doc_maxlen,
                checkpoint_every=checkpoint_every, resume=resume,
            )
        else:
            all_doclens, all_pids, num_embeddings = encode_and_compress_collection(
                model, collection, config, codec,
                work_dir=shard_dir,
                cids_path=cids_path, residuals_path=residuals_path,
                batch_size=batch_size, doc_maxlen=doc_maxlen,
                monitor=monitor,
                checkpoint_every=checkpoint_every, resume=resume,
            )

        # Commit Stage 2+3: codes are finalized; persist the per-doc metadata too.
        saver.save_doclens(all_doclens)
        saver.save_pids(all_pids)
        saver.save_build_state({"fingerprint": fingerprint, "stage": "encoded"})

    monitor.report("after Stage 2+3 (encode+compress)")

    # ------------------------------------------------------------------
    # Stage 4: Index Inversion
    # ------------------------------------------------------------------
    # Deterministic and idempotent — re-derived from the finalized centroid_ids.npy,
    # so a resume that reaches here simply rebuilds it (no re-encoding).
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
    # codec / centroid_ids.npy / packed_residuals.npy / doclens / pids were already
    # persisted by earlier stages; only the remaining artifacts need saving here.
    saver.save_inverted_lists(inverted_lists)
    saver.save_metadata({
        "num_passages": len(all_pids),
        "num_embeddings": num_embeddings,
        "num_centroids": int(codec.num_centroids),
        "dim": int(codec.dim),
        "nbits": config.nbits,
        "bytes_per_vector": codec.bytes_per_vector,
        "num_gpus_used": ngpus,
    })
    saver.save_build_state({"fingerprint": fingerprint, "stage": "complete"})

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
