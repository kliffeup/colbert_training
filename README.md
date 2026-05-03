# ColBERTv2 Training

A clean, from-scratch reimplementation of **ColBERTv2** (Contextualized Late Interaction over BERT v2) for passage retrieval, incorporating:

- **Denoised supervision** — cross-encoder distillation with KL-Divergence loss + in-batch negatives
- **Residual compression** — centroid-based encoding with quantized residuals (6–10× smaller index)
- **Custom inverted-list retrieval** — replaces FAISS nearest-neighbor search at query time

Full pipeline: training → indexing → retrieval → evaluation on **MS MARCO**, **BEIR** (13 datasets), and **LoTTE** (12 test sets).

## Installation

```bash
pip install -e .

# For GPU-accelerated FAISS (optional but recommended):
pip install faiss-gpu
```

## Quick Start

### 1. Download Data

```bash
# MS MARCO passage ranking dataset
bash scripts/download_msmarco.sh data/

# BEIR benchmarks (optional, for out-of-domain evaluation)
python scripts/download_beir.py --output_dir data/beir

# LoTTE benchmarks (optional, for out-of-domain evaluation)
python scripts/download_lotte.py --output_dir data/lotte
```

### 2. Phase 1: Triple-Based Training

Train with standard BM25 negatives (150k steps as pre-finetuning):

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml \
    --mode triples
```

### 3. Distillation Preparation

Index training passages, retrieve hard negatives, score with cross-encoder, and build 64-way tuples:

```bash
python scripts/distill.py \
    --config configs/default.yaml \
    --checkpoint experiments/checkpoints/phase1_final.pt
```

### 4. Phase 2: Distillation Training

Train with KL-Divergence distillation + in-batch cross-entropy (400k steps):

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml \
    --mode distill \
    --init_from experiments/checkpoints/phase1_final.pt \
    --tuples data/tuples/tuples.jsonl
```

### 5. Indexing

Build the ColBERTv2 index with residual compression:

```bash
python scripts/index.py \
    --config configs/default.yaml \
    --checkpoint experiments/checkpoints/phase2_final.pt
```

### 6. Evaluation

#### MS MARCO

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint experiments/checkpoints/phase2_final.pt \
    --index_path experiments/index
```

#### BEIR (zero-shot)

```bash
python scripts/evaluate_beir.py \
    --config configs/default.yaml \
    --checkpoint experiments/checkpoints/phase2_final.pt
```

#### LoTTE (zero-shot)

```bash
python scripts/evaluate_lotte.py \
    --config configs/default.yaml \
    --checkpoint experiments/checkpoints/phase2_final.pt
```

## Architecture

```
colbert/
├── config.py              # Configuration dataclass
├── modeling/
│   ├── colbert.py         # ColBERT model (shared BERT encoder + linear projection)
│   ├── tokenization.py    # Query/document tokenizers with [Q]/[D] markers
│   └── similarity.py      # MaxSim scoring function
├── data/
│   ├── download.py        # MS MARCO download utilities
│   ├── download_beir.py   # BEIR dataset download + conversion
│   ├── download_lotte.py  # LoTTE dataset download
│   ├── collection.py      # Passage collection reader
│   ├── queries.py         # Query reader
│   ├── triples.py         # Training triples dataset (Phase 1)
│   ├── distillation.py    # Distillation tuples dataset (Phase 2)
│   └── ranking.py         # Qrels and ranking utilities
├── training/
│   ├── trainer.py         # Phase 1 training loop (pairwise CE)
│   ├── distill_trainer.py # Phase 2 training loop (KL-Div + in-batch CE)
│   ├── loss.py            # Loss functions
│   └── utils.py           # Checkpointing, distributed helpers
├── distillation/
│   ├── score_passages.py  # Cross-encoder scoring
│   └── build_tuples.py    # Build distillation tuples
├── indexing/
│   ├── encoder.py         # Batch-encode collection passages
│   ├── residual_codec.py  # Residual compression codec
│   ├── index_builder.py   # Three-stage indexing pipeline
│   └── saver.py           # Index I/O
└── evaluation/
    ├── retriever.py       # Centroid-based retrieval + exact re-ranking
    ├── metrics.py         # MRR@10, Recall@k, nDCG@10, Success@5
    ├── beir_evaluator.py  # BEIR benchmark orchestrator
    └── lotte_evaluator.py # LoTTE benchmark orchestrator
```

## Key Design Choices

| Aspect | Detail |
|--------|--------|
| **Encoder** | `bert-base-uncased` (110M params), shared between queries and documents |
| **Embedding dim** | 128 (linear projection from 768) |
| **Query augmentation** | `[MASK]` padding to fixed length for soft expansion |
| **Phase 1 loss** | Pairwise softmax cross-entropy (BM25 negatives) |
| **Phase 2 loss** | KL-Divergence from MiniLM cross-encoder + in-batch CE |
| **Compression** | Centroid ID (4B) + 2-bit quantized residual (32B) = 36 bytes/vector |
| **Retrieval** | Centroid probing → decompress → approximate MaxSim → exact re-rank |
| **Index size** | ~25 GiB for MS MARCO (vs ~154 GiB uncompressed) |

## Expected Results

### MS MARCO Passage Ranking (dev set)

| Metric | ColBERT v1 | ColBERTv2 |
|--------|-----------|-----------|
| MRR@10 | 36.0% | **39.7%** |
| Recall@50 | 82.9% | **86.8%** |
| Recall@1000 | 96.8% | **98.4%** |

### BEIR (nDCG@10, zero-shot)

ColBERTv2 achieves state-of-the-art on most BEIR search tasks (DBPedia, FiQA, NQ, TREC-COVID, NFCorpus).

### LoTTE (Success@5, zero-shot)

ColBERTv2 outperforms all baselines across all topics for both search and forum query types.

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-5 | Learning rate |
| `warmup` | 20,000 | Linear warmup steps |
| `maxsteps` | 150,000 | Phase 1 training steps |
| `distill_maxsteps` | 400,000 | Phase 2 training steps |
| `nway` | 64 | Passages per distillation example |
| `nbits` | 2 | Bits per residual dimension |
| `nprobe` | 2 | Centroids probed per query token |

## References

- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)
- [MS MARCO Passage Ranking](https://microsoft.github.io/msmarco/)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
