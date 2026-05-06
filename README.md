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

## Document Training (MS MARCO Doc v1)

The same scripts also train and evaluate on full **documents** (longer than passages,
~3.2M docs in MS MARCO Document v1). Two retrieval-time strategies are supported via
the `task` and `doc_segmentation` config keys:

| `task` | `doc_segmentation` | Strategy |
|---|---|---|
| `passage` | (ignored) | Default passage retrieval (above) |
| `document` | `none` | End-to-end: each doc encoded as one sequence with `doc_maxlen` up to the encoder's `model_max_length` (512 for BERT, **8192 for ModernBERT**) |
| `document` | `maxp` | Documents pre-segmented into overlapping passages; index keyed by passage IDs; retrieval aggregates max-per-doc |

### 1. Download MS MARCO Doc v1

```bash
bash scripts/download_msmarco_docs.sh data/docs
```

Fetches `msmarco-docs.tsv` (~22 GB compressed), train/dev queries + qrels, and the
BM25 top-100 candidates per training query. Per-file fetches are isolated, so a single
404/timeout won't abort the rest. **No pre-mined triples are downloaded** — Microsoft
doesn't ship them for the document task; the preprocess step mines triples from
`msmarco-doctrain-top100` + `msmarco-doctrain-qrels.tsv`.

### 2. Preprocess

End-to-end (one row per doc, full untruncated text):

```bash
python scripts/preprocess_msmarco_docs.py \
    --mode e2e --input data/docs --output data/docs \
    --format title_body
```

MaxP (sliding-window passages):

```bash
python scripts/preprocess_msmarco_docs.py \
    --mode maxp --input data/docs --output data/docs \
    --tokenizer bert-base-uncased \
    --passage-window 180 --passage-stride 90
```

Both modes also mine Phase 1 training triples (`triples.docs.tsv` for e2e or
`triples.passages.tsv` for maxp) from `msmarco-doctrain-top100`. Two strategies:

| `--negative-strategy` | Behavior |
|---|---|
| `random` (default) | Uniformly sample N negatives per (query, positive) — more diversity. |
| `top` | Take the top-N BM25-scored negatives — deterministic, harder, no diversity. |

Other knobs: `--negatives-per-positive` (default 4), `--seed` (only used by `random`).

Field formatting strategies (`--format`): `body_only`, `title_body`, `url_title_body`, `tagged`.
The `tagged` strategy uses `--field-format-template` (default
`<title>{title}</title><body>{body}</body>`) which lets long-context models distinguish
field boundaries.

### 3. Train + Index + Evaluate

The same scripts work — just point at a document config:

```bash
# End-to-end with BERT (512-token cap)
torchrun --nproc_per_node=4 scripts/train.py --config configs/document_e2e_bert.yaml --mode triples

# End-to-end with ModernBERT (8192-token cap)
torchrun --nproc_per_node=4 scripts/train.py --config configs/document_e2e_modernbert.yaml --mode triples

# MaxP with BERT
torchrun --nproc_per_node=4 scripts/train.py --config configs/document_maxp.yaml --mode triples
```

Indexing and evaluation use the same scripts; in MaxP mode evaluation transparently
retrieves K' = `retrieve_top_k * max_passages_per_doc_factor` passages and aggregates
to the top `retrieve_top_k` documents.

### Field-level masking ("encode whole, score part")

Useful when you want the encoder to *see* the full document (so the indexed positions
are context-aware) but only some fields to *contribute* to MaxSim scoring and the index.
For example, encode `<title>...</title><body>...</body>` end-to-end but score on title
only.

The mechanism: register sentinel tokens (e.g. `[TITLE_BEGIN]`/`[TITLE_END]`) as additional
special tokens so each is a single atomic token id, then mask out everything outside the
declared indexed fields after the encoder. Body positions get hard-zeroed before MaxSim,
so they contribute nothing to score / loss / index — but they still influence title
embeddings via self-attention.

In `configs/document_e2e_modernbert.yaml`:

```yaml
field_format: "tagged"
field_format_template: "[TITLE_BEGIN]{title}[TITLE_END][BODY_BEGIN]{body}[BODY_END]"
field_markers:
  title: ["[TITLE_BEGIN]", "[TITLE_END]"]
  body:  ["[BODY_BEGIN]",  "[BODY_END]"]
indexed_fields: ["title"]      # only title content scores
index_field_markers: false     # drop the markers themselves from the index
index_special_tokens: true     # keep [CLS]/[SEP]/[D] regardless
```

The same mask flows through training, indexing, and retrieval — masked positions are
zeroed post-encoder ([colbert/modeling/colbert.py:80](colbert/modeling/colbert.py#L80)),
so MaxSim, in-batch CE, and KL-distillation losses all only see the indexed fields.

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
