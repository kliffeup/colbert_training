"""Preprocess MS MARCO Document v1 raw files into the formats consumed by training/indexing.

Two modes:

1. ``e2e`` — end-to-end full-document training. Apply a configurable field-format
   strategy (body_only / title_body / url_title_body / tagged) and write a single
   ``collection.docs.tsv`` (``docid<TAB>text``). **No character truncation** — the
   tokenizer enforces length later via ``doc_maxlen``.

2. ``maxp`` — sliding-window passage segmentation. Tokenize each formatted document
   with the configured HF tokenizer, slide a (window, stride) over the token IDs,
   write per-passage rows to ``collection.passages.tsv`` plus a
   ``passage_to_doc.tsv`` mapping.

Both modes also normalize the TREC-format qrels (``qid 0 docid rel``) into the
3-column form (``qid<TAB>docid<TAB>rel``) the rest of the pipeline expects, and
copy queries through unchanged.

Doc-level triples (``msmarco-doctriples.tsv``) are converted accordingly:
  - e2e: passed through (docids are already the IDs we use)
  - maxp: positive docid → first passage of that doc; negative docid → first passage
    of the negative doc.

Usage:
    python scripts/preprocess_msmarco_docs.py --mode e2e --input data/docs --output data/docs
    python scripts/preprocess_msmarco_docs.py --mode maxp --input data/docs --output data/docs \
        --tokenizer bert-base-uncased --passage-window 180 --passage-stride 90
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colbert.documents.formatter import SUPPORTED_STRATEGIES, format_doc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tsv_safe(text: str) -> str:
    """Replace tab/newline characters with spaces so a single doc fits on one TSV line.
    Does NOT truncate — full content is preserved."""
    if text is None:
        return ""
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _iter_docs(input_dir: Path):
    """Yield dict rows from msmarco-docs.tsv. Cols: docid, url, title, body."""
    path = input_dir / "msmarco-docs.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/download_msmarco_docs.sh first."
        )
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            # docs.tsv has 4 columns; defensively pad if some are missing
            while len(parts) < 4:
                parts.append("")
            docid, url, title, body = parts[0], parts[1], parts[2], "\t".join(parts[3:])
            yield {"docid": docid, "url": url, "title": title, "body": body}
            if line_no % 100_000 == 0:
                logger.info(f"  ...read {line_no:,} lines")


def _normalize_qrels(src: Path, dst: Path) -> None:
    """TREC qrels (qid 0 docid rel) -> 3-col qrels (qid<TAB>docid<TAB>rel).

    Whitespace-separated input; the leading "0" iteration field is dropped.
    """
    if not src.exists():
        logger.warning(f"qrels not found: {src} — skipping")
        return
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            fout.write(f"{qid}\t{docid}\t{rel}\n")
    logger.info(f"Wrote qrels: {dst}")


def _copy_queries(src: Path, dst: Path) -> None:
    if not src.exists():
        logger.warning(f"queries not found: {src} — skipping")
        return
    if dst.resolve() == src.resolve():
        return
    dst.write_bytes(src.read_bytes())
    logger.info(f"Wrote queries: {dst}")


def _load_train_queries(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, text = parts
            out[qid] = text
    return out


def _load_train_positives(path: Path) -> dict[str, set[str]]:
    """Read TREC-format qrels (`qid 0 docid rel`, whitespace-separated) -> {qid: {docids}}."""
    if not path.exists():
        return {}
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                rel_i = int(rel)
            except ValueError:
                continue
            if rel_i <= 0:
                continue
            out.setdefault(qid, set()).add(docid)
    return out


def _iter_top100(path: Path):
    """Yield (qid, docid) in BM25-rank order from a TREC run file.

    Format: `qid Q0 docid rank score runname` (whitespace-separated).
    """
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            qid, _, docid = parts[0], parts[1], parts[2]
            yield qid, docid


NEGATIVE_STRATEGIES = ("random", "top")


def build_triples(
    input_dir: Path,
    output_path: Path,
    negatives_per_positive: int,
    seed: int,
    negative_strategy: str = "random",
    docid_remap: dict[str, str] | None = None,
) -> None:
    """Mine ``query<TAB>pos<TAB>neg`` triples from BM25 top-100 + qrels.

    Args:
        input_dir: Directory containing ``msmarco-doctrain-{queries,qrels}.tsv`` and
            ``msmarco-doctrain-top100``.
        output_path: TSV target.
        negatives_per_positive: Number of negatives per (query, positive) pair.
        seed: RNG seed (only used when ``negative_strategy='random'``).
        negative_strategy: How to pick negatives from the BM25 candidate list (after
            removing labeled positives):
              * ``'random'`` — uniform random sample of N (default, gives variety).
              * ``'top'``    — take the first N candidates in BM25-rank order, i.e. the
                               highest-scored negatives. Deterministic; produces harder
                               negatives at the cost of less diversity.
        docid_remap: Optional ``docid -> id`` mapping to apply to both pos and neg before
            writing (e.g. doc -> first-passage id for maxp mode). Triples whose pos or neg
            cannot be mapped are dropped.
    """
    import random

    if negative_strategy not in NEGATIVE_STRATEGIES:
        raise ValueError(
            f"unknown negative_strategy={negative_strategy!r}; "
            f"expected one of {NEGATIVE_STRATEGIES}"
        )

    qrels_path = input_dir / "msmarco-doctrain-qrels.tsv"
    queries_path = input_dir / "msmarco-doctrain-queries.tsv"
    top100_path = input_dir / "msmarco-doctrain-top100"

    positives = _load_train_positives(qrels_path)
    queries = _load_train_queries(queries_path)
    if not positives or not queries or not top100_path.exists():
        logger.warning(
            f"Cannot mine triples — missing one of "
            f"qrels={qrels_path.exists()} / queries={queries_path.exists()} / "
            f"top100={top100_path.exists()}. Skipping."
        )
        return

    logger.info(
        f"Mining triples from {top100_path.name} + qrels "
        f"({len(positives):,} queries with positives, {negatives_per_positive} neg/pos, "
        f"strategy={negative_strategy}"
        + (f", seed={seed}" if negative_strategy == 'random' else "")
        + ")"
    )

    # Group top-100 candidates by qid (preserving rank order). Streams once.
    qid_to_candidates: dict[str, list[str]] = {}
    for qid, docid in _iter_top100(top100_path):
        qid_to_candidates.setdefault(qid, []).append(docid)

    rng = random.Random(seed)
    n_triples = 0
    n_no_candidates = 0
    n_remap_dropped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for qid, pos_docids in positives.items():
            query_text = queries.get(qid)
            if query_text is None:
                continue
            candidates = qid_to_candidates.get(qid, [])
            negative_pool = [d for d in candidates if d not in pos_docids]
            if not negative_pool:
                n_no_candidates += 1
                continue
            for pos_docid in pos_docids:
                k = min(negatives_per_positive, len(negative_pool))
                if negative_strategy == "top":
                    # Highest-scored negatives — first k in BM25-rank order.
                    negs = negative_pool[:k]
                else:
                    negs = rng.sample(negative_pool, k)
                for neg_docid in negs:
                    if docid_remap is not None:
                        pos_id = docid_remap.get(pos_docid)
                        neg_id = docid_remap.get(neg_docid)
                        if pos_id is None or neg_id is None:
                            n_remap_dropped += 1
                            continue
                    else:
                        pos_id, neg_id = pos_docid, neg_docid
                    fout.write(f"{query_text}\t{pos_id}\t{neg_id}\n")
                    n_triples += 1

    logger.info(
        f"Wrote {n_triples:,} triples to {output_path}"
        + (f" ({n_no_candidates:,} queries had no negative candidates)" if n_no_candidates else "")
        + (f" ({n_remap_dropped:,} dropped due to missing remap)" if n_remap_dropped else "")
    )


# ---------------------------------------------------------------------------
# Mode: e2e
# ---------------------------------------------------------------------------

def run_e2e(
    input_dir: Path,
    output_dir: Path,
    field_format: str,
    field_format_template: str,
    negatives_per_positive: int,
    seed: int,
    negative_strategy: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_collection = output_dir / "collection.docs.tsv"
    n = 0
    with open(out_collection, "w", encoding="utf-8") as fout:
        for row in _iter_docs(input_dir):
            text = format_doc(row, field_format, field_format_template)
            fout.write(f"{row['docid']}\t{_tsv_safe(text)}\n")
            n += 1
    logger.info(f"Wrote {n:,} docs to {out_collection}")

    _normalize_qrels(
        input_dir / "msmarco-doctrain-qrels.tsv",
        output_dir / "qrels.docs.train.tsv",
    )
    _normalize_qrels(
        input_dir / "msmarco-docdev-qrels.tsv",
        output_dir / "qrels.docs.dev.tsv",
    )
    _copy_queries(
        input_dir / "msmarco-doctrain-queries.tsv",
        output_dir / "queries.docs.train.tsv",
    )
    _copy_queries(
        input_dir / "msmarco-docdev-queries.tsv",
        output_dir / "queries.docs.dev.tsv",
    )

    # MS MARCO Doc v1 does not ship pre-mined triples; build them from BM25 top-100 + qrels.
    build_triples(
        input_dir=input_dir,
        output_path=output_dir / "triples.docs.tsv",
        negatives_per_positive=negatives_per_positive,
        seed=seed,
        negative_strategy=negative_strategy,
    )


# ---------------------------------------------------------------------------
# Mode: maxp
# ---------------------------------------------------------------------------

def run_maxp(
    input_dir: Path,
    output_dir: Path,
    field_format: str,
    field_format_template: str,
    tokenizer_name: str,
    window: int,
    stride: int,
    negatives_per_positive: int,
    seed: int,
    negative_strategy: str,
) -> None:
    from transformers import AutoTokenizer

    from colbert.documents.segmenter import segment

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    out_passages = output_dir / "collection.passages.tsv"
    out_pmap = output_dir / "passage_to_doc.tsv"

    doc_first_passage: dict[str, str] = {}  # docid -> first passage_id (for triples)
    n_docs = 0
    n_passages = 0

    with open(out_passages, "w", encoding="utf-8") as fpass, \
         open(out_pmap, "w", encoding="utf-8") as fmap:
        for row in _iter_docs(input_dir):
            docid = row["docid"]
            text = format_doc(row, field_format, field_format_template)
            passages = segment(text, tokenizer, window=window, stride=stride)
            for idx, ptext in enumerate(passages):
                pid = f"{docid}_p{idx}"
                fpass.write(f"{pid}\t{_tsv_safe(ptext)}\n")
                fmap.write(f"{pid}\t{docid}\n")
                if idx == 0:
                    doc_first_passage[docid] = pid
                n_passages += 1
            n_docs += 1
            if n_docs % 50_000 == 0:
                logger.info(f"  ...segmented {n_docs:,} docs into {n_passages:,} passages")

    logger.info(f"Wrote {n_docs:,} docs / {n_passages:,} passages to {out_passages}")
    logger.info(f"Wrote passage->doc map: {out_pmap}")

    # Doc-level qrels and queries are kept as-is; aggregation happens at retrieval.
    _normalize_qrels(
        input_dir / "msmarco-doctrain-qrels.tsv",
        output_dir / "qrels.docs.train.tsv",
    )
    _normalize_qrels(
        input_dir / "msmarco-docdev-qrels.tsv",
        output_dir / "qrels.docs.dev.tsv",
    )
    _copy_queries(
        input_dir / "msmarco-doctrain-queries.tsv",
        output_dir / "queries.docs.train.tsv",
    )
    _copy_queries(
        input_dir / "msmarco-docdev-queries.tsv",
        output_dir / "queries.docs.dev.tsv",
    )

    # Mine triples from BM25 top-100 + qrels and remap docids to their first passage.
    build_triples(
        input_dir=input_dir,
        output_path=output_dir / "triples.passages.tsv",
        negatives_per_positive=negatives_per_positive,
        seed=seed,
        negative_strategy=negative_strategy,
        docid_remap=doc_first_passage,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["e2e", "maxp"])
    p.add_argument("--input", required=True, type=Path, help="Dir with msmarco-docs.tsv etc.")
    p.add_argument("--output", required=True, type=Path, help="Dir for preprocessed files.")
    p.add_argument(
        "--format",
        dest="field_format",
        default="title_body",
        choices=SUPPORTED_STRATEGIES,
        help="How to combine docid/url/title/body fields into the indexed text.",
    )
    p.add_argument(
        "--field-format-template",
        default="<title>{title}</title><body>{body}</body>",
        help="Format string used when --format=tagged. Available placeholders: "
             "{docid} {url} {title} {body}.",
    )
    p.add_argument(
        "--tokenizer",
        default="bert-base-uncased",
        help="HF tokenizer name; used only in --mode maxp for sliding-window segmentation.",
    )
    p.add_argument("--passage-window", type=int, default=180)
    p.add_argument("--passage-stride", type=int, default=90)
    p.add_argument(
        "--negatives-per-positive",
        type=int,
        default=4,
        help="Number of BM25-top100 negatives sampled per (query, positive) pair when "
             "mining Phase 1 triples.",
    )
    p.add_argument(
        "--negative-strategy",
        choices=NEGATIVE_STRATEGIES,
        default="random",
        help="How to pick negatives from the BM25 candidate list: "
             "'random' = uniform sample of N (default, more variety); "
             "'top' = take the top-N highest-scored candidates in BM25 rank order "
             "(deterministic, harder negatives, no diversity).",
    )
    p.add_argument(
        "--seed", type=int, default=12345,
        help="RNG seed for reproducible negative sampling (used only by --negative-strategy=random).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "e2e":
        run_e2e(
            input_dir=args.input,
            output_dir=args.output,
            field_format=args.field_format,
            field_format_template=args.field_format_template,
            negatives_per_positive=args.negatives_per_positive,
            seed=args.seed,
            negative_strategy=args.negative_strategy,
        )
    else:
        run_maxp(
            input_dir=args.input,
            output_dir=args.output,
            field_format=args.field_format,
            field_format_template=args.field_format_template,
            tokenizer_name=args.tokenizer,
            window=args.passage_window,
            stride=args.passage_stride,
            negatives_per_positive=args.negatives_per_positive,
            seed=args.seed,
            negative_strategy=args.negative_strategy,
        )


if __name__ == "__main__":
    main()
