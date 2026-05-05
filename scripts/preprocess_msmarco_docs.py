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


# ---------------------------------------------------------------------------
# Mode: e2e
# ---------------------------------------------------------------------------

def run_e2e(
    input_dir: Path,
    output_dir: Path,
    field_format: str,
    field_format_template: str,
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

    # Triples are kept doc-level in e2e mode (they reference docids directly)
    src_triples = input_dir / "msmarco-doctriples.tsv"
    dst_triples = output_dir / "triples.docs.tsv"
    if src_triples.exists() and src_triples.resolve() != dst_triples.resolve():
        dst_triples.write_bytes(src_triples.read_bytes())
        logger.info(f"Copied triples: {dst_triples}")


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

    # Convert doc-level triples (qid<TAB>pos_docid<TAB>neg_docid or query<TAB>pos<TAB>neg)
    # to passage-level by mapping each docid to its first passage.
    src_triples = input_dir / "msmarco-doctriples.tsv"
    dst_triples = output_dir / "triples.passages.tsv"
    if not src_triples.exists():
        logger.warning(f"{src_triples} missing — skipping triples conversion")
        return

    n_kept = 0
    n_skipped = 0
    with open(src_triples, encoding="utf-8") as fin, open(dst_triples, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                n_skipped += 1
                continue
            # MS MARCO doctriples format: query<TAB>pos_docid<TAB>neg_docid
            # (text-form triples; use the docids from columns 1 and 2)
            query_or_qid, pos_docid, neg_docid = parts[0], parts[1], parts[2]
            pos_pid = doc_first_passage.get(pos_docid)
            neg_pid = doc_first_passage.get(neg_docid)
            if pos_pid is None or neg_pid is None:
                n_skipped += 1
                continue
            fout.write(f"{query_or_qid}\t{pos_pid}\t{neg_pid}\n")
            n_kept += 1
    logger.info(
        f"Converted triples: {n_kept:,} kept, {n_skipped:,} skipped (unknown docid). "
        f"Output: {dst_triples}"
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
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "e2e":
        run_e2e(
            input_dir=args.input,
            output_dir=args.output,
            field_format=args.field_format,
            field_format_template=args.field_format_template,
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
        )


if __name__ == "__main__":
    main()
