#!/bin/bash
# Download MS MARCO Document Ranking v1 dataset using aria2c for parallel multi-connection
# downloads. Falls back to wget if aria2c is not available.
#
# Files fetched:
#   msmarco-docs.tsv.gz             (corpus: docid \t url \t title \t body)
#   msmarco-doctrain-queries.tsv.gz
#   msmarco-doctrain-qrels.tsv.gz   (TREC format: qid 0 docid 1)
#   msmarco-doctrain-top100.gz      (BM25 top-100 per training query, optional)
#   msmarco-doctriples.tsv.gz       (BM25-mined triples for Phase 1)
#   msmarco-docdev-queries.tsv.gz
#   msmarco-docdev-qrels.tsv.gz
#
# Usage:  bash scripts/download_msmarco_docs.sh [data_dir] [aria2c_connections]
set -euo pipefail

DATA_DIR="${1:-data/docs}"
CONNECTIONS="${2:-8}"
mkdir -p "$DATA_DIR"

BASE_URL="https://msmarco.z22.web.core.windows.net/msmarcoranking"

if command -v aria2c &>/dev/null; then
    DOWNLOADER="aria2c"
    echo "Using aria2c with $CONNECTIONS connections per file"
else
    DOWNLOADER="wget"
    echo "aria2c not found, falling back to wget (single connection)"
    echo "Install aria2c for faster downloads: sudo apt install aria2"
fi

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "Already exists: $dest"
        return
    fi
    echo "Downloading: $url -> $dest"
    if [ "$DOWNLOADER" = "aria2c" ]; then
        aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 1M \
            --dir="$(dirname "$dest")" --out="$(basename "$dest")" \
            --console-log-level=warn --summary-interval=5 \
            "$url"
    else
        wget -q --show-progress -O "$dest" "$url"
    fi
}

# 1. Document corpus (largest file: ~22 GB compressed)
if [ ! -f "$DATA_DIR/msmarco-docs.tsv" ]; then
    download "$BASE_URL/msmarco-docs.tsv.gz" "$DATA_DIR/msmarco-docs.tsv.gz"
    echo "Decompressing msmarco-docs.tsv.gz ..."
    gunzip -k "$DATA_DIR/msmarco-docs.tsv.gz"
fi

# 2. Train queries
if [ ! -f "$DATA_DIR/msmarco-doctrain-queries.tsv" ]; then
    download "$BASE_URL/msmarco-doctrain-queries.tsv.gz" "$DATA_DIR/msmarco-doctrain-queries.tsv.gz"
    gunzip -k "$DATA_DIR/msmarco-doctrain-queries.tsv.gz"
fi

# 3. Train qrels (TREC format: qid 0 docid 1)
if [ ! -f "$DATA_DIR/msmarco-doctrain-qrels.tsv" ]; then
    download "$BASE_URL/msmarco-doctrain-qrels.tsv.gz" "$DATA_DIR/msmarco-doctrain-qrels.tsv.gz"
    gunzip -k "$DATA_DIR/msmarco-doctrain-qrels.tsv.gz"
fi

# 4. Train BM25 top-100 candidates (optional, used for distillation hard negatives)
if [ ! -f "$DATA_DIR/msmarco-doctrain-top100" ]; then
    download "$BASE_URL/msmarco-doctrain-top100.gz" "$DATA_DIR/msmarco-doctrain-top100.gz"
    gunzip -k "$DATA_DIR/msmarco-doctrain-top100.gz"
fi

# 5. Pre-mined doc-level triples for Phase 1
if [ ! -f "$DATA_DIR/msmarco-doctriples.tsv" ]; then
    download "$BASE_URL/msmarco-doctriples.tsv.gz" "$DATA_DIR/msmarco-doctriples.tsv.gz"
    gunzip -k "$DATA_DIR/msmarco-doctriples.tsv.gz"
fi

# 6. Dev queries + qrels
if [ ! -f "$DATA_DIR/msmarco-docdev-queries.tsv" ]; then
    download "$BASE_URL/msmarco-docdev-queries.tsv.gz" "$DATA_DIR/msmarco-docdev-queries.tsv.gz"
    gunzip -k "$DATA_DIR/msmarco-docdev-queries.tsv.gz"
fi
if [ ! -f "$DATA_DIR/msmarco-docdev-qrels.tsv" ]; then
    download "$BASE_URL/msmarco-docdev-qrels.tsv.gz" "$DATA_DIR/msmarco-docdev-qrels.tsv.gz"
    gunzip -k "$DATA_DIR/msmarco-docdev-qrels.tsv.gz"
fi

echo
echo "MS MARCO Document v1 ready in $DATA_DIR"
echo "Next: run scripts/preprocess_msmarco_docs.py to build the collection / passages."
