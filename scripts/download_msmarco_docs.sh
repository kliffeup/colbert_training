#!/bin/bash
# Download MS MARCO Document Ranking v1 dataset using aria2c for parallel multi-connection
# downloads. Falls back to wget if aria2c is not available.
#
# Files fetched:
#   msmarco-docs.tsv.gz             (corpus: docid \t url \t title \t body)
#   msmarco-doctrain-queries.tsv.gz
#   msmarco-doctrain-qrels.tsv.gz   (TREC format: qid 0 docid 1)
#   msmarco-doctrain-top100.gz      (BM25 top-100 per training query)
#   msmarco-docdev-queries.tsv.gz
#   msmarco-docdev-qrels.tsv.gz
#
# NOTE: there is no pre-built `msmarco-doctriples.tsv` in the official distribution;
# Microsoft only ships triples for the passage task. Phase 1 triples are mined from
# top100 + qrels by `scripts/preprocess_msmarco_docs.py`.
#
# Per-file fetches are isolated so a single 404 does not abort the rest of the run.
#
# Usage:  bash scripts/download_msmarco_docs.sh [data_dir] [aria2c_connections]
set -uo pipefail

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
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "Already exists: $dest"
        return 0
    fi
    # Drop any zero-byte placeholder from a previously-failed run so the downloader
    # treats this as a fresh attempt.
    [ -f "$dest" ] && [ ! -s "$dest" ] && rm -f "$dest"
    echo "Downloading: $url -> $dest"
    local rc=0
    if [ "$DOWNLOADER" = "aria2c" ]; then
        aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 1M \
            --dir="$(dirname "$dest")" --out="$(basename "$dest")" \
            --console-log-level=warn --summary-interval=5 \
            "$url" || rc=$?
    else
        wget -q --show-progress -O "$dest" "$url" || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        echo "WARNING: download failed for $url (exit $rc) — continuing." >&2
        # Remove zero-byte file the failed downloader may have left behind.
        [ -f "$dest" ] && [ ! -s "$dest" ] && rm -f "$dest"
        return "$rc"
    fi
}

# Each fetch_and_unpack handles its own errors so a single 404 / network blip does not
# abort the rest of the script.
fetch_and_unpack() {
    local url="$1"
    local gz_dest="$2"
    local final="$3"
    if [ -f "$final" ] && [ -s "$final" ]; then
        echo "Already exists: $final"
        return 0
    fi
    if ! download "$url" "$gz_dest"; then
        return 0  # warning already printed by download()
    fi
    if ! gunzip -k "$gz_dest"; then
        echo "WARNING: gunzip failed for $gz_dest — file may be corrupt." >&2
        rm -f "$gz_dest"
    fi
}

# 1. Document corpus (largest file: ~22 GB compressed)
fetch_and_unpack "$BASE_URL/msmarco-docs.tsv.gz" \
    "$DATA_DIR/msmarco-docs.tsv.gz" "$DATA_DIR/msmarco-docs.tsv"

# 2. Train queries
fetch_and_unpack "$BASE_URL/msmarco-doctrain-queries.tsv.gz" \
    "$DATA_DIR/msmarco-doctrain-queries.tsv.gz" "$DATA_DIR/msmarco-doctrain-queries.tsv"

# 3. Train qrels (TREC format: qid 0 docid 1)
fetch_and_unpack "$BASE_URL/msmarco-doctrain-qrels.tsv.gz" \
    "$DATA_DIR/msmarco-doctrain-qrels.tsv.gz" "$DATA_DIR/msmarco-doctrain-qrels.tsv"

# 4. Train BM25 top-100 candidates — used by preprocess to mine Phase 1 triples and
#    by Phase 2 distillation as the hard-negative pool.
fetch_and_unpack "$BASE_URL/msmarco-doctrain-top100.gz" \
    "$DATA_DIR/msmarco-doctrain-top100.gz" "$DATA_DIR/msmarco-doctrain-top100"

# 5. Dev queries + qrels
fetch_and_unpack "$BASE_URL/msmarco-docdev-queries.tsv.gz" \
    "$DATA_DIR/msmarco-docdev-queries.tsv.gz" "$DATA_DIR/msmarco-docdev-queries.tsv"
fetch_and_unpack "$BASE_URL/msmarco-docdev-qrels.tsv.gz" \
    "$DATA_DIR/msmarco-docdev-qrels.tsv.gz" "$DATA_DIR/msmarco-docdev-qrels.tsv"

echo
echo "MS MARCO Document v1 raw files ready in $DATA_DIR"
echo "Next: run scripts/preprocess_msmarco_docs.py — it will also mine Phase 1 triples"
echo "      from msmarco-doctrain-top100 + msmarco-doctrain-qrels.tsv."
