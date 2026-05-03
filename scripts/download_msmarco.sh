#!/bin/bash
# Download MS MARCO passage ranking dataset using aria2c for parallel multi-connection downloads.
# Falls back to wget if aria2c is not available.
set -euo pipefail

DATA_DIR="${1:-data}"
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

# Collection
if [ ! -f "$DATA_DIR/collection.tsv" ]; then
    download "$BASE_URL/collection.tar.gz" "$DATA_DIR/collection.tar.gz"
    tar -xzf "$DATA_DIR/collection.tar.gz" -C "$DATA_DIR"
fi

# Queries
if [ ! -f "$DATA_DIR/queries.train.tsv" ] || [ ! -f "$DATA_DIR/queries.dev.small.tsv" ]; then
    download "$BASE_URL/queries.tar.gz" "$DATA_DIR/queries.tar.gz"
    tar -xzf "$DATA_DIR/queries.tar.gz" -C "$DATA_DIR"
fi

# Qrels
download "$BASE_URL/qrels.train.tsv" "$DATA_DIR/qrels.train.tsv"
download "$BASE_URL/qrels.dev.small.tsv" "$DATA_DIR/qrels.dev.small.tsv"

# Training triples (largest file ~4GB)
if [ ! -f "$DATA_DIR/triples.train.small.tsv" ]; then
    download "$BASE_URL/triples.train.small.tar.gz" "$DATA_DIR/triples.train.small.tar.gz"
    tar -xzf "$DATA_DIR/triples.train.small.tar.gz" -C "$DATA_DIR"
fi

echo "MS MARCO data ready in $DATA_DIR"
