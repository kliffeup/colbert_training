#!/usr/bin/env python3
"""Download BEIR benchmark datasets and convert to TSV format.

Usage:
    python scripts/download_beir.py --output_dir data/beir
    python scripts/download_beir.py --output_dir data/beir --datasets nq fiqa scifact
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colbert.data.download_beir import download_beir_datasets, BEIR_DATASETS


def main():
    parser = argparse.ArgumentParser(description="Download BEIR Datasets")
    parser.add_argument("--output_dir", type=str, default="data/beir")
    parser.add_argument("--datasets", nargs="*", default=None, help="Specific datasets to download")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    datasets = args.datasets or BEIR_DATASETS
    download_beir_datasets(args.output_dir, datasets)


if __name__ == "__main__":
    main()
