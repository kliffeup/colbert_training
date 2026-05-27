#!/usr/bin/env python3
"""Download LoTTE benchmark datasets.

Usage:
    python scripts/download_lotte.py --output_dir data/lotte
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colbert.dataset.download_lotte import download_lotte


def main():
    parser = argparse.ArgumentParser(description="Download LoTTE Datasets")
    parser.add_argument("--output_dir", type=str, default="data/lotte")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    download_lotte(args.output_dir)


if __name__ == "__main__":
    main()
