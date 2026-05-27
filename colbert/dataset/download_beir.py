"""BEIR download utilities — not implemented in this MS MARCO-focused checkout.

The MS MARCO document training pipeline does not need BEIR. If you want
BEIR zero-shot evaluation, install the official `beir` package and use
its dataset utilities, or port the upstream Stanford ColBERT
`colbert/data/download_beir.py` into this module.
"""

from __future__ import annotations

# Kept as a placeholder so callers can `import` without crashing at module load.
BEIR_DATASETS: list[str] = []


def download_beir_datasets(*args, **kwargs):  # noqa: D401
    raise NotImplementedError(
        "BEIR download is not implemented in this checkout. Use the `beir` package "
        "or port the upstream Stanford ColBERT data/download_beir.py."
    )
