"""LoTTE download utilities — not implemented in this MS MARCO-focused checkout.

The MS MARCO document training pipeline does not need LoTTE. Port from upstream
Stanford ColBERT `colbert/data/download_lotte.py` if you want it.
"""

from __future__ import annotations


def download_lotte(*args, **kwargs):  # noqa: D401
    raise NotImplementedError(
        "LoTTE download is not implemented in this checkout. Port the upstream "
        "Stanford ColBERT data/download_lotte.py if needed."
    )
