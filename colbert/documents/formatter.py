"""Field-formatting strategies for document collections.

Used by preprocessing and any runtime path that combines fields (title, body, url) of a
multi-column corpus row into a single text string for the encoder.
"""

from __future__ import annotations

from typing import Mapping


SUPPORTED_STRATEGIES = ("body_only", "title_body", "url_title_body", "tagged")


def format_doc(row: Mapping[str, str], strategy: str, template: str | None = None) -> str:
    """Combine fields of a document row into a single text string.

    Args:
        row: Mapping with keys among {"docid", "url", "title", "body"}. Missing keys are
            treated as empty strings.
        strategy: One of SUPPORTED_STRATEGIES.
        template: Format string for the "tagged" strategy. Must reference field names
            via str.format placeholders, e.g. "<title>{title}</title><body>{body}</body>".

    Returns:
        Formatted text. Always non-None; empty fields collapse to surrounding whitespace.
    """
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()
    url = (row.get("url") or "").strip()

    if strategy == "body_only":
        return body
    if strategy == "title_body":
        return f"{title} {body}".strip()
    if strategy == "url_title_body":
        return f"{url} {title} {body}".strip()
    if strategy == "tagged":
        if not template:
            raise ValueError("strategy='tagged' requires a non-empty template.")
        safe_row = {"docid": row.get("docid", ""), "url": url, "title": title, "body": body}
        return template.format(**safe_row)

    raise ValueError(
        f"Unknown field_format strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}."
    )
