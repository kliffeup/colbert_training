"""Document-level retrieval wrapper for MaxP-style indexes.

For ``task=document, doc_segmentation=maxp``, the index is keyed by passage IDs
(``D123_p0``, ``D123_p1``, ...). At evaluation time we want top-K *documents*, so
we ask the underlying retriever for K' = K * factor passage hits and aggregate
them via :func:`colbert.documents.passage_doc_map.aggregate`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from colbert.config import ColBERTConfig
from colbert.documents.passage_doc_map import PassageDocMap, aggregate
from colbert.evaluation.retriever import ColBERTRetriever

logger = logging.getLogger(__name__)


def retrieve_documents(
    retriever: ColBERTRetriever,
    queries: List[str],
    config: ColBERTConfig,
    pmap: PassageDocMap | None = None,
) -> Dict[int, List[Tuple[str, float]]]:
    """Run passage-level retrieval and aggregate to doc-level.

    Args:
        retriever: A loaded ColBERTRetriever pointing at a MaxP passage index.
        queries: Query texts.
        config: ColBERTConfig (uses retrieve_top_k, max_passages_per_doc_factor,
            passage_to_doc_map).
        pmap: Optional pre-loaded PassageDocMap; if None, loads from config path.

    Returns:
        Dict mapping query index -> [(docid, score), ...] sorted descending,
        truncated to retrieve_top_k.
    """
    if pmap is None:
        logger.info(f"Loading passage->doc map from {config.passage_to_doc_map}")
        pmap = PassageDocMap.load(config.passage_to_doc_map)

    top_k_docs = config.retrieve_top_k
    top_k_passages = top_k_docs * max(1, config.max_passages_per_doc_factor)

    logger.info(
        f"Retrieving top-{top_k_passages} passages, then aggregating to top-{top_k_docs} docs."
    )
    passage_ranking = retriever.retrieve(queries, top_k=top_k_passages)
    return aggregate(passage_ranking, pmap, top_k=top_k_docs)
