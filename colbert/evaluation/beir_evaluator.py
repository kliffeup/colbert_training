"""BEIR benchmark zero-shot evaluation orchestrator."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from colbert.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.dataset.collection import Collection
from colbert.dataset.queries import Queries
from colbert.dataset.ranking import save_ranking
from colbert.indexing.index_builder import build_index
from colbert.evaluation.retriever import ColBERTRetriever
from colbert.evaluation.metrics import ndcg_at_k, evaluate_ranking

logger = logging.getLogger(__name__)

BEIR_DATASETS = [
    "dbpedia-entity",
    "fiqa",
    "nq",
    "hotpotqa",
    "nfcorpus",
    "trec-covid",
    "touche-2020",
    "arguana",
    "climate-fever",
    "fever",
    "quora",
    "scidocs",
    "scifact",
]

# Datasets requiring special query max lengths
QUERY_MAXLEN_OVERRIDES = {
    "arguana": 300,
    "climate-fever": 64,
}


def _load_beir_dataset(
    dataset_name: str, beir_data_dir: str
) -> Tuple[str, str, str]:
    """Load BEIR dataset files.

    Returns paths to (collection_tsv, queries_tsv, qrels_tsv).
    """
    dataset_dir = Path(beir_data_dir) / dataset_name

    corpus_path = dataset_dir / "collection.tsv"
    queries_path = dataset_dir / "queries.tsv"
    qrels_path = dataset_dir / "qrels.tsv"

    if not corpus_path.exists():
        _convert_beir_to_tsv(dataset_name, beir_data_dir)

    return str(corpus_path), str(queries_path), str(qrels_path)


def _convert_beir_to_tsv(dataset_name: str, beir_data_dir: str) -> None:
    """Download BEIR dataset using the beir library and convert to TSV format."""
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    dataset_dir = Path(beir_data_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    download_dir = Path(beir_data_dir) / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    data_path = beir_util.download_and_unzip(url, str(download_dir))

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    with open(dataset_dir / "collection.tsv", "w", encoding="utf-8") as f:
        for pid, doc in corpus.items():
            text = doc.get("title", "") + " " + doc.get("text", "")
            text = text.strip().replace("\t", " ").replace("\n", " ")
            f.write(f"{pid}\t{text}\n")

    with open(dataset_dir / "queries.tsv", "w", encoding="utf-8") as f:
        for qid, query in queries.items():
            query = query.strip().replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{query}\n")

    with open(dataset_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        for qid, rels in qrels.items():
            for pid, rel in rels.items():
                f.write(f"{qid}\t0\t{pid}\t{rel}\n")

    logger.info(f"Converted BEIR dataset '{dataset_name}' to TSV at {dataset_dir}")


def evaluate_beir(
    model: ColBERT,
    config: ColBERTConfig,
    datasets: List[str] | None = None,
    batch_size: int = 128,
) -> Dict[str, Dict[str, float]]:
    """Run zero-shot evaluation on BEIR benchmark datasets.

    Args:
        model: Trained ColBERT model.
        config: Configuration.
        datasets: List of BEIR dataset names (default: all).
        batch_size: Encoding batch size.

    Returns:
        Dict mapping dataset_name -> {metric_name: value}.
    """
    datasets = datasets or BEIR_DATASETS
    results: Dict[str, Dict[str, float]] = {}

    for dataset_name in datasets:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating on BEIR: {dataset_name}")
        logger.info(f"{'=' * 60}")

        try:
            corpus_path, queries_path, qrels_path = _load_beir_dataset(
                dataset_name, config.beir_data_dir
            )
        except Exception as e:
            logger.error(f"Failed to load BEIR dataset '{dataset_name}': {e}")
            continue

        collection = Collection(corpus_path)
        queries = Queries(queries_path)

        from colbert.dataset.ranking import load_qrels
        qrels = load_qrels(qrels_path)

        query_maxlen = QUERY_MAXLEN_OVERRIDES.get(dataset_name, config.query_maxlen)
        eval_config = config.override(doc_maxlen=config.beir_doc_maxlen)

        index_dir = os.path.join(config.output_dir, "beir_indices", dataset_name)
        build_index(
            model, collection, eval_config,
            index_path=index_dir,
            batch_size=batch_size,
            doc_maxlen=config.beir_doc_maxlen,
        )

        retriever = ColBERTRetriever(model, index_dir, eval_config)

        query_list = queries.items()
        qid_list = [qid for qid, _ in query_list]
        query_texts = [text for _, text in query_list]

        raw_results = retriever.retrieve(
            query_texts, top_k=1000, query_maxlen=query_maxlen,
        )

        ranking = {
            qid_list[q_idx]: raw_results[q_idx]
            for q_idx in raw_results
        }

        dataset_metrics = evaluate_ranking(ranking, qrels)
        results[dataset_name] = dataset_metrics

        logger.info(f"Results for {dataset_name}:")
        for metric, value in dataset_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Compute averages
    if results:
        avg_ndcg = sum(r["nDCG@10"] for r in results.values()) / len(results)
        logger.info(f"\nAverage nDCG@10 across {len(results)} BEIR datasets: {avg_ndcg:.4f}")

    return results
