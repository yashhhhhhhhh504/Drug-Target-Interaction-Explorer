"""Public query API — retrieves, re-ranks, and generates answers with MLflow tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow

from .config import Config
from .retrieve import RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query: str
    answer: str
    sources: list[RetrievedDoc]


def run(query: str, cfg: Config) -> QueryResult:
    """Full query pipeline: retrieve → rerank → generate → log to MLflow.

    Args:
        query: natural language question
        cfg: loaded Config

    Returns:
        QueryResult with answer and supporting sources
    """
    from .embed import build_embedder
    from .generate import generate
    from .graph import load_graph
    from .retrieve import retrieve
    from .store import get_collection

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="query"):
        mlflow.log_param("query", query)
        mlflow.log_param("llm_provider", cfg.llm.provider)
        mlflow.log_param("embedding_provider", cfg.embedding.provider)
        mlflow.log_param("rerank_top_k", cfg.retrieval.rerank_top_k)

        embedder = build_embedder(cfg)
        collection = get_collection(cfg)
        graph = load_graph(cfg)

        mlflow.log_param("graph_available", graph is not None)

        if collection.count() == 0:
            logger.warning(
                "ChromaDB collection is empty. Run `dti ingest` then `dti embed` first."
            )

        docs = retrieve(query, cfg, collection, graph=graph, embedder=embedder)

        mlflow.log_metric("candidates_retrieved", len(docs))
        if docs:
            mlflow.log_metric("top_rerank_score", docs[0].rerank_score)
            mlflow.log_metric("min_rerank_score", docs[-1].rerank_score)

        for i, doc in enumerate(docs, start=1):
            mlflow.log_metric(f"source_{i}_score", doc.rerank_score)
            mlflow.log_param(f"source_{i}_type", doc.metadata.get("doc_type", ""))
            mlflow.log_param(f"source_{i}_id", doc.doc_id)

        answer = generate(query, docs, cfg)

        mlflow.log_param("answer_length", len(answer))

        return QueryResult(query=query, answer=answer, sources=docs)
