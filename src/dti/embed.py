"""Embedding wrapper — BioLORD (local) or OpenAI (remote), selected via config."""

from __future__ import annotations

import logging
from typing import Protocol

import mlflow

from .config import Config

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BioLORDEmbedder:
    def __init__(self, model_name: str) -> None:
        import os
        from sentence_transformers import SentenceTransformer

        logger.info("Loading BioLORD model: %s", model_name)
        # Suppress verbose model-load report printed to stdout
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._model = SentenceTransformer(model_name, trust_remote_code=False)
        self._model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]


class OpenAIEmbedder:
    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model_name = model_name
        logger.info("Using OpenAI embedding model: %s", model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        # OpenAI API accepts up to 2048 inputs per call; batch in groups of 512
        results: list[list[float]] = []
        for chunk in _chunk_list(texts, 512):
            resp = self._client.embeddings.create(model=self._model_name, input=chunk)
            results.extend([item.embedding for item in resp.data])
        return results


class OllamaEmbedder:
    """Embed via Ollama's local API — no API key, runs fully offline."""

    def __init__(self, model_name: str, base_url: str) -> None:
        import requests

        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._requests = requests
        logger.info("Using Ollama embedding model: %s at %s", model_name, base_url)

    def encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        # Ollama processes one text at a time for embeddings
        for text in texts:
            resp = self._requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model_name, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results


def build_embedder(cfg: Config) -> Embedder:
    if cfg.embedding.provider == "openai":
        return OpenAIEmbedder(cfg.embedding.openai_model)
    if cfg.embedding.provider == "ollama":
        return OllamaEmbedder(cfg.embedding.ollama_model, cfg.embedding.ollama_base_url)
    return BioLORDEmbedder(cfg.embedding.biolord_model)


def run(cfg: Config) -> None:
    """Embed all documents from JSONLines and store in ChromaDB.

    Idempotent: skips document IDs already present in the collection.
    """
    from .chunk import build_all
    from .ingest import read_jsonl
    from .store import get_collection, upsert_documents

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="embed"):
        provider = cfg.embedding.provider
        model_name = {
            "openai": cfg.embedding.openai_model,
            "ollama": cfg.embedding.ollama_model,
        }.get(provider, cfg.embedding.biolord_model)
        mlflow.log_param("embedding_provider", provider)
        mlflow.log_param("embedding_model", model_name)

        targets = read_jsonl(cfg.data.raw_dir / "targets.jsonl")
        compounds = read_jsonl(cfg.data.raw_dir / "compounds.jsonl")

        logger.info("Building documents from %d targets, %d compounds", len(targets), len(compounds))
        docs = build_all(targets, compounds)
        mlflow.log_metric("documents_total", len(docs))
        logger.info("Built %d documents", len(docs))

        collection = get_collection(cfg)

        # Find which doc IDs are already stored
        existing_ids: set[str] = set()
        try:
            existing = collection.get(ids=[d.doc_id for d in docs])
            existing_ids = set(existing["ids"])
        except Exception:
            pass

        new_docs = [d for d in docs if d.doc_id not in existing_ids]
        mlflow.log_metric("documents_skipped", len(docs) - len(new_docs))
        mlflow.log_metric("documents_to_embed", len(new_docs))

        if not new_docs:
            logger.info("All documents already embedded. Nothing to do.")
            return

        logger.info("Embedding %d new documents (skipping %d already stored)", len(new_docs), len(existing_ids))
        embedder = build_embedder(cfg)

        texts = [d.text for d in new_docs]
        try:
            embeddings = embedder.encode(texts)
        except Exception as exc:
            logger.error("Embedding failed, retrying once: %s", exc)
            embeddings = embedder.encode(texts)

        upsert_documents(collection, new_docs, embeddings)
        mlflow.log_metric("documents_embedded", len(new_docs))
        logger.info("Embedded and stored %d documents", len(new_docs))


def _chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]
