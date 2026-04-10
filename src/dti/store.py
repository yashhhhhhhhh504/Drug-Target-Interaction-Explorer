"""ChromaDB read/write operations."""

from __future__ import annotations

import logging

import chromadb

from .chunk import Document
from .config import Config

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "drug_target_interactions"


def get_client(cfg: Config) -> chromadb.ClientAPI:
    cfg.data.db_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(cfg.data.db_dir))


def get_collection(cfg: Config) -> chromadb.Collection:
    client = get_client(cfg)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_ephemeral_collection() -> chromadb.Collection:
    """Return an in-memory collection for testing — not persisted to disk."""
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


_CHROMA_MAX_BATCH = 5000  # ChromaDB hard limit is 5461; stay safely below it


def upsert_documents(
    collection: chromadb.Collection,
    docs: list[Document],
    embeddings: list[list[float]],
) -> None:
    if not docs:
        return
    for start in range(0, len(docs), _CHROMA_MAX_BATCH):
        batch_docs = docs[start : start + _CHROMA_MAX_BATCH]
        batch_embs = embeddings[start : start + _CHROMA_MAX_BATCH]
        collection.upsert(
            ids=[d.doc_id for d in batch_docs],
            documents=[d.text for d in batch_docs],
            embeddings=batch_embs,
            metadatas=[d.metadata for d in batch_docs],
        )
        logger.info("Upserted batch %d–%d into ChromaDB", start, start + len(batch_docs))


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int,
    where: dict | None = None,
) -> list[dict]:
    """Query the collection and return a list of result dicts.

    Each dict has: id, text, metadata, distance.
    """
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
        output.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
            "distance": dist,
        })
    return output
