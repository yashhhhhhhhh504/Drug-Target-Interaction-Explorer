"""Load and validate config.yaml with environment variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EmbeddingConfig:
    provider: str
    biolord_model: str
    openai_model: str
    ollama_model: str
    ollama_base_url: str


@dataclass
class LLMConfig:
    provider: str
    anthropic_model: str
    ollama_model: str
    ollama_base_url: str
    local_model: str = "microsoft/Phi-3-mini-4k-instruct"
    local_device: str = "auto"     # "auto" | "mps" | "cpu" | "cuda"
    local_max_new_tokens: int = 1000


@dataclass
class RetrievalConfig:
    initial_k: int
    rerank_top_k: int
    reranker_model: str


@dataclass
class DataConfig:
    raw_dir: Path
    docs_dir: Path
    graph_dir: Path
    db_dir: Path


@dataclass
class IngestConfig:
    target_families: list[str]
    activity_types: list[str]
    max_compounds_per_target: int


@dataclass
class SourcesConfig:
    chembl: bool
    uniprot: bool
    pubchem: bool
    bindingdb: bool
    drugbank: bool


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass
class Config:
    embedding: EmbeddingConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    data: DataConfig
    ingest: IngestConfig
    sources: SourcesConfig
    mlflow: MLflowConfig

    @property
    def enabled_sources(self) -> list[str]:
        return [
            name
            for name in ["chembl", "uniprot", "pubchem", "bindingdb", "drugbank"]
            if getattr(self.sources, name)
        ]


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load(path: Optional[Path] = None) -> Config:
    """Load config from YAML file, applying environment variable overrides."""
    config_path = path or _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Environment variable overrides
    emb = raw["embedding"]
    emb["provider"] = os.environ.get("EMBEDDING_PROVIDER", emb["provider"])

    llm = raw["llm"]
    llm["provider"] = os.environ.get("LLM_PROVIDER", llm["provider"])

    data = raw["data"]

    return Config(
        embedding=EmbeddingConfig(
            provider=emb["provider"],
            biolord_model=emb["biolord_model"],
            openai_model=emb["openai_model"],
            ollama_model=emb.get("ollama_model", "nomic-embed-text"),
            ollama_base_url=emb.get("ollama_base_url", llm["ollama_base_url"]),
        ),
        llm=LLMConfig(
            provider=llm["provider"],
            anthropic_model=llm["anthropic_model"],
            ollama_model=llm["ollama_model"],
            ollama_base_url=llm["ollama_base_url"],
            local_model=llm.get("local_model", "microsoft/Phi-3-mini-4k-instruct"),
            local_device=llm.get("local_device", "auto"),
            local_max_new_tokens=llm.get("local_max_new_tokens", 1000),
        ),
        retrieval=RetrievalConfig(
            initial_k=raw["retrieval"]["initial_k"],
            rerank_top_k=raw["retrieval"]["rerank_top_k"],
            reranker_model=raw["retrieval"]["reranker_model"],
        ),
        data=DataConfig(
            raw_dir=Path(data["raw_dir"]),
            docs_dir=Path(data["docs_dir"]),
            graph_dir=Path(data["graph_dir"]),
            db_dir=Path(data["db_dir"]),
        ),
        ingest=IngestConfig(
            target_families=raw["ingest"]["target_families"],
            activity_types=raw["ingest"]["activity_types"],
            max_compounds_per_target=raw["ingest"]["max_compounds_per_target"],
        ),
        sources=SourcesConfig(
            chembl=raw["sources"]["chembl"],
            uniprot=raw["sources"]["uniprot"],
            pubchem=raw["sources"]["pubchem"],
            bindingdb=raw["sources"]["bindingdb"],
            drugbank=raw["sources"]["drugbank"],
        ),
        mlflow=MLflowConfig(
            tracking_uri=raw["mlflow"]["tracking_uri"],
            experiment_name=raw["mlflow"]["experiment_name"],
        ),
    )
