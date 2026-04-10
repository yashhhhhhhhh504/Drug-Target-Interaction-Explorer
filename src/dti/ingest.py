"""Ingest orchestrator: fetch from enabled sources → write JSONLines artifacts."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlflow
import requests

from .config import Config
from .sources.base import DataSource

logger = logging.getLogger(__name__)

_UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


def _enrich_chembl_targets_with_uniprot(targets: list[dict]) -> list[dict]:
    """Cross-reference ChEMBL targets with UniProt by gene name.

    Populates `function`, `go_terms`, `diseases`, and `uniprot_id` for targets
    sourced from ChEMBL (which have gene symbols but no biological annotations).
    Skips targets that already have annotations or are missing a gene symbol.
    """
    chembl_targets = [t for t in targets if t.get("source") == "chembl" and t.get("gene")]
    logger.info("Enriching %d ChEMBL targets with UniProt annotations…", len(chembl_targets))

    enriched = 0
    for t in chembl_targets:
        gene = t["gene"].strip()
        if t.get("function") or t.get("go_terms") or t.get("diseases"):
            continue

        try:
            params = {
                "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "size": 1,
                "fields": "accession,cc_function,go,cc_disease",
            }
            resp = requests.get(f"{_UNIPROT_BASE}/search", params=params, timeout=20)
            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                # Try broader gene search
                params["query"] = f"gene:{gene} AND organism_id:9606 AND reviewed:true"
                resp = requests.get(f"{_UNIPROT_BASE}/search", params=params, timeout=20)
                resp.raise_for_status()
                results = resp.json().get("results", [])

            if not results:
                logger.debug("No UniProt entry found for gene %s", gene)
                continue

            entry = results[0]
            t["uniprot_id"] = entry.get("primaryAccession", "")

            # Function annotation
            for comment in entry.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        t["function"] = texts[0].get("value", "")
                    break

            # GO terms (keep all three categories: P=process, F=function, C=component)
            go_terms = []
            for ref in entry.get("uniProtKBCrossReferences", []):
                if ref.get("database") == "GO":
                    for prop in ref.get("properties", []):
                        if prop.get("key") == "GoTerm":
                            go_terms.append(prop["value"])
            t["go_terms"] = go_terms[:20]

            # Disease associations
            diseases = []
            for comment in entry.get("comments", []):
                if comment.get("commentType") == "DISEASE":
                    d = comment.get("disease", {})
                    # UniProt uses "diseaseId" for the disease name string
                    disease_name = d.get("diseaseId") or d.get("diseaseName", "")
                    if isinstance(disease_name, dict):
                        disease_name = disease_name.get("value", "")
                    if disease_name and disease_name.strip():
                        diseases.append(disease_name.strip())
            t["diseases"] = diseases

            enriched += 1
            logger.debug("Enriched %s (UniProt: %s, GO terms: %d, diseases: %d)",
                         gene, t["uniprot_id"], len(t["go_terms"]), len(t["diseases"]))

            # Polite rate limit — UniProt asks for max 3 req/s from automated clients
            time.sleep(0.4)

        except Exception as exc:
            logger.warning("UniProt enrichment failed for gene %s: %s", gene, exc)

    logger.info("UniProt enrichment complete: %d/%d ChEMBL targets annotated",
                enriched, len(chembl_targets))
    return targets


def _build_sources(cfg: Config) -> dict[str, DataSource]:
    sources: dict[str, DataSource] = {}

    if cfg.sources.chembl:
        from .sources.chembl import ChEMBLSource
        sources["chembl"] = ChEMBLSource()

    if cfg.sources.uniprot:
        from .sources.uniprot import UniProtSource
        sources["uniprot"] = UniProtSource()

    if cfg.sources.pubchem:
        from .sources.pubchem import PubChemSource
        sources["pubchem"] = PubChemSource()

    if cfg.sources.bindingdb:
        from .sources.bindingdb import BindingDBSource
        sources["bindingdb"] = BindingDBSource()

    if cfg.sources.drugbank:
        from .sources.drugbank import DrugBankSource
        sources["drugbank"] = DrugBankSource()

    return sources


def run(cfg: Config, override_sources: list[str] | None = None) -> None:
    """Fetch data from all enabled sources and write JSONLines to data/raw/.

    Args:
        cfg: loaded Config object
        override_sources: if provided, use these sources instead of config
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="ingest"):
        mlflow.log_param("target_families", cfg.ingest.target_families)
        mlflow.log_param("activity_types", cfg.ingest.activity_types)
        mlflow.log_param("max_compounds_per_target", cfg.ingest.max_compounds_per_target)

        sources = _build_sources(cfg)
        if override_sources:
            sources = {k: v for k, v in sources.items() if k in override_sources}

        mlflow.log_param("active_sources", list(sources.keys()))
        logger.info("Active sources: %s", list(sources.keys()))

        raw_dir = cfg.data.raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        failed_path = raw_dir / "failed_targets.txt"

        # --- Fetch targets ---
        all_targets: list[dict] = []
        seen_target_ids: set[str] = set()

        for source_name, source in sources.items():
            logger.info("[%s] Fetching targets for families: %s", source_name, cfg.ingest.target_families)
            try:
                targets = source.fetch_targets(cfg.ingest.target_families)
                for t in targets:
                    if t["id"] not in seen_target_ids:
                        seen_target_ids.add(t["id"])
                        all_targets.append(t)
            except Exception as exc:
                logger.error("[%s] fetch_targets failed: %s", source_name, exc)

        # Cross-enrich ChEMBL targets with UniProt biological annotations
        # (function, GO terms, disease associations, UniProt accession)
        if cfg.sources.chembl and cfg.sources.uniprot:
            all_targets = _enrich_chembl_targets_with_uniprot(all_targets)

        targets_path = raw_dir / "targets.jsonl"
        _write_jsonl(targets_path, all_targets)
        mlflow.log_metric("targets_fetched", len(all_targets))
        logger.info("Wrote %d targets to %s", len(all_targets), targets_path)

        # --- Build cross-source ID mapping ---
        # Maps non-UniProt target IDs (e.g. CHEMBL203) to UniProt accessions
        # so that BindingDB (which requires UniProt IDs) can fetch activities
        # for targets originally discovered by ChEMBL.
        id_to_uniprot: dict[str, str] = {}
        for t in all_targets:
            uid = t.get("uniprot_id", "")
            if uid and t["id"] != uid:
                id_to_uniprot[t["id"]] = uid
            # UniProt-sourced targets already use accession as their ID
            if t.get("source") == "uniprot":
                id_to_uniprot[t["id"]] = t["id"]

        if id_to_uniprot:
            logger.info(
                "Cross-source ID map: %d targets have UniProt accessions for BindingDB",
                len(id_to_uniprot),
            )
            mlflow.log_metric("targets_with_uniprot_mapping", len(id_to_uniprot))

        # --- Fetch activities per target ---
        all_compounds: list[dict] = []
        failed_targets: list[str] = []

        for target in all_targets:
            target_id = target["id"]
            target_family = target["target_family"]
            compound_count_before = len(all_compounds)

            for source_name, source in sources.items():
                # For BindingDB, use the mapped UniProt accession instead of
                # the native target ID (e.g. CHEMBL203 → P00533)
                fetch_id = target_id
                if source_name == "bindingdb":
                    fetch_id = id_to_uniprot.get(target_id, target_id)

                try:
                    activities = source.fetch_activities(
                        fetch_id,
                        cfg.ingest.activity_types,
                        cfg.ingest.max_compounds_per_target,
                    )
                    for a in activities:
                        a["target_family"] = target_family
                        # Normalize target_id back to the canonical ID so
                        # downstream chunking/graph can join on a single key
                        a["target_id"] = target_id
                    all_compounds.extend(activities)
                    if activities:
                        logger.debug(
                            "[%s] %s (fetched as %s): %d activities",
                            source_name, target_id, fetch_id, len(activities),
                        )
                except Exception as exc:
                    logger.warning(
                        "[%s] fetch_activities failed for %s: %s",
                        source_name, target_id, exc,
                    )
                    failed_targets.append(f"{source_name}:{target_id}")

            if len(all_compounds) == compound_count_before:
                logger.debug("No activities found for target %s", target_id)

        compounds_path = raw_dir / "compounds.jsonl"
        _write_jsonl(compounds_path, all_compounds)
        mlflow.log_metric("compounds_fetched", len(all_compounds))
        mlflow.log_metric("failed_targets", len(failed_targets))

        if failed_targets:
            failed_path.write_text("\n".join(failed_targets) + "\n")
            mlflow.log_artifact(str(failed_path))
            logger.warning("%d target fetches failed — see %s", len(failed_targets), failed_path)

        mlflow.log_artifact(str(targets_path))
        mlflow.log_artifact(str(compounds_path))

        logger.info(
            "Ingest complete: %d targets, %d compound-activity records",
            len(all_targets), len(all_compounds),
        )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
