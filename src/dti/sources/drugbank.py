"""DrugBank API adapter (requires DRUGBANK_API_KEY)."""

from __future__ import annotations

import logging
import os

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.drugbank.com/v1"


class DrugBankSource(DataSource):
    def __init__(self) -> None:
        self._api_key = os.environ.get("DRUGBANK_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "DRUGBANK_API_KEY environment variable is required for DrugBank source. "
                "Set it in your .env file or disable drugbank in config.yaml."
            )

    def _headers(self) -> dict:
        return {"Authorization": self._api_key, "Content-Type": "application/json"}

    def fetch_targets(self, families: list[str]) -> list[dict]:
        targets: list[dict] = []
        seen: set[str] = set()
        for family in families:
            try:
                results = self._search_targets(family)
                for t in results:
                    if t["id"] not in seen:
                        seen.add(t["id"])
                        targets.append(t)
            except Exception as exc:
                logger.warning("DrugBank target search failed for '%s': %s", family, exc)
        logger.info("DrugBank: fetched %d targets", len(targets))
        return targets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _search_targets(self, family: str) -> list[dict]:
        resp = requests.get(
            f"{_BASE_URL}/targets",
            headers=self._headers(),
            params={"q": family, "organism": "Human", "page": 1, "per_page": 100},
            timeout=30,
        )
        resp.raise_for_status()
        targets = []
        for entry in resp.json():
            targets.append({
                "id": f"DB_target_{entry.get('id', '')}",
                "source": "drugbank",
                "gene": entry.get("gene_name", ""),
                "protein_name": entry.get("name", ""),
                "organism": "Homo sapiens",
                "function": entry.get("general_function", ""),
                "go_terms": [],
                "diseases": [],
                "target_family": family,
            })
        return targets

    def fetch_activities(
        self,
        target_id: str,
        activity_types: list[str],
        max_compounds: int,
    ) -> list[dict]:
        if not target_id.startswith("DB_target_"):
            return []
        db_id = target_id.replace("DB_target_", "")
        try:
            return self._fetch_drugs_for_target(db_id, max_compounds)
        except Exception as exc:
            logger.warning("DrugBank activities failed for %s: %s", target_id, exc)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_drugs_for_target(self, db_id: str, max_compounds: int) -> list[dict]:
        resp = requests.get(
            f"{_BASE_URL}/targets/{db_id}/drugs",
            headers=self._headers(),
            params={"page": 1, "per_page": min(max_compounds, 100)},
            timeout=30,
        )
        resp.raise_for_status()
        records = []
        for drug in resp.json():
            records.append({
                "id": drug.get("drugbank_id", ""),
                "source": "drugbank",
                "name": drug.get("name", ""),
                "smiles": drug.get("smiles", ""),
                "target_id": f"DB_target_{db_id}",
                "target_name": "",
                "activity_type": "Interaction",
                "activity_value": 1.0,
                "activity_units": "binary",
                "assay_description": drug.get("mechanism_of_action", ""),
                "target_family": "",
            })
        return records[:max_compounds]
