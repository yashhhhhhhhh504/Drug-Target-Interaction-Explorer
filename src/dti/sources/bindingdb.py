"""BindingDB REST API adapter."""

from __future__ import annotations

import logging
import re

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.bindingdb.org/axis2/services/BDBService"

# UniProt accession: 6-10 alphanumeric, starts with [A-N,R-Z][0-9] or [OPQ][0-9]
_UNIPROT_RE = re.compile(r"^[A-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z0-9]{0,4})?$")


class BindingDBSource(DataSource):
    def fetch_targets(self, families: list[str]) -> list[dict]:
        # BindingDB does not expose a target-family search endpoint.
        # Targets are seeded from ChEMBL/UniProt; BindingDB enriches activities.
        return []

    def fetch_activities(
        self,
        target_id: str,
        activity_types: list[str],
        max_compounds: int,
    ) -> list[dict]:
        # target_id must be a UniProt accession (e.g. P00533, Q9Y6K9)
        if not _UNIPROT_RE.match(target_id):
            return []
        try:
            return self._fetch_by_uniprot(target_id, max_compounds)
        except Exception as exc:
            logger.warning("BindingDB activities failed for %s: %s", target_id, exc)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_by_uniprot(self, uniprot_id: str, max_compounds: int) -> list[dict]:
        params = {
            "uniprot_id": uniprot_id,
            "response_format": "json",
            "cutoff": max_compounds,
        }
        url = f"{_BASE_URL}/getLigandsByUniprot"
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

        affinities = data.get("affinities", [])
        records = []
        for entry in affinities[:max_compounds]:
            ki = entry.get("ki")
            ic50 = entry.get("ic50")
            kd = entry.get("kd")

            if ic50:
                act_type, act_val = "IC50", ic50
            elif ki:
                act_type, act_val = "Ki", ki
            elif kd:
                act_type, act_val = "Kd", kd
            else:
                continue

            try:
                act_val = float(act_val)
            except (TypeError, ValueError):
                continue

            records.append({
                "id": f"BDB_{entry.get('monomerID', '')}",
                "source": "bindingdb",
                "name": entry.get("ligandName", ""),
                "smiles": entry.get("smile", ""),
                "target_id": uniprot_id,
                "target_name": entry.get("targetName", uniprot_id),
                "activity_type": act_type,
                "activity_value": act_val,
                "activity_units": "nM",
                "pchembl_value": None,
                "assay_chembl_id": "",
                "assay_type": "",
                "assay_description": entry.get("assayDescription", ""),
                "document_chembl_id": "",
                "document_year": None,
                "target_organism": "Homo sapiens",
                "data_validity_comment": "",
                "target_family": "",
            })
        return records
