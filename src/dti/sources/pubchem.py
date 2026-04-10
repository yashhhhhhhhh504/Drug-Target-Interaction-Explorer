"""PubChem BioAssay REST API adapter."""

from __future__ import annotations

import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# PubChem protein target search via gene name
_FAMILY_GENES = {
    "kinase": ["EGFR", "BRAF", "CDK4", "CDK6", "ABL1", "SRC", "ERBB2", "MET"],
    "gpcr": ["ADRB2", "DRD2", "HTR2A", "CHRM1", "OPRD1", "CXCR4"],
    "nuclear_receptor": ["ESR1", "AR", "PPARG", "NR3C1", "RARA"],
}


class PubChemSource(DataSource):
    def fetch_targets(self, families: list[str]) -> list[dict]:
        targets: list[dict] = []
        seen: set[str] = set()
        for family in families:
            genes = _FAMILY_GENES.get(family, [])
            for gene in genes:
                tid = f"pubchem_gene_{gene}"
                if tid not in seen:
                    seen.add(tid)
                    targets.append({
                        "id": tid,
                        "source": "pubchem",
                        "gene": gene,
                        "protein_name": gene,
                        "organism": "Homo sapiens",
                        "function": "",
                        "go_terms": [],
                        "diseases": [],
                        "target_family": family,
                    })
        logger.info("PubChem: yielding %d gene targets for activity lookup", len(targets))
        return targets

    def fetch_activities(
        self,
        target_id: str,
        activity_types: list[str],
        max_compounds: int,
    ) -> list[dict]:
        if not target_id.startswith("pubchem_gene_"):
            return []
        gene = target_id.replace("pubchem_gene_", "")
        try:
            return self._fetch_bioassay_data(gene, max_compounds)
        except Exception as exc:
            logger.warning("PubChem activities failed for %s: %s", gene, exc)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_bioassay_data(self, gene: str, max_compounds: int) -> list[dict]:
        # Step 1: find bioassays targeting this gene
        url = f"{_BASE_URL}/assay/target/genesymbol/{gene}/aids/JSON"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        aids = resp.json().get("IdentifierList", {}).get("AID", [])[:5]

        records = []
        for aid in aids:
            try:
                compounds = self._fetch_assay_compounds(aid, gene, max_compounds - len(records))
                records.extend(compounds)
                if len(records) >= max_compounds:
                    break
            except Exception as exc:
                logger.debug("PubChem assay %s fetch error: %s", aid, exc)

        return records[:max_compounds]

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _fetch_assay_compounds(self, aid: int, gene: str, limit: int) -> list[dict]:
        url = f"{_BASE_URL}/assay/aid/{aid}/concise/JSON"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

        table = data.get("PC_AssaySubmit", {}).get("assay", {})
        rows = data.get("PC_AssaySubmit", {}).get("data", [])

        records = []
        for row in rows[:limit]:
            cid = row.get("sid", "")
            outcome = row.get("outcome", 0)
            if outcome not in (2, 3):  # 2=active, 3=inactive
                continue
            records.append({
                "id": f"CID{cid}",
                "source": "pubchem",
                "name": f"CID{cid}",
                "smiles": "",
                "target_id": f"pubchem_gene_{gene}",
                "target_name": gene,
                "activity_type": "Active" if outcome == 2 else "Inactive",
                "activity_value": float(outcome == 2),
                "activity_units": "binary",
                "assay_description": f"PubChem AID {aid}",
                "target_family": "",
            })
        return records
