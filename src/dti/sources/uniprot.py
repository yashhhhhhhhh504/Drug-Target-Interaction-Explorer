"""UniProt REST API adapter."""

from __future__ import annotations

import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://rest.uniprot.org/uniprotkb"

# UniProt keyword IDs for target families
_FAMILY_KEYWORD_IDS = {
    "kinase": "KW-0418",
    "gpcr": "KW-0297",
    "nuclear_receptor": "KW-0539",
}


class UniProtSource(DataSource):
    def fetch_targets(self, families: list[str]) -> list[dict]:
        targets: list[dict] = []
        seen: set[str] = set()

        for family in families:
            kw_id = _FAMILY_KEYWORD_IDS.get(family)
            if not kw_id:
                logger.warning("UniProt: no keyword ID for family '%s', skipping", family)
                continue
            try:
                results = self._search_targets(kw_id, family)
                for t in results:
                    if t["id"] not in seen:
                        seen.add(t["id"])
                        targets.append(t)
            except Exception as exc:
                logger.warning("UniProt target search failed for '%s': %s", family, exc)

        logger.info("UniProt: fetched %d unique targets", len(targets))
        return targets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _search_targets(self, keyword_id: str, family: str) -> list[dict]:
        params = {
            "query": f"keyword:{keyword_id} AND organism_id:9606 AND reviewed:true",
            "format": "json",
            "size": 200,
            "fields": "accession,gene_names,protein_name,cc_function,go,cc_disease",
        }
        resp = requests.get(f"{_BASE_URL}/search", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        targets = []
        for entry in data.get("results", []):
            accession = entry["primaryAccession"]
            gene = ""
            if entry.get("genes"):
                gene = entry["genes"][0].get("geneName", {}).get("value", "")

            protein_name = ""
            pn = entry.get("proteinDescription", {})
            if pn.get("recommendedName"):
                protein_name = pn["recommendedName"].get("fullName", {}).get("value", "")

            function = ""
            for comment in entry.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function = texts[0].get("value", "")
                    break

            go_terms = []
            for ref in entry.get("uniProtKBCrossReferences", []):
                if ref.get("database") == "GO":
                    for prop in ref.get("properties", []):
                        if prop.get("key") == "GoTerm":
                            go_terms.append(prop["value"])

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

            targets.append({
                "id": accession,
                "source": "uniprot",
                "gene": gene,
                "protein_name": protein_name,
                "organism": "Homo sapiens",
                "function": function,
                "go_terms": go_terms[:20],
                "diseases": diseases,
                "target_family": family,
            })
        return targets

    def fetch_activities(
        self,
        target_id: str,
        activity_types: list[str],
        max_compounds: int,
    ) -> list[dict]:
        # UniProt provides protein annotations, not bioactivity data.
        # Activity data comes from ChEMBL/BindingDB. Return empty here.
        return []
