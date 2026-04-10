"""ChEMBL data source adapter."""

from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSource

logger = logging.getLogger(__name__)

# ChEMBL target type → our family label
_FAMILY_MAP = {
    "kinase": "SINGLE PROTEIN",
    "gpcr": "GPCR",
    "nuclear_receptor": "NUCLEAR RECEPTOR",
}

# ChEMBL keyword search terms per family
# Multiple keywords capture targets whose preferred name doesn't contain the
# obvious family term (e.g. EGFR = "Epidermal growth factor receptor erbB1").
_FAMILY_KEYWORDS = {
    "kinase": [
        "kinase",
        "tyrosine kinase",
        "growth factor receptor",
        "receptor tyrosine",
        "protein kinase",
        "JAK",
        "SRC family",
    ],
    "gpcr": ["receptor", "GPCR", "G protein-coupled"],
    "nuclear_receptor": ["nuclear receptor", "NR", "steroid receptor"],
}


def _extract_gene_symbol(target: dict) -> str:
    """Extract the official HGNC gene symbol from ChEMBL target_components.

    ChEMBL stores GENE_SYMBOL in target_components[0].target_component_synonyms.
    Falls back to the first word of pref_name if not found.
    """
    for component in target.get("target_components") or []:
        for syn in component.get("target_component_synonyms") or []:
            if syn.get("syn_type") == "GENE_SYMBOL":
                return syn["component_synonym"]
    # Fallback: first word of preferred name
    return (target.get("pref_name") or "").split(" ")[0]


class ChEMBLSource(DataSource):
    def __init__(self) -> None:
        from chembl_webresource_client.new_client import new_client

        self._target_api = new_client.target
        self._activity_api = new_client.activity
        self._molecule_api = new_client.molecule

    def fetch_targets(self, families: list[str]) -> list[dict]:
        targets: list[dict] = []
        seen: set[str] = set()

        for family in families:
            keywords = _FAMILY_KEYWORDS.get(family, [family])
            for kw in keywords:
                try:
                    results = self._search_targets(kw, family)
                    for t in results:
                        if t["id"] not in seen:
                            seen.add(t["id"])
                            targets.append(t)
                except Exception as exc:
                    logger.warning("ChEMBL target search failed for '%s': %s", kw, exc)

        logger.info("ChEMBL: fetched %d unique targets", len(targets))
        return targets

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _search_targets(self, keyword: str, family: str) -> list[dict]:
        raw = list(
            self._target_api.search(keyword).filter(
                target_type="SINGLE PROTEIN",
                organism="Homo sapiens",
            )[:500]
        )
        targets = []
        for t in raw:
            targets.append({
                "id": t["target_chembl_id"],
                "source": "chembl",
                "gene": _extract_gene_symbol(t),
                "protein_name": t.get("pref_name", ""),
                "organism": t.get("organism", "Homo sapiens"),
                "function": "",
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
        try:
            return self._fetch_activities_inner(target_id, activity_types, max_compounds)
        except Exception as exc:
            logger.warning("ChEMBL activities failed for %s: %s", target_id, exc)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_activities_inner(
        self,
        target_id: str,
        activity_types: list[str],
        max_compounds: int,
    ) -> list[dict]:
        raw = list(
            self._activity_api.filter(
                target_chembl_id=target_id,
                standard_type__in=activity_types,
                standard_relation="=",
            )
            .only([
                "molecule_chembl_id",
                "canonical_smiles",
                "molecule_pref_name",
                "standard_type",
                "standard_value",
                "standard_units",
                "pchembl_value",
                "assay_chembl_id",
                "assay_type",
                "assay_description",
                "document_chembl_id",
                "document_year",
                "target_pref_name",
                "target_organism",
                "data_validity_comment",
            ])[:max_compounds]
        )

        records = []
        for a in raw:
            if not a.get("standard_value") or not a.get("canonical_smiles"):
                continue
            try:
                value = float(a["standard_value"])
            except (TypeError, ValueError):
                continue

            # Normalize units to nM for consistent comparison
            units = a.get("standard_units", "nM")
            if units == "uM":
                value = value * 1000
                units = "nM"
            elif units == "mM":
                value = value * 1e6
                units = "nM"
            elif units == "pM":
                value = value / 1000
                units = "nM"

            # pChEMBL is -log10(value_in_M), higher = more potent
            pchembl = None
            if a.get("pchembl_value"):
                try:
                    pchembl = float(a["pchembl_value"])
                except (TypeError, ValueError):
                    pass

            records.append({
                "id": a["molecule_chembl_id"],
                "source": "chembl",
                "name": a.get("molecule_pref_name") or a["molecule_chembl_id"],
                "smiles": a["canonical_smiles"],
                "target_id": target_id,
                "target_name": a.get("target_pref_name", target_id),
                "activity_type": a["standard_type"],
                "activity_value": value,
                "activity_units": units,
                "pchembl_value": pchembl,
                "assay_chembl_id": a.get("assay_chembl_id", ""),
                "assay_type": a.get("assay_type", ""),  # B=Binding, F=Functional, A=ADME
                "assay_description": a.get("assay_description", ""),
                "document_chembl_id": a.get("document_chembl_id", ""),
                "document_year": a.get("document_year"),
                "target_organism": a.get("target_organism", "Homo sapiens"),
                "data_validity_comment": a.get("data_validity_comment", ""),
                "target_family": "",
            })
        return records
