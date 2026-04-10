"""Unit tests for chunk.py — no API calls, uses fixture data only."""

import json
from pathlib import Path

import pytest

from dti.chunk import (
    Document,
    build_all,
    build_compound_activity_docs,
    build_selectivity_docs,
    build_target_profiles,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[dict]:
    records = []
    with open(FIXTURES / name) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture
def targets():
    return load_fixture("targets.jsonl")


@pytest.fixture
def compounds():
    return load_fixture("compounds.jsonl")


# ── target profiles ────────────────────────────────────────────────────────

class TestBuildTargetProfiles:
    def test_returns_one_doc_per_target(self, targets, compounds):
        docs = build_target_profiles(targets, compounds)
        assert len(docs) == len(targets)

    def test_doc_ids_are_unique(self, targets, compounds):
        docs = build_target_profiles(targets, compounds)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_gene_name_in_text(self, targets, compounds):
        docs = build_target_profiles(targets, compounds)
        egfr_doc = next(d for d in docs if "EGFR" in d.text)
        assert "EGFR" in egfr_doc.text

    def test_activity_summary_included(self, targets, compounds):
        docs = build_target_profiles(targets, compounds)
        egfr_doc = next(d for d in docs if "EGFR" in d.text)
        assert "Known ligands" in egfr_doc.text

    def test_metadata_doc_type(self, targets, compounds):
        docs = build_target_profiles(targets, compounds)
        for doc in docs:
            assert doc.metadata["doc_type"] == "target_profile"

    def test_metadata_target_family(self, targets, compounds):
        docs = build_target_profiles(targets, compounds)
        families = {d.metadata["target_family"] for d in docs}
        assert "kinase" in families
        assert "gpcr" in families

    def test_no_activities_handled_gracefully(self, targets):
        docs = build_target_profiles(targets, [])
        assert len(docs) == len(targets)
        for doc in docs:
            assert "None in dataset" in doc.text


# ── compound activity docs ─────────────────────────────────────────────────

class TestBuildCompoundActivityDocs:
    def test_returns_doc_per_unique_compound(self, compounds, targets):
        docs = build_compound_activity_docs(compounds, targets)
        unique_compounds = {c["id"] for c in compounds}
        assert len(docs) == len(unique_compounds)

    def test_smiles_in_text(self, compounds, targets):
        docs = build_compound_activity_docs(compounds, targets)
        gefitinib = next(d for d in docs if "Gefitinib" in d.text)
        assert "COc1cc2ncnc" in gefitinib.text

    def test_activity_values_in_text(self, compounds, targets):
        docs = build_compound_activity_docs(compounds, targets)
        gefitinib = next(d for d in docs if "Gefitinib" in d.text)
        assert "0.033" in gefitinib.text or "IC50" in gefitinib.text

    def test_metadata_doc_type(self, compounds, targets):
        docs = build_compound_activity_docs(compounds, targets)
        for doc in docs:
            assert doc.metadata["doc_type"] == "compound_activity"

    def test_metadata_compound_id(self, compounds, targets):
        docs = build_compound_activity_docs(compounds, targets)
        compound_ids = {d.metadata["compound_id"] for d in docs}
        assert "CHEMBL553" in compound_ids

    def test_chunking_large_compound(self, targets):
        # Compound with 16 targets should be split into 2 chunks
        large_compound = [
            {
                "id": "CHEMBL_BIG",
                "source": "chembl",
                "name": "BigMolecule",
                "smiles": "C",
                "target_id": f"T{i}",
                "target_name": f"Target{i}",
                "activity_type": "IC50",
                "activity_value": float(i),
                "activity_units": "nM",
                "assay_description": "",
                "target_family": "kinase",
            }
            for i in range(16)
        ]
        docs = build_compound_activity_docs(large_compound, [])
        assert len(docs) == 2  # 10 + 6

    def test_smiles_repeated_in_each_chunk(self, targets):
        large_compound = [
            {
                "id": "CHEMBL_BIG",
                "source": "chembl",
                "name": "BigMolecule",
                "smiles": "UNIQUE_SMILES",
                "target_id": f"T{i}",
                "target_name": f"Target{i}",
                "activity_type": "IC50",
                "activity_value": 1.0,
                "activity_units": "nM",
                "assay_description": "",
                "target_family": "kinase",
            }
            for i in range(16)
        ]
        docs = build_compound_activity_docs(large_compound, [])
        for doc in docs:
            assert "UNIQUE_SMILES" in doc.text


# ── selectivity docs ───────────────────────────────────────────────────────

class TestBuildSelectivityDocs:
    def test_returns_docs_for_same_family_pairs(self, compounds, targets):
        docs = build_selectivity_docs(compounds, targets)
        # EGFR vs ERBB2 are both kinase — should produce a selectivity doc
        pair_texts = [d.text for d in docs]
        assert any("EGFR" in t and "ERBB2" in t for t in pair_texts)

    def test_no_cross_family_pairs(self, compounds, targets):
        docs = build_selectivity_docs(compounds, targets)
        for doc in docs:
            # Each doc should reference a single family
            assert doc.metadata["doc_type"] == "selectivity_comparison"

    def test_ratio_calculation(self, compounds, targets):
        docs = build_selectivity_docs(compounds, targets)
        egfr_erbb2 = next(
            (d for d in docs if "EGFR" in d.text and "ERBB2" in d.text), None
        )
        assert egfr_erbb2 is not None
        # Gefitinib: EGFR 0.033 uM, ERBB2 3.7 uM → ratio ~112x
        assert "112x" in egfr_erbb2.text or "Gefitinib" in egfr_erbb2.text

    def test_metadata_doc_type(self, compounds, targets):
        docs = build_selectivity_docs(compounds, targets)
        for doc in docs:
            assert doc.metadata["doc_type"] == "selectivity_comparison"


# ── build_all ──────────────────────────────────────────────────────────────

class TestBuildAll:
    def test_combines_all_three_types(self, targets, compounds):
        docs = build_all(targets, compounds)
        types = {d.metadata["doc_type"] for d in docs}
        expected = {
            "target_profile", "compound_activity", "selectivity_comparison",
            "compound_selectivity", "target_selectivity_index", "evidence_depth",
        }
        assert types == expected

    def test_all_doc_ids_unique(self, targets, compounds):
        docs = build_all(targets, compounds)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))
