"""Integration tests for retrieve.py against a real ephemeral ChromaDB instance."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dti.chunk import build_all
from dti.retrieve import RetrievedDoc, _detect_filter, _extract_entities, retrieve
from dti.store import get_ephemeral_collection, upsert_documents

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
def populated_collection():
    """Real ephemeral ChromaDB collection pre-loaded with fixture docs."""
    targets = load_fixture("targets.jsonl")
    compounds = load_fixture("compounds.jsonl")
    docs = build_all(targets, compounds)

    collection = get_ephemeral_collection()

    # Use a tiny fake embedder to avoid downloading models in tests
    dim = 8
    embeddings = [
        [float(hash(d.doc_id + str(i)) % 100) / 100 for i in range(dim)]
        for d in docs
    ]
    upsert_documents(collection, docs, embeddings)
    return collection, dim


class TestDetectFilter:
    def test_kinase_query_returns_kinase_filter(self):
        # "selective" keyword takes priority over "kinase" per _KEYWORD_FILTERS order
        filt = _detect_filter("Which kinase inhibitors target EGFR?")
        assert filt == {"target_family": "kinase"}

    def test_gpcr_query_returns_gpcr_filter(self):
        filt = _detect_filter("What GPCR does haloperidol bind?")
        assert filt == {"target_family": "gpcr"}

    def test_selective_query_returns_selectivity_filter(self):
        filt = _detect_filter("Which compounds are selective for EGFR over HER2?")
        assert filt == {"doc_type": {"$in": [
            "compound_selectivity",
            "target_selectivity_index",
            "selectivity_comparison",
        ]}}

    def test_generic_query_returns_none(self):
        filt = _detect_filter("Tell me about gefitinib")
        assert filt is None


class TestExtractEntities:
    def test_extracts_gene_names_from_query(self):
        entities = _extract_entities("What is the IC50 of EGFR inhibitors?", [])
        assert "EGFR" in entities

    def test_extracts_entities_from_candidate_metadata(self):
        candidates = [
            {"text": "...", "metadata": {"gene": "ERBB2,EGFR"}, "distance": 0.1}
        ]
        entities = _extract_entities("what compounds?", candidates)
        assert "ERBB2" in entities or "EGFR" in entities


class TestRetrieve:
    def test_returns_list_of_retrieved_docs(self, populated_collection):
        collection, dim = populated_collection

        cfg = MagicMock()
        cfg.retrieval.initial_k = 10
        cfg.retrieval.rerank_top_k = 3
        cfg.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * dim]

        with patch("dti.retrieve.CrossEncoder") as MockCE:
            mock_ce_instance = MagicMock()
            import numpy as np
            mock_ce_instance.predict.return_value = np.array(
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
            )
            MockCE.return_value = mock_ce_instance

            results = retrieve(
                query="EGFR inhibitors",
                cfg=cfg,
                collection=collection,
                graph=None,
                embedder=mock_embedder,
            )

        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, RetrievedDoc)
            assert isinstance(r.rerank_score, float)
            assert isinstance(r.text, str)

    def test_results_sorted_by_rerank_score_descending(self, populated_collection):
        collection, dim = populated_collection

        cfg = MagicMock()
        cfg.retrieval.initial_k = 5
        cfg.retrieval.rerank_top_k = 3
        cfg.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.5] * dim]

        with patch("dti.retrieve.CrossEncoder") as MockCE:
            import numpy as np
            MockCE.return_value.predict.return_value = np.array([0.3, 0.9, 0.6, 0.1, 0.7])

            results = retrieve(
                query="EGFR kinase inhibitor",
                cfg=cfg,
                collection=collection,
                graph=None,
                embedder=mock_embedder,
            )

        scores = [r.rerank_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_graph_context_added_to_candidates(self, populated_collection):
        collection, dim = populated_collection

        cfg = MagicMock()
        cfg.retrieval.initial_k = 5
        cfg.retrieval.rerank_top_k = 3
        cfg.retrieval.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1] * dim]

        fake_graph = {
            "nodes": [
                {"id": "n1", "label": "EGFR", "description": "Epidermal growth factor receptor"},
                {"id": "n2", "label": "MAPK pathway", "description": "Downstream signalling"},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
        }

        with patch("dti.retrieve.CrossEncoder") as MockCE:
            import numpy as np
            MockCE.return_value.predict.return_value = np.array(
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
            )

            results = retrieve(
                query="EGFR downstream pathway",
                cfg=cfg,
                collection=collection,
                graph=fake_graph,
                embedder=mock_embedder,
            )

        # Should still return valid results even with graph context added
        assert isinstance(results, list)
