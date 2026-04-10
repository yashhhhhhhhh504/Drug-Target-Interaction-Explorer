"""Tests for graph.py — traverse(), find_relations(), is_relation_query()."""

import pytest

from dti.graph import find_relations, is_relation_query, traverse

# ── minimal test graph ─────────────────────────────────────────────────────
# EGFR --[inhibits]--> Gefitinib
# EGFR --[associated with]--> Lung cancer
# Gefitinib --[targets]--> VEGFR2
# MAPK --[downstream of]--> EGFR

FAKE_GRAPH = {
    "nodes": [
        {"id": "n1", "label": "EGFR", "description": "Epidermal growth factor receptor"},
        {"id": "n2", "label": "Gefitinib", "description": "EGFR inhibitor IC50=0.033uM"},
        {"id": "n3", "label": "Lung cancer", "description": "Non-small cell lung carcinoma"},
        {"id": "n4", "label": "VEGFR2", "description": "Vascular endothelial growth factor receptor 2"},
        {"id": "n5", "label": "MAPK pathway", "description": "Mitogen-activated protein kinase signalling"},
    ],
    "edges": [
        {"source": "n1", "target": "n2", "label": "inhibited by"},
        {"source": "n1", "target": "n3", "label": "associated with"},
        {"source": "n2", "target": "n4", "label": "targets"},
        {"source": "n5", "target": "n1", "label": "downstream of"},
    ],
}


# ── is_relation_query ──────────────────────────────────────────────────────

class TestIsRelationQuery:
    def test_relation_keyword_detected(self):
        assert is_relation_query("What is the relation between EGFR and HER2?")

    def test_pathway_keyword_detected(self):
        assert is_relation_query("Which pathway does EGFR activate?")

    def test_downstream_keyword_detected(self):
        assert is_relation_query("What is downstream of EGFR?")

    def test_connected_keyword_detected(self):
        assert is_relation_query("How are EGFR and MAPK connected?")

    def test_potency_query_not_relation(self):
        assert not is_relation_query("What is the IC50 of gefitinib?")

    def test_selectivity_query_not_relation(self):
        assert not is_relation_query("Which EGFR inhibitors are selective?")


# ── traverse ───────────────────────────────────────────────────────────────

class TestTraverse:
    def test_returns_relationship_sentences(self):
        snippets = traverse(FAKE_GRAPH, ["EGFR"], depth=1)
        assert len(snippets) > 0
        # Should be sentences like "EGFR [inhibited by] Gefitinib: ..."
        assert any("EGFR" in s or "Gefitinib" in s or "Lung cancer" in s for s in snippets)

    def test_sentences_include_relation_context(self):
        snippets = traverse(FAKE_GRAPH, ["EGFR"], depth=1)
        # Sentences should contain biological relationship language
        assert any(
            "associated" in s.lower() or "member" in s.lower()
            or "binds" in s.lower() or "inhibits" in s.lower()
            or "involved" in s.lower() or "function" in s.lower()
            for s in snippets
        )

    def test_depth_1_returns_fewer_results_than_depth_2(self):
        snippets_d1 = traverse(FAKE_GRAPH, ["EGFR"], depth=1)
        snippets_d2 = traverse(FAKE_GRAPH, ["EGFR"], depth=2)
        # Deeper traversal always returns at least as many relationship sentences
        assert len(snippets_d2) >= len(snippets_d1)

    def test_depth_2_returns_two_hop_neighbours(self):
        snippets = traverse(FAKE_GRAPH, ["EGFR"], depth=2)
        # VEGFR2 is 2 hops away — should appear at depth=2
        assert any("VEGFR2" in s for s in snippets)

    def test_empty_graph_returns_empty(self):
        assert traverse({}, ["EGFR"]) == []

    def test_no_matching_entity_returns_empty(self):
        snippets = traverse(FAKE_GRAPH, ["UNKNOWN_GENE_XYZ"])
        assert snippets == []

    def test_case_insensitive_entity_match(self):
        snippets = traverse(FAKE_GRAPH, ["egfr"])
        assert len(snippets) > 0

    def test_includes_description_in_sentence(self):
        snippets = traverse(FAKE_GRAPH, ["EGFR"], depth=1)
        # Gefitinib has description "EGFR inhibitor IC50=0.033uM" — should appear
        assert any("IC50" in s or "inhibitor" in s for s in snippets)


# ── find_relations ─────────────────────────────────────────────────────────

class TestFindRelations:
    def test_finds_path_between_connected_entities(self):
        # EGFR and Lung cancer are directly connected (1 hop)
        paths = find_relations(FAKE_GRAPH, "EGFR", "Lung cancer")
        assert len(paths) > 0
        # Shortest path should be short (direct or via 1 intermediate)
        assert len(paths[0]) <= 3

    def test_finds_two_hop_path(self):
        # EGFR → Gefitinib → VEGFR2 is a 2-hop path
        paths = find_relations(FAKE_GRAPH, "EGFR", "VEGFR2", max_depth=3)
        assert len(paths) > 0
        assert len(paths[0]) == 2

    def test_path_contains_relation_sentences(self):
        paths = find_relations(FAKE_GRAPH, "EGFR", "Lung cancer")
        assert len(paths) > 0
        for sentence in paths[0]:
            # Sentences should be non-empty natural language
            assert len(sentence) > 5

    def test_returns_empty_when_target_not_in_graph(self):
        # "Aspirin" does not exist as a node — no path possible
        paths = find_relations(FAKE_GRAPH, "EGFR", "Aspirin", max_depth=3)
        assert paths == []

    def test_returns_empty_for_unknown_entity(self):
        paths = find_relations(FAKE_GRAPH, "UNKNOWN", "EGFR")
        assert paths == []

    def test_returns_empty_graph(self):
        assert find_relations({}, "EGFR", "HER2") == []

    def test_shortest_paths_first(self):
        paths = find_relations(FAKE_GRAPH, "EGFR", "VEGFR2", max_depth=3)
        if len(paths) > 1:
            # Shorter paths should come first
            lengths = [len(p) for p in paths]
            assert lengths == sorted(lengths)


# ── retrieve integration: graph context improves candidates ────────────────

class TestGraphAwareRetrieval:
    """Tests that retrieve() correctly injects and formats graph context."""

    def test_graph_context_formatted_with_header(self):
        from dti.retrieve import _format_graph_context
        result = _format_graph_context("EGFR [inhibited by] Gefitinib: IC50=0.033uM", ["EGFR"])
        assert "[Biological Knowledge Graph" in result
        assert "EGFR" in result
        assert "Gefitinib" in result

    def test_relation_query_triggers_depth_2(self):
        """Verify that is_relation_query() returns True for the right queries."""
        from dti.graph import is_relation_query
        assert is_relation_query("How is EGFR related to the MAPK pathway?")
        assert is_relation_query("What is the interaction between EGFR and HER2?")
        assert not is_relation_query("What is the IC50 of erlotinib?")
