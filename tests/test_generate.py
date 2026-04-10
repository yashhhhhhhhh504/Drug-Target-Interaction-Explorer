"""Unit tests for generate.py — mocks LLM calls, no real API requests."""

from unittest.mock import MagicMock, patch

import pytest

from dti.generate import _SYSTEM_PROMPT, build_context, generate
from dti.retrieve import RetrievedDoc


def make_doc(doc_id: str, text: str, score: float = 0.9) -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc_id,
        text=text,
        metadata={"doc_type": "target_profile"},
        vector_distance=0.1,
        rerank_score=score,
    )


@pytest.fixture
def sample_docs():
    return [
        make_doc("doc1", "EGFR: IC50 = 0.033 uM for Gefitinib", score=0.95),
        make_doc("doc2", "ERBB2: IC50 = 3.7 uM for Gefitinib", score=0.80),
    ]


class TestBuildContext:
    def test_includes_source_labels(self, sample_docs):
        context = build_context(sample_docs)
        assert "[Source 1]" in context
        assert "[Source 2]" in context

    def test_includes_document_text(self, sample_docs):
        context = build_context(sample_docs)
        assert "Gefitinib" in context
        assert "0.033" in context

    def test_includes_relevance_score(self, sample_docs):
        context = build_context(sample_docs)
        assert "0.950" in context or "0.95" in context

    def test_empty_docs_returns_empty_string(self):
        assert build_context([]) == ""


class TestGenerate:
    def test_no_docs_returns_helpful_message(self):
        cfg = MagicMock()
        result = generate("What is gefitinib?", [], cfg)
        assert "No relevant documents" in result

    def test_anthropic_provider_calls_anthropic(self, sample_docs):
        cfg = MagicMock()
        cfg.llm.provider = "anthropic"
        cfg.llm.anthropic_model = "claude-sonnet-4-6-20251001"

        with patch("dti.generate._call_anthropic", return_value="Mocked answer") as mock_call:
            result = generate("What is the IC50 of gefitinib?", sample_docs, cfg)
            mock_call.assert_called_once()
            assert result == "Mocked answer"

    def test_ollama_provider_calls_ollama(self, sample_docs):
        cfg = MagicMock()
        cfg.llm.provider = "ollama"
        cfg.llm.ollama_model = "llama3.2"

        with patch("dti.generate._call_ollama", return_value="Ollama answer") as mock_call:
            result = generate("What is the IC50 of gefitinib?", sample_docs, cfg)
            mock_call.assert_called_once()
            assert result == "Ollama answer"

    def test_llm_failure_returns_fallback_with_docs(self, sample_docs):
        cfg = MagicMock()
        cfg.llm.provider = "anthropic"

        with patch("dti.generate._call_anthropic", side_effect=RuntimeError("API down")):
            result = generate("What is gefitinib?", sample_docs, cfg)
            assert "LLM generation failed" in result
            assert "Retrieved evidence" in result

    def test_system_prompt_contains_citation_instructions(self):
        assert "[Source N]" in _SYSTEM_PROMPT
        assert "IC50" in _SYSTEM_PROMPT

    def test_user_message_contains_query_and_context(self, sample_docs):
        """Verify the user message structure passed to the LLM."""
        cfg = MagicMock()
        cfg.llm.provider = "anthropic"

        captured = {}

        def fake_anthropic(user_msg, cfg):
            captured["user_msg"] = user_msg
            return "answer"

        with patch("dti.generate._call_anthropic", side_effect=fake_anthropic):
            generate("Which drugs target EGFR?", sample_docs, cfg)

        assert "Which drugs target EGFR?" in captured["user_msg"]
        assert "[Source 1]" in captured["user_msg"]
        assert "Gefitinib" in captured["user_msg"]
