"""Tests for embed.py — BioLORD embedder and provider routing."""

from unittest.mock import MagicMock, patch

import pytest

from dti.embed import BioLORDEmbedder, OllamaEmbedder, OpenAIEmbedder, build_embedder


class TestBioLORDEmbedder:
    def test_encode_returns_list_of_float_lists(self):
        """BioLORD encode() returns one embedding vector per input text."""
        import numpy as np

        mock_model = MagicMock()
        # Simulate 768-dim BioLORD output
        mock_model.encode.return_value = np.array([
            [0.1] * 768,
            [0.2] * 768,
        ])

        embedder = BioLORDEmbedder.__new__(BioLORDEmbedder)
        embedder._model = mock_model
        embedder._model_name = "FremyCompany/BioLORD-2023"

        result = embedder.encode(["EGFR inhibitor", "kinase activity"])
        assert len(result) == 2
        assert len(result[0]) == 768
        assert all(isinstance(v, float) for v in result[0])

    def test_encode_batch_size_respected(self):
        """encode() passes batch_size=32 to the underlying model."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768])

        embedder = BioLORDEmbedder.__new__(BioLORDEmbedder)
        embedder._model = mock_model
        embedder._model_name = "FremyCompany/BioLORD-2023"

        embedder.encode(["test"])
        call_kwargs = mock_model.encode.call_args
        assert call_kwargs.kwargs.get("batch_size") == 32

    def test_biolord_is_default_provider(self):
        """build_embedder() returns BioLORDEmbedder when provider=biolord."""
        cfg = MagicMock()
        cfg.embedding.provider = "biolord"
        cfg.embedding.biolord_model = "FremyCompany/BioLORD-2023"

        with patch("dti.embed.BioLORDEmbedder") as MockBioLORD:
            MockBioLORD.return_value = MagicMock()
            embedder = build_embedder(cfg)
            MockBioLORD.assert_called_once_with("FremyCompany/BioLORD-2023")


class TestOllamaEmbedder:
    def test_encode_calls_ollama_api(self):
        import requests as req

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

        embedder = OllamaEmbedder.__new__(OllamaEmbedder)
        embedder._model_name = "nomic-embed-text"
        embedder._base_url = "http://localhost:11434"
        embedder._requests = MagicMock()
        embedder._requests.post.return_value = mock_resp

        result = embedder.encode(["test text"])
        assert result == [[0.1, 0.2, 0.3]]
        embedder._requests.post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "test text"},
            timeout=60,
        )

    def test_encode_multiple_texts(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.5, 0.6]}

        embedder = OllamaEmbedder.__new__(OllamaEmbedder)
        embedder._model_name = "nomic-embed-text"
        embedder._base_url = "http://localhost:11434"
        embedder._requests = MagicMock()
        embedder._requests.post.return_value = mock_resp

        result = embedder.encode(["text one", "text two", "text three"])
        assert len(result) == 3
        assert embedder._requests.post.call_count == 3


class TestBuildEmbedder:
    def test_routes_to_biolord(self):
        cfg = MagicMock()
        cfg.embedding.provider = "biolord"
        cfg.embedding.biolord_model = "FremyCompany/BioLORD-2023"

        with patch("dti.embed.BioLORDEmbedder") as Mock:
            Mock.return_value = MagicMock()
            build_embedder(cfg)
            Mock.assert_called_once_with("FremyCompany/BioLORD-2023")

    def test_routes_to_openai(self):
        cfg = MagicMock()
        cfg.embedding.provider = "openai"
        cfg.embedding.openai_model = "text-embedding-3-large"

        with patch("dti.embed.OpenAIEmbedder") as Mock:
            Mock.return_value = MagicMock()
            build_embedder(cfg)
            Mock.assert_called_once_with("text-embedding-3-large")

    def test_routes_to_ollama(self):
        cfg = MagicMock()
        cfg.embedding.provider = "ollama"
        cfg.embedding.ollama_model = "nomic-embed-text"
        cfg.embedding.ollama_base_url = "http://localhost:11434"

        with patch("dti.embed.OllamaEmbedder") as Mock:
            Mock.return_value = MagicMock()
            build_embedder(cfg)
            Mock.assert_called_once_with("nomic-embed-text", "http://localhost:11434")

    def test_unknown_provider_falls_back_to_biolord(self):
        cfg = MagicMock()
        cfg.embedding.provider = "unknown_provider"
        cfg.embedding.biolord_model = "FremyCompany/BioLORD-2023"

        with patch("dti.embed.BioLORDEmbedder") as Mock:
            Mock.return_value = MagicMock()
            build_embedder(cfg)
            Mock.assert_called_once()
