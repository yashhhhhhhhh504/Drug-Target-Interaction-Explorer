"""Smoke tests for CLI commands via click.testing.CliRunner — no real API calls."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from dti.cli import cli

FIXTURES = Path(__file__).parent / "fixtures"
TEST_CONFIG = Path(__file__).parent / "fixtures" / "test_config.yaml"


@pytest.fixture(autouse=True)
def write_test_config(tmp_path):
    """Write a minimal config.yaml for CLI tests to a temp location."""
    cfg_content = f"""
embedding:
  provider: biolord
  biolord_model: FremyCompany/BioLORD-2023
  openai_model: text-embedding-3-large
llm:
  provider: anthropic
  anthropic_model: claude-sonnet-4-6-20251001
  ollama_model: llama3.2
  ollama_base_url: http://localhost:11434
retrieval:
  initial_k: 5
  rerank_top_k: 2
  reranker_model: cross-encoder/ms-marco-MiniLM-L-12-v2
data:
  raw_dir: {tmp_path}/raw
  docs_dir: {tmp_path}/docs
  graph_dir: {tmp_path}/graph
  db_dir: {tmp_path}/db
ingest:
  target_families: [kinase]
  activity_types: [IC50]
  max_compounds_per_target: 10
sources:
  chembl: false
  uniprot: false
  pubchem: false
  bindingdb: false
  drugbank: false
mlflow:
  tracking_uri: {tmp_path}/mlruns
  experiment_name: test-experiment
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_content)
    return cfg_path


class TestIngestCommand:
    def test_ingest_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output.lower() or "sources" in result.output.lower()

    def test_ingest_runs_with_no_sources(self, write_test_config):
        runner = CliRunner()
        with patch("dti.ingest.run") as mock_run:
            result = runner.invoke(cli, ["ingest", "--config", str(write_test_config)])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_ingest_passes_source_overrides(self, write_test_config):
        runner = CliRunner()
        with patch("dti.ingest.run") as mock_run:
            result = runner.invoke(
                cli,
                ["ingest", "--config", str(write_test_config), "--sources", "chembl"],
            )
            assert result.exit_code == 0
            call_kwargs = mock_run.call_args
            override = call_kwargs.kwargs.get("override_sources") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else [])
            assert "chembl" in (override or [])


class TestEmbedCommand:
    def test_embed_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["embed", "--help"])
        assert result.exit_code == 0

    def test_embed_runs(self, write_test_config):
        runner = CliRunner()
        with patch("dti.embed.run") as mock_run:
            result = runner.invoke(cli, ["embed", "--config", str(write_test_config)])
            assert result.exit_code == 0
            mock_run.assert_called_once()


class TestGraphCommand:
    def test_graph_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["graph", "--help"])
        assert result.exit_code == 0

    def test_graph_runs(self, write_test_config):
        runner = CliRunner()
        with patch("dti.graph.run") as mock_run:
            result = runner.invoke(cli, ["graph", "--config", str(write_test_config)])
            assert result.exit_code == 0
            mock_run.assert_called_once()


class TestQueryCommand:
    def test_query_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0

    def test_query_runs_and_prints_answer(self, write_test_config):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.answer = "Gefitinib has IC50 = 0.033 uM [Source 1]"
        mock_result.sources = []

        with patch("dti.query.run", return_value=mock_result):
            result = runner.invoke(
                cli,
                ["query", "--config", str(write_test_config),
                 "Which EGFR inhibitors are selective?"],
            )
        assert result.exit_code == 0
        assert "Gefitinib" in result.output

    def test_query_no_sources_flag(self, write_test_config):
        runner = CliRunner()
        mock_result = MagicMock()
        mock_result.answer = "Test answer"
        mock_result.sources = []

        with patch("dti.query.run", return_value=mock_result):
            result = runner.invoke(
                cli,
                ["query", "--config", str(write_test_config),
                 "--no-sources", "test query"],
            )
        assert result.exit_code == 0
        assert "SUPPORTING EVIDENCE" not in result.output
