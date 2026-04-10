#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Drug-Target Interaction Explorer — one-command launcher
#  Usage:  ./start.sh [query|ingest|embed|graph|setup|mlflow]
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
DTI="$VENV/bin/dti"

# ── helpers ──────────────────────────────────────────────────
green()  { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red()    { echo -e "\033[31m$*\033[0m"; }
die()    { red "ERROR: $*"; exit 1; }

require_venv() {
  [[ -f "$DTI" ]] || die "Virtual environment not found. Run:  ./start.sh setup"
}

require_api_key() {
  [[ -f "$SCRIPT_DIR/.env" ]] && source "$SCRIPT_DIR/.env" 2>/dev/null || true
  if [[ -z "${ANTHROPIC_API_KEY:-}" && "${LLM_PROVIDER:-anthropic}" == "anthropic" ]]; then
    yellow "ANTHROPIC_API_KEY is not set."
    yellow "Set it in .env or export it, then re-run."
    yellow "  echo 'ANTHROPIC_API_KEY=sk-...' >> .env"
    exit 1
  fi
}

require_models() {
  # Download/verify all models before first use — safe to re-run (skips cached)
  bash "$SCRIPT_DIR/download_models.sh"
}

require_data() {
  if [[ ! -f "$SCRIPT_DIR/data/raw/targets.jsonl" ]]; then
    yellow "No data found. Running ingest first..."
    run_ingest
  fi
  if [[ ! -d "$SCRIPT_DIR/db/chromadb" ]] || [[ -z "$(ls -A "$SCRIPT_DIR/db/chromadb" 2>/dev/null)" ]]; then
    yellow "ChromaDB is empty. Running embed first..."
    run_embed
  fi
}

# ── stage runners ─────────────────────────────────────────────
run_setup() {
  green "Setting up Drug-Target Interaction Explorer..."
  cd "$SCRIPT_DIR"

  if [[ ! -d "$VENV" ]]; then
    python3 -m venv .venv
    green "Virtual environment created."
  else
    green "Virtual environment already exists."
  fi

  "$VENV/bin/pip" install -e ".[dev]" -q
  green "Dependencies installed."

  if [[ ! -f ".env" ]]; then
    cp .env.example .env
    yellow "Created .env from .env.example — add your ANTHROPIC_API_KEY to .env"
  fi

  green ""
  green "Downloading models (BioLORD + reranker + local LLM)..."
  bash "$SCRIPT_DIR/download_models.sh" || yellow "Model download can be retried with: ./download_models.sh"

  green ""
  green "Setup complete! Next steps:"
  green "  1. Run pipeline:  ./start.sh pipeline   (ingest → embed → graph)"
  green "  2. Start chat:    ./start.sh query"
  green "  (No API key needed — local LLM is the default)"
}

run_ingest() {
  require_venv
  green "Ingesting data from ChEMBL + UniProt..."
  cd "$SCRIPT_DIR"
  [[ -f ".env" ]] && export $(grep -v '^#' .env | xargs) 2>/dev/null || true
  "$DTI" ingest "$@"
  green "Ingest complete → data/raw/"
}

run_embed() {
  require_venv
  green "Embedding documents into ChromaDB..."
  cd "$SCRIPT_DIR"
  [[ -f ".env" ]] && export $(grep -v '^#' .env | xargs) 2>/dev/null || true
  "$DTI" embed
  green "Embedding complete → db/chromadb/"
}

run_graph() {
  require_venv
  green "Building knowledge graph..."
  cd "$SCRIPT_DIR"
  [[ -f ".env" ]] && export $(grep -v '^#' .env | xargs) 2>/dev/null || true
  "$DTI" graph
  green "Graph complete → data/graph/graph.html"
}

run_query() {
  require_venv
  cd "$SCRIPT_DIR"
  [[ -f ".env" ]] && export $(grep -v '^#' .env | xargs) 2>/dev/null || true

  # Download missing models (skips instantly if already cached)
  require_models

  # Check API key only when provider is anthropic
  LLM_PROVIDER_VAL="${LLM_PROVIDER:-$(grep 'provider:' config.yaml | head -1 | awk '{print $2}')}"
  if [[ "${LLM_PROVIDER_VAL}" == "anthropic" ]]; then
    require_api_key
  fi

  require_data

  if [[ -n "${1:-}" ]]; then
    # Single one-shot question
    "$DTI" query "$@"
  else
    # Interactive session: models loaded ONCE, fast for every question
    "$DTI" chat
  fi
}

run_mlflow() {
  require_venv
  green "Opening MLflow UI at http://127.0.0.1:5000"
  cd "$SCRIPT_DIR"
  "$VENV/bin/mlflow" ui --backend-store-uri sqlite:///mlruns.db
}

run_pipeline() {
  green "Running full pipeline: ingest → embed → graph"
  run_ingest
  run_embed
  run_graph
  green ""
  green "Pipeline complete. You can now run:  ./start.sh query"
}

# ── entrypoint ────────────────────────────────────────────────
CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
  setup)    run_setup ;;
  ingest)   run_ingest "$@" ;;
  embed)    run_embed ;;
  graph)    run_graph ;;
  query)    run_query "$@" ;;
  pipeline) run_pipeline ;;
  mlflow)   run_mlflow ;;
  help|--help|-h)
    echo ""
    echo "Usage: ./start.sh <command> [args]"
    echo ""
    echo "Commands:"
    echo "  setup      Create venv, install dependencies, create .env"
    echo "  pipeline   Run full ingest → embed → graph pipeline"
    echo "  ingest     Fetch data from ChEMBL/UniProt (run once)"
    echo "  embed      Embed documents into ChromaDB (run once)"
    echo "  graph      Build graphifyy knowledge graph (optional)"
    echo "  query      Ask one question (or interactive session — models loaded once)"
    echo "  mlflow     Open MLflow tracking UI"
    echo ""
    echo "Examples:"
    echo "  ./start.sh setup"
    echo "  ./start.sh pipeline"
    echo "  ./start.sh query"
    echo "  ./start.sh query \"Which EGFR inhibitors are selective over HER2?\""
    echo "  ./start.sh mlflow"
    ;;
  *)
    red "Unknown command: $CMD"
    echo "Run  ./start.sh help  for usage."
    exit 1
    ;;
esac
