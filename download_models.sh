#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  download_models.sh — pre-cache all models used by the DTI pipeline
#
#  Run automatically by start.sh on first use.
#  Safe to re-run: already-cached models are skipped instantly.
#
#  Models downloaded:
#    1. FremyCompany/BioLORD-2023            (~460 MB)  — biomedical embedder
#    2. cross-encoder/ms-marco-MiniLM-L-12-v2 (~130 MB) — reranker
#    3. Local LLM from config.yaml           (~4–8 GB)  — generative model
#       default: microsoft/Phi-3-mini-4k-instruct
#       alt:     BioMistral/BioMistral-7B-DARE (set local_model in config.yaml)
#
#  All models are stored in: ~/.cache/huggingface/hub/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PY="$VENV/bin/python3"
CONFIG="$SCRIPT_DIR/config.yaml"

green()  { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red()    { echo -e "\033[31m$*\033[0m"; }

[[ -f "$PY" ]] || { red "Virtual environment not found. Run: ./start.sh setup"; exit 1; }

# ── Read local_model from config.yaml ────────────────────────────────────────
LOCAL_MODEL=$("$PY" -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
llm = cfg.get('llm', {})
# Env var override
import os
provider = os.environ.get('LLM_PROVIDER', llm.get('provider','local'))
model = llm.get('local_model','microsoft/Phi-3-mini-4k-instruct')
print(model)
" 2>/dev/null || echo "microsoft/Phi-3-mini-4k-instruct")

LLM_PROVIDER=$("$PY" -c "
import yaml, os
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(os.environ.get('LLM_PROVIDER', cfg.get('llm',{}).get('provider','local')))
" 2>/dev/null || echo "local")

# ── Helper: check if a HuggingFace model is already cached ───────────────────
model_is_cached() {
  local model_id="$1"
  # Convert "org/model" → "models--org--model" directory name
  local dir_name="models--$(echo "$model_id" | tr '/' '--')"
  local cache_dir="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
  local model_dir="$cache_dir/$dir_name"
  # Model is cached if the snapshots directory exists and is non-empty
  [[ -d "$model_dir/snapshots" ]] && [[ -n "$(ls -A "$model_dir/snapshots" 2>/dev/null)" ]]
}

# ── 1. BioLORD embedding model ────────────────────────────────────────────────
BIOLORD="FremyCompany/BioLORD-2023"
if model_is_cached "$BIOLORD"; then
  green "✓ BioLORD already cached"
else
  yellow "Downloading BioLORD embedding model (~460 MB)…"
  "$PY" -c "
from sentence_transformers import SentenceTransformer
import logging
logging.disable(logging.CRITICAL)
SentenceTransformer('$BIOLORD')
print('  BioLORD cached.')
"
  green "✓ BioLORD downloaded"
fi

# ── 2. Cross-encoder reranker ─────────────────────────────────────────────────
RERANKER="cross-encoder/ms-marco-MiniLM-L-12-v2"
if model_is_cached "$RERANKER"; then
  green "✓ Reranker already cached"
else
  yellow "Downloading cross-encoder reranker (~130 MB)…"
  "$PY" -c "
from sentence_transformers import CrossEncoder
import logging
logging.disable(logging.CRITICAL)
CrossEncoder('$RERANKER')
print('  Reranker cached.')
"
  green "✓ Reranker downloaded"
fi

# ── 3. Local generative LLM ───────────────────────────────────────────────────
if [[ "$LLM_PROVIDER" == "local" ]]; then
  if model_is_cached "$LOCAL_MODEL"; then
    green "✓ Local LLM ($LOCAL_MODEL) already cached"
  else
    yellow ""
    yellow "Downloading local LLM: $LOCAL_MODEL"
    if [[ "$LOCAL_MODEL" == *"BioMistral"* ]]; then
      yellow "  Size: ~8 GB — this will take several minutes on first run."
    else
      yellow "  Size: ~4 GB — this will take a few minutes on first run."
    fi
    yellow "  Files saved to: ~/.cache/huggingface/hub/"
    yellow ""

    # snapshot_download saves to disk without loading into RAM — much faster
    "$PY" -c "
import logging, warnings
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')
from huggingface_hub import snapshot_download
model_name = '$LOCAL_MODEL'
print(f'  Downloading {model_name}…', flush=True)
path = snapshot_download(
    repo_id=model_name,
    ignore_patterns=['*.gguf', 'original/*'],
)
print(f'  Saved to: {path}')
"
    green "✓ Local LLM downloaded and cached"
  fi
else
  green "✓ LLM provider is '$LLM_PROVIDER' — no local model download needed"
fi

green ""
green "All models ready. Pipeline will start instantly."
