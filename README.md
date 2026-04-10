# Drug-Target Interaction Explorer

A fully local, offline-capable RAG pipeline for drug discovery research. It ingests bioactivity data from ChEMBL, UniProt, PubChem, and BindingDB, builds a biomedical knowledge graph using graphifyy, and answers natural language questions about drug-target interactions with cited, quantitative evidence.

```
"Tell me about BTK as a drug target - what diseases is it relevant to
 and what are the most potent compounds against it?"

 BTK (Bruton's tyrosine kinase) is a non-receptor tyrosine kinase
 indispensable for B lymphocyte development, differentiation and signaling.
 BTK is associated with X-linked agammaglobulinemia [Source 7].
 The most potent compound is CHEMBL458333 (Ki=0.25 nM) [Source 1].
 CHEMBL1916891 shows 873x selectivity for BTK over FRK (IC50=0.7 nM) [Source 1]...
```

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
- [Usage](#usage)
- [Configuration](#configuration)
- [LLM Providers](#llm-providers)
- [Data Sources](#data-sources)
- [Document Types](#document-types)
- [Knowledge Graph](#knowledge-graph)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Benchmark Prompts](#benchmark-prompts)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Multi-source data ingestion** from ChEMBL, UniProt, PubChem, BindingDB (and optionally DrugBank)
- **7 specialized document types** covering target profiles, compound activity, selectivity comparisons, evidence depth, and assay discrepancies
- **Biomedical knowledge graph** with disease associations, GO terms, protein families, and drug-target edges — queryable at runtime
- **BioLORD embeddings** (biomedically-tuned sentence transformer) for domain-specific semantic search
- **Cross-encoder re-ranking** (ms-marco-MiniLM) with document-type diversity enforcement
- **3 LLM providers**: fully local (Phi-3-mini, no API key), Ollama (any local model), or Anthropic Claude (best quality)
- **Research-grade prompts**: anti-hallucination rules, evidence vs relevance distinction, IC50/Ki methodology caveats, selectivity direction checks
- **Interactive chat mode** with warm model caching (models load once, queries are fast)
- **MLflow experiment tracking** for every query, ingest, and embed run
- **One-command setup**: `./start.sh setup` handles everything

---

## Architecture

```
                          RAW DATA SOURCES
           ChEMBL    UniProt    PubChem    BindingDB
              |          |         |           |
              v          v         v           v
         +-------------------------------------------------+
         |              INGEST STAGE                        |
         |  Fetch targets by family (kinase, GPCR, NR)      |
         |  Fetch compound activities (IC50, Ki, EC50)       |
         |  Cross-enrich with UniProt GO terms & diseases    |
         +-------------------------------------------------+
                            |
              targets.jsonl + compounds.jsonl
                            |
              +-------------+-------------+
              |                           |
              v                           v
   +--------------------+     +------------------------+
   |    EMBED STAGE      |     |    GRAPH STAGE         |
   |                     |     |                        |
   | 7 document types    |     | NetworkX graph         |
   | BioLORD embeddings  |     | Nodes: targets,        |
   | ChromaDB storage    |     |   compounds, diseases,  |
   |                     |     |   GO terms, families    |
   +--------------------+     | D3.js visualization     |
              |                +------------------------+
              |                           |
              v                           v
   +--------------------------------------------------+
   |              QUERY PIPELINE                        |
   |                                                    |
   |  1. Metadata pre-filter (selectivity/evidence/     |
   |     research query detection)                      |
   |  2. BioLORD vector search (top-50 candidates)      |
   |  3. Selectivity & evidence doc injection            |
   |  4. Knowledge graph traversal (depth 1-2)           |
   |     - Disease associations, GO terms, pathways      |
   |     - Batched by category for dense context         |
   |  5. Cross-encoder re-ranking (ms-marco MiniLM)      |
   |  6. Doc-type diversity enforcement                  |
   |  7. Smart context packing for LLM token budget      |
   |  8. LLM generation with anti-hallucination prompt   |
   +--------------------------------------------------+
                            |
                            v
              Answer with [Source N] citations
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/drug-target-explorer.git
cd drug-target-explorer

# One-command setup (creates venv, installs deps, downloads models)
./start.sh setup

# Run the full pipeline (ingest data → embed → build graph)
./start.sh pipeline

# Start interactive chat
./start.sh query
```

**That's it.** No API keys needed — the default configuration uses a fully local LLM (Phi-3-mini-4k-instruct, ~4GB).

---

## Installation

### Prerequisites

- **Python 3.11+**
- **~6 GB disk space** for models (BioLORD ~460MB, reranker ~130MB, Phi-3-mini ~4GB)
- **8 GB+ RAM** (16 GB recommended for comfortable operation)
- **macOS** (Apple Silicon MPS acceleration) or **Linux** (CUDA GPU optional)

### Manual Setup

If you prefer not to use `start.sh`:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Download models (BioLORD + reranker + local LLM)
bash download_models.sh

# Run pipeline stages
dti ingest      # Fetch data from ChEMBL, UniProt, etc.
dti embed       # Embed documents into ChromaDB
dti graph       # Build knowledge graph

# Start querying
dti chat        # Interactive mode (recommended)
dti query "Your question here"  # Single question
```

### Verifying Installation

```bash
# Check CLI is available
dti --version

# Check all stages
dti ingest --help
dti embed --help
dti graph --help
dti query --help
dti chat --help
dti relations --help
```

---

## Pipeline Stages

### 1. Ingest (`dti ingest`)

Fetches target and compound data from enabled sources:

- **ChEMBL**: Primary source for bioactivity data (IC50, Ki, EC50 values), enriched with pChEMBL values, assay types, document IDs, and data validity comments
- **UniProt**: Protein function descriptions, GO terms (biological process, molecular function, cellular compartment), disease associations
- **PubChem**: Additional compound annotations
- **BindingDB**: Ki/Kd binding data for UniProt-accession targets

Output: `data/raw/targets.jsonl` and `data/raw/compounds.jsonl`

```bash
# Default: all enabled sources, all families
dti ingest

# Override families
dti ingest --families kinase --families gpcr

# Override sources
dti ingest --sources chembl --sources uniprot
```

### 2. Embed (`dti embed`)

Generates 7 document types from raw data, embeds them with BioLORD, and stores in ChromaDB:

```bash
dti embed
# Output: db/chromadb/ (persistent vector database)
```

This stage is **idempotent** — re-running skips already-embedded documents.

### 3. Graph (`dti graph`)

Builds a biomedical knowledge graph and interactive HTML visualization:

```bash
dti graph
# Output: data/graph/graph.json + data/graph/graph.html

# Open visualization in browser
open data/graph/graph.html
```

### 4. Query (`dti query` / `dti chat`)

```bash
# Single question
dti query "Which BTK inhibitors have >10x selectivity?"

# Interactive session (models loaded once — much faster for multiple questions)
dti chat

# Explore entity relationships in the knowledge graph
dti relations BTK "agammaglobulinemia"
dti relations TRETINOIN RARA --depth 2
```

---

## Usage

### Interactive Chat (Recommended)

```bash
./start.sh query
# or
source .venv/bin/activate
dti chat
```

Models load once (~5-10s), then each query runs in ~15-30s. Type `help` for examples, `exit` to quit.

### Single Queries

```bash
dti query "What is the most potent compound against BTK?"
dti query "Compare selectivity of CHEMBL1997617 across its tested targets"
dti query "Which nuclear receptor has the best evidence depth?"
```

### Knowledge Graph Relations

```bash
# Find paths between two entities
dti relations BTK "X-linked agammaglobulinemia"
dti relations RARA TRETINOIN

# Open interactive graph
dti relations BTK EGFR --open-graph
```

### MLflow Experiment Tracking

```bash
./start.sh mlflow
# Opens UI at http://127.0.0.1:5000
```

Every ingest, embed, and query run is tracked with parameters, metrics, and artifacts.

---

## Configuration

All settings are in `config.yaml`. Environment variables in `.env` override select values.

### `config.yaml`

```yaml
embedding:
  provider: biolord          # "biolord" (local, recommended) | "openai" | "ollama"
  biolord_model: FremyCompany/BioLORD-2023
  openai_model: text-embedding-3-large
  ollama_model: nomic-embed-text
  ollama_base_url: http://localhost:11434

llm:
  provider: local            # "local" (no API key) | "ollama" | "anthropic"
  anthropic_model: claude-sonnet-4-6-20251001
  ollama_model: llama3.2
  ollama_base_url: http://localhost:11434

  # Local HuggingFace LLM options:
  local_model: microsoft/Phi-3-mini-4k-instruct   # 3.8B, ~4GB, fast
  # local_model: microsoft/Phi-3-mini-128k-instruct  # same model, 128k context (needs 24GB+ GPU)
  # local_model: BioMistral/BioMistral-7B-DARE       # 7B, ~8GB, biomedical fine-tuned
  local_device: auto         # "auto" | "mps" | "cuda" | "cpu"
  local_max_new_tokens: 500  # increase to 800-1000 for longer research answers

retrieval:
  initial_k: 50             # candidates from vector search
  rerank_top_k: 8           # final docs after cross-encoder reranking
  reranker_model: cross-encoder/ms-marco-MiniLM-L-12-v2

ingest:
  target_families:
    - kinase
    - gpcr
    - nuclear_receptor
  activity_types:
    - IC50
    - Ki
    - EC50
  max_compounds_per_target: 5000

sources:
  chembl: true
  uniprot: true
  pubchem: true
  bindingdb: true
  drugbank: false            # requires DRUGBANK_API_KEY

mlflow:
  tracking_uri: sqlite:///mlruns.db
  experiment_name: drug-target-explorer
```

### Environment Variables (`.env`)

```bash
# Fully local — no keys needed (default)
# LLM_PROVIDER=local
# EMBEDDING_PROVIDER=biolord

# Anthropic (best quality answers)
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local server, larger models)
# LLM_PROVIDER=ollama
# EMBEDDING_PROVIDER=ollama

# Apple Silicon MPS memory management
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

---

## LLM Providers

| Provider | Model | Quality | Speed | Cost | Setup |
|----------|-------|---------|-------|------|-------|
| **local** (default) | Phi-3-mini-4k (3.8B) | Good | ~20s/query | Free | None |
| **ollama** | llama3.2 / mistral / etc. | Very Good | ~10-30s | Free | `ollama serve` |
| **anthropic** | Claude Sonnet | Excellent | ~5s | API credits | `ANTHROPIC_API_KEY` |

### Switching Providers

Edit `config.yaml`:

```yaml
llm:
  provider: anthropic   # or "ollama" or "local"
```

Or use environment variable:

```bash
LLM_PROVIDER=anthropic dti query "your question"
```

### Ollama Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Start server and pull models
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text  # optional: for Ollama embeddings too

# Set provider
export LLM_PROVIDER=ollama
dti chat
```

---

## Data Sources

### ChEMBL (Primary)

The main source for quantitative bioactivity data. Fetches:
- IC50, Ki, EC50 values in nM
- pChEMBL values for cross-endpoint comparison
- Assay type (B=Binding, F=Functional, A=ADME, T=Toxicity)
- Assay and document IDs for evidence tracking
- Data validity comments (ChEMBL quality flags)
- Target organism

### UniProt

Enriches targets with biological context:
- Protein function descriptions
- GO terms: biological process (P), molecular function (F), cellular compartment (C)
- Disease associations
- UniProt accession IDs (enables BindingDB cross-reference)

### PubChem

Additional compound metadata and annotations.

### BindingDB

Binding affinity data (Ki, Kd) accessed via UniProt accession IDs. Enriches the activity dataset with additional experimental measurements.

### DrugBank (Optional)

Requires `DRUGBANK_API_KEY`. Enable in config:

```yaml
sources:
  drugbank: true
```

---

## Document Types

The pipeline generates 7 specialized document types, each optimized for different query patterns:

| Type | Per | Purpose | Example Query |
|------|-----|---------|--------------|
| **target_profile** | target | Protein function, GO terms, diseases, activity summary | "What diseases is BTK associated with?" |
| **compound_activity** | compound | Activity across all tested targets | "What targets does CHEMBL1997617 hit?" |
| **selectivity_comparison** | target pair | Head-to-head selectivity (both directions) | "Which compounds are selective for PDE3A over PDE3B?" |
| **compound_selectivity** | compound | Ranked potency profile across targets | "How selective is Dasatinib?" |
| **target_selectivity_index** | target | Comprehensive selectivity landscape | "What is the most selective BTK inhibitor?" |
| **evidence_depth** | target | Confidence scores, n_assays, n_pubs, CV | "How reliable is the data for BTK inhibitors?" |
| **assay_discrepancy** | global | Compound-target pairs with >10x disagreement | "Are there inconsistencies in the kinase data?" |

### Evidence Depth Scoring

Each compound-target pair gets a confidence score (0-100) based on:

| Factor | Points | Meaning |
|--------|--------|---------|
| n_assays x 10 | 0-50 | Independent experimental validations |
| n_publications x 10 | 0-30 | Independent papers reporting data |
| n_records x 2 | 0-10 | Total measurements |
| pChEMBL available | +10 | Normalized cross-endpoint comparison possible |
| CV < 0.5 | +15 | Good value consistency across measurements |
| No validity flags | +5 | No ChEMBL quality warnings |

Interpretation: >60 = well-validated, 30-60 = moderate, <30 = weak evidence.

---

## Knowledge Graph

The knowledge graph enriches every query with biological context:

### Node Types

| Type | Color | Examples |
|------|-------|---------|
| Target (protein) | Blue | BTK, EGFR, RARA |
| Compound (drug) | Orange | Dasatinib, Tretinoin |
| Disease | Red | X-linked agammaglobulinemia |
| Family | Teal | Kinase, GPCR, Nuclear Receptor |
| GO Term | Green | MAPK cascade, ATP binding |

### Edge Types

- `inhibits` / `activates` / `binds` (compound -> target, with IC50/Ki value)
- `associated_with` (target -> disease)
- `belongs_to_family` (target -> family)
- `involved_in` / `has_molecular_function` / `located_in` (target -> GO term)

### Graph Traversal at Query Time

- **All queries**: depth-1 traversal (direct neighbors of mentioned entities)
- **Relation queries** ("how is X connected to Y?"): depth-2 traversal
- Graph snippets are **batched by category** (diseases, GO terms, drug interactions) into dense documents that compete well in cross-encoder reranking

### Interactive Visualization

```bash
open data/graph/graph.html
```

D3.js force-directed graph with click-to-inspect nodes, zoom, and drag.

---

## Retrieval Pipeline

The retrieval pipeline uses a 6-step process to find the most relevant documents:

### 1. Query Classification

Queries are classified into types that determine retrieval strategy:

- **Selectivity queries** ("selective for X over Y"): pre-filter to selectivity docs
- **Evidence queries** ("how reliable", "confidence"): inject evidence_depth docs
- **Research queries** ("mechanism", "therapeutic", "which would you advance"): ensure diverse doc types (target profile + graph context + activity data)
- **Relation queries** ("how is X connected to Y"): deeper graph traversal
- **Simple queries**: standard vector search

### 2. Vector Search

BioLORD embeds the query and retrieves top-50 candidates from ChromaDB with cosine similarity.

### 3. Document Injection

- **Selectivity queries**: inject both directional selectivity_comparison docs (BioLORD can't distinguish "A over B" from "B over A")
- **Evidence queries**: inject evidence_depth, assay_discrepancy, and compound_activity docs

### 4. Knowledge Graph Enrichment

Graph traversal adds biological context as batched documents:
- Disease associations batch
- GO terms / biological functions batch
- Drug-target interaction batch
- Classification / family batch

### 5. Cross-Encoder Re-ranking

ms-marco-MiniLM-L-12-v2 scores all candidates against the query. Top-8 survive.

### 6. Diversity Enforcement

Ensures critical doc types aren't crowded out:
- Evidence queries: at least 1 evidence_depth + 1 compound_activity doc
- Research queries: at least 1 target_profile + 1 graph_context doc
- Replaces the lowest-scored doc of the **most common** type (usually compound_selectivity)

### 7. Smart Context Packing (Local LLM)

For the 4k-context Phi-3 model, docs are selected by diversity rather than just top-N:
- **Evidence queries**: evidence_depth first, then activity, then bio context (3 docs)
- **Research queries**: bio context first, then activity, then evidence (3 docs)
- **Simple queries**: top-1 doc only

---

## Benchmark Prompts

Test the pipeline with these prompts, ordered by difficulty:

### Tier 1 — Basic Retrieval

```
"What is the most potent compound against BTK?"
"List compounds selective for PDE3A over PDE3B"
"What is the IC50 of staurosporine?"
```

### Tier 2 — Multi-dimensional

```
"Compare BTK inhibitors CHEMBL1997617 and CHEMBL281957 - which has better selectivity?"
"What diseases could be treated by inhibiting RARA, RARB, and RARG?"
"For the kinase family, which target has the weakest evidence despite potent compounds?"
```

### Tier 3 — Research Synthesis (hardest)

```
"BTK is associated with X-linked agammaglobulinemia. Explain the biological
mechanism linking BTK inhibition to B-cell deficiency using GO terms and
pathway data. Would a compound with 79x selectivity for BTK over MAP4K5
be safe for this indication?"

"Rank all targets by drug development readiness considering potent compounds
under 100 nM, evidence depth, disease relevance, and selectivity. Which
target should enter lead optimization first?"

"A compound tested against BTK shows IC50=0.4 nM but was only tested against
2 targets. Another shows Ki=0.7943 nM across 8 targets with 79x selectivity.
The first has no evidence depth data. Which is the better clinical candidate?
Address the IC50 vs Ki comparison caveat."
```

> **Note:** Tier 3 prompts push Phi-3-mini (3.8B) to its limits. For best results on complex research questions, use `provider: anthropic` or `provider: ollama` with a 7B+ model.

---

## Project Structure

```
drug-target-explorer/
|-- config.yaml              # All pipeline configuration
|-- pyproject.toml            # Python package definition
|-- start.sh                  # One-command launcher
|-- download_models.sh        # Model download script
|-- .env.example              # Environment variable template
|-- .gitignore
|-- README.md
|
|-- src/dti/                  # Main package
|   |-- __init__.py
|   |-- cli.py                # Click CLI (dti ingest/embed/graph/query/chat/relations)
|   |-- config.py             # Config loading + env var overrides
|   |-- ingest.py             # Multi-source data ingestion
|   |-- chunk.py              # 7 document type generators (926 lines)
|   |-- embed.py              # BioLORD/OpenAI/Ollama embedding + ChromaDB storage
|   |-- store.py              # ChromaDB read/write operations
|   |-- graph.py              # Knowledge graph building + D3.js visualization
|   |-- retrieve.py           # Vector search + graph traversal + reranking
|   |-- generate.py           # LLM answer generation (3 providers)
|   |-- query.py              # Query orchestration + MLflow logging
|   |
|   |-- sources/              # Data source adapters
|       |-- base.py           # Abstract base class
|       |-- chembl.py         # ChEMBL REST API
|       |-- uniprot.py        # UniProt REST API
|       |-- pubchem.py        # PubChem REST API
|       |-- bindingdb.py      # BindingDB REST API
|       |-- drugbank.py       # DrugBank API (optional)
|
|-- tests/                    # Test suite
|   |-- test_chunk.py
|   |-- test_cli.py
|   |-- test_embed.py
|   |-- test_generate.py
|   |-- test_graph.py
|   |-- test_retrieve.py
|
|-- data/                     # Generated at runtime (not in git)
|   |-- raw/                  # targets.jsonl, compounds.jsonl
|   |-- graph/                # graph.json, graph.html
|
|-- db/                       # ChromaDB persistent storage (not in git)
|-- mlruns.db                 # MLflow experiment database (not in git)
```

---

## Troubleshooting

### "command not found: dti"

You need to be in the project directory with the venv activated:

```bash
cd /path/to/drug-target-explorer
source .venv/bin/activate
dti --version
```

Or use the launcher:

```bash
./start.sh query
```

### MPS Out of Memory (Apple Silicon)

Add to your `.env`:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

If the 128k model is too large, switch to the 4k model (default):

```yaml
local_model: microsoft/Phi-3-mini-4k-instruct
```

### Slow First Query

The first query downloads and loads 3 models (~5-10s). Subsequent queries in `dti chat` mode reuse cached models and run in ~15-30s.

### Empty or Poor Answers

1. **Check data exists**: `ls data/raw/targets.jsonl data/raw/compounds.jsonl`
2. **Check ChromaDB**: `dti embed` (idempotent, safe to re-run)
3. **Check knowledge graph**: `dti graph`
4. **Increase generation budget**: Set `local_max_new_tokens: 800` in config.yaml
5. **Switch to a larger LLM**: `provider: anthropic` or `provider: ollama`

### EGFR Not Found

EGFR may not appear in ChEMBL keyword search for "kinase". Current dataset targets depend on what ChEMBL returns for the configured `target_families`. Check `data/raw/targets.jsonl` for available targets.

### Ingest Takes Too Long

Default `max_compounds_per_target: 5000` fetches extensively. Reduce it:

```yaml
ingest:
  max_compounds_per_target: 1000
```

### Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## How It Works: End-to-End Example

**Query**: "Tell me about BTK as a drug target"

1. **Classification**: Research query detected ("tell me about") -> skip pre-filter, ensure diverse doc types
2. **Vector search**: BioLORD embeds query, finds 50 candidates in ChromaDB
3. **Graph traversal**: Finds BTK in knowledge graph, adds disease associations (X-linked agammaglobulinemia), GO terms (protein kinase activity, B-cell activation), family membership (Kinase)
4. **Graph batching**: Groups 75 snippets into 3-4 dense documents by category
5. **Cross-encoder reranking**: ms-marco scores all candidates, keeps top-8
6. **Diversity enforcement**: Ensures target_profile + graph_context docs survive (replaces least-needed compound_selectivity docs)
7. **Context packing**: Selects 3 diverse docs for Phi-3-mini (bio context + activity + selectivity)
8. **LLM generation**: Phi-3-mini synthesizes answer using research prompt, citing [Source N]
9. **Output**: Answer with disease context, mechanism, potency values, and citations

---

## License

MIT

---

## Acknowledgments

- **ChEMBL** — EMBL-EBI bioactivity database
- **UniProt** — Universal Protein Resource
- **PubChem** — NCBI chemical database
- **BindingDB** — Binding affinity database
- **BioLORD** — Biomedically-tuned sentence transformer (FremyCompany)
- **Phi-3-mini** — Microsoft's compact language model
- **ChromaDB** — Vector database
- **MLflow** — Experiment tracking
