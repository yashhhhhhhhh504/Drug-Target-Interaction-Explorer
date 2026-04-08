# Drug-Target Interaction Explorer 



---

## Overview

A local, daily-use research tool for querying drug-target interaction data via natural language. The system ingests data from multiple pharmaceutical databases (ChEMBL, UniProt, PubChem, BindingDB, DrugBank), builds a vector index and a knowledge graph, and answers questions with grounded, cited answers pulled from real bioactivity measurements.
**Primary goal:** Research utility — data fidelity, pipeline robustness, and query accuracy over UI aesthetics.  
**v1 scope:** Working RAG pipeline with pretrained cross-encoder. Cross-encoder fine-tuning on selectivity pairs is v2.

---

## Project Structure

```
drug_relation/
├── config.yaml                        # Single source of truth for all tunables
├── pyproject.toml                     # Package definition + CLI entry points
├── data/
│   ├── raw/                           # JSONLines output from ingest stage
│   │   ├── targets.jsonl
│   │   ├── compounds.jsonl
│   │   └── failed_targets.txt         # Targets that failed ingestion for manual retry
│   ├── docs/                          # Markdown files exported for graphifyy
│   └── graph/                         # graphifyy output
│       ├── graph.json
│       ├── graph.html
│       └── GRAPH_REPORT.md
├── db/
│   └── chromadb/                      # Persistent ChromaDB vector store
├── src/
│   └── dti/
│       ├── __init__.py
│       ├── config.py                  # Loads and validates config.yaml
│       ├── sources/                   # Pluggable database adapters
│       │   ├── __init__.py
│       │   ├── base.py                # DataSource ABC
│       │   ├── chembl.py
│       │   ├── uniprot.py
│       │   ├── pubchem.py
│       │   ├── bindingdb.py
│       │   └── drugbank.py
│       ├── ingest.py                  # Orchestrates sources → JSONLines
│       ├── chunk.py                   # Builds 3 document types from JSONLines
│       ├── embed.py                   # Embedding wrapper (BioLORD or OpenAI)
│       ├── store.py                   # ChromaDB read/write
│       ├── graph.py                   # Exports docs to markdown, shells to graphify
│       ├── retrieve.py                # Vector search + graph traversal + cross-encoder
│       ├── generate.py                # LLM generation (Anthropic or Ollama)
│       ├── query.py                   # Public API: retrieve + generate
│       └── cli.py                     # CLI commands
├── tests/
│   ├── fixtures/                      # Static test data (no API calls)
│   ├── test_chunk.py
│   ├── test_retrieve.py
│   ├── test_generate.py
│   └── test_cli.py
└── docs/
    └── superpowers/specs/
```

---

## CLI Commands

Four independently re-runnable stages:

```bash
dti ingest [--sources chembl uniprot pubchem bindingdb drugbank]
           [--families kinase gpcr nuclear_receptor]
# Fetches from selected sources → data/raw/*.jsonl

dti embed
# Reads JSONLines → builds documents → embeds → stores in ChromaDB
# Idempotent: skips already-embedded document IDs

dti graph
# Exports chunked docs to data/docs/*.md → runs graphify → data/graph/
# Produces graph.json (used at query time) + graph.html (interactive viz)

dti query "<natural language question>"
# Vector search + graph traversal → cross-encoder rerank → LLM → answer with citations
```

---

## Configuration

`config.yaml` is the single source of truth. Environment variables override config values at runtime.

```yaml
embedding:
  provider: biolord          # env: EMBEDDING_PROVIDER — "biolord" | "openai"
  biolord_model: FremyCompany/BioLORD-2023
  openai_model: text-embedding-3-large

llm:
  provider: anthropic        # env: LLM_PROVIDER — "anthropic" | "ollama"
  anthropic_model: claude-sonnet-4-6-20251001
  ollama_model: llama3.2
  ollama_base_url: http://localhost:11434

retrieval:
  initial_k: 25              # Candidates pulled from ChromaDB
  rerank_top_k: 5            # Survivors after cross-encoder
  reranker_model: cross-encoder/ms-marco-MiniLM-L-12-v2

data:
  raw_dir: data/raw
  docs_dir: data/docs
  graph_dir: data/graph
  db_dir: db/chromadb

ingest:
  target_families: [kinase, gpcr, nuclear_receptor]
  activity_types: [IC50, Ki, EC50]
  max_compounds_per_target: 500

sources:
  chembl: true
  uniprot: true
  pubchem: false
  bindingdb: false
  drugbank: false            # Requires DRUGBANK_API_KEY env var
```

---

## Data Sources — Pluggable Adapter Interface

Each source implements the `DataSource` ABC:

```python
class DataSource(ABC):
    @abstractmethod
    def fetch_targets(self, families: list[str]) -> list[dict]: ...

    @abstractmethod
    def fetch_activities(self, target_id: str) -> list[dict]: ...
```

**v1 sources:**

| Source | Data provided | Auth required |
|--------|--------------|---------------|
| ChEMBL | Compound-target activity (IC50, Ki, EC50), assay descriptions | None |
| UniProt | Protein function, GO terms, disease associations, domains | None |
| PubChem | Compound properties, bioassay data | None |
| BindingDB | Binding affinities, Ki data | None |
| DrugBank | Approved drug info, drug-drug interactions | API key (`DRUGBANK_API_KEY`) |

All sources write to the same JSONLines schema so downstream stages are source-agnostic:

```jsonc
// targets.jsonl — one JSON object per line
{
  "id": "P00533",
  "source": "uniprot",
  "gene": "EGFR",
  "protein_name": "Epidermal growth factor receptor",
  "organism": "Homo sapiens",
  "function": "...",
  "go_terms": ["kinase activity", "..."],
  "diseases": ["Non-small cell lung carcinoma", "..."],
  "target_family": "kinase"
}

// compounds.jsonl — one JSON object per line
{
  "id": "CHEMBL12345",
  "source": "chembl",
  "name": "Gefitinib",
  "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
  "target_id": "P00533",
  "target_name": "EGFR",
  "activity_type": "IC50",
  "activity_value": 0.033,
  "activity_units": "uM",
  "assay_description": "...",
  "target_family": "kinase"
}
```

---

## Document Types & Chunking

Three document types are built from the JSONLines by `chunk.py`:

### 1. Target Profile (one per protein)
Combines UniProt function, GO terms, disease associations, and a statistical summary of known ligand activity ranges. ~200–400 tokens. Never split.

```
Target: EGFR (Epidermal growth factor receptor)
UniProt: P00533 | Organism: Homo sapiens | Family: kinase
Function: Receptor tyrosine kinase. Binds EGF family ligands...
GO Terms: protein tyrosine kinase activity, signal transduction...
Diseases: Non-small cell lung carcinoma, Inflammatory skin...
Known ligands: 1,247 compounds tested. IC50 range: 0.001–50 uM.
Most potent: Afatinib (IC50 = 0.0003 uM), Osimertinib (IC50 = 0.0008 uM)
```

### 2. Compound Activity (one per compound)
Lists activity across all targets. Split into chunks of 10 targets if a compound has >15 targets. Compound ID and SMILES repeated in every chunk.

```
Compound: Gefitinib (CHEMBL12345)
SMILES: COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1
  vs EGFR (P00533): IC50 = 0.033 uM [ChEMBL assay: CHEMBL123]
  vs HER2 (P04626): IC50 = 3.7 uM [ChEMBL assay: CHEMBL456]
  vs VEGFR2 (P35968): IC50 > 10 uM [ChEMBL assay: CHEMBL789]
```

### 3. Selectivity Comparison (precomputed target pairs)
One document per related target pair within the same family (e.g., EGFR/HER2, CDK4/CDK6). Directly answers the most common selectivity queries.

```
Selectivity comparison: EGFR vs HER2
Compounds with >10x selectivity for EGFR over HER2:
  Gefitinib: EGFR IC50 = 0.033 uM, HER2 IC50 = 3.7 uM (ratio: 112x)
  Erlotinib: EGFR IC50 = 0.02 uM, HER2 IC50 = 4.5 uM (ratio: 225x)
...
```

**Metadata schema** (stored in ChromaDB alongside every chunk):

```python
{
    "doc_type": "target_profile",  # "target_profile" | "compound_activity" | "selectivity_comparison"
    "gene": "EGFR",
    "uniprot_id": "P00533",
    "compound_id": "CHEMBL12345",
    "targets": "EGFR,HER2",        # comma-separated string — ChromaDB requires scalar metadata values
    "target_family": "kinase",
    "sources": "chembl,uniprot",   # comma-separated string — same reason
    "chunk_index": 0,
}
```

---

## Embedding

`embed.py` wraps two providers, selected via config/env:

| Provider | Model | Run location | Requires |
|----------|-------|-------------|---------|
| BioLORD | `FremyCompany/BioLORD-2023` | Local (sentence-transformers) | None |
| OpenAI | `text-embedding-3-large` | Remote | `OPENAI_API_KEY` |

`dti embed` is idempotent — it checks existing ChromaDB document IDs and skips already-embedded documents. Re-run safely after adding new data.

---

## Data Flow (full pipeline)

```
[ChEMBL / UniProt / PubChem / BindingDB / DrugBank]
            ↓
        ingest.py
            ↓
  data/raw/targets.jsonl
  data/raw/compounds.jsonl
            ↓
        chunk.py  →  3 document types
            ↓
        embed.py  →  BioLORD | OpenAI
            ↓
        store.py  →  db/chromadb/
            ↓ (parallel)
        graph.py  →  data/docs/*.md  →  graphify CLI  →  data/graph/
                                                          ├── graph.json
                                                          ├── graph.html
                                                          └── GRAPH_REPORT.md
            ↓ (at query time)
       retrieve.py
         ├── ChromaDB vector search (top-25, optional metadata pre-filter)
         └── Graph traversal from matched entities via graph.json
                  ↓
         Combined candidate set → cross-encoder rerank (top-5)
                  ↓
       generate.py  →  Anthropic | Ollama
                  ↓
       Answer + [Source N] citations printed to stdout
```

---

## Retrieval & Re-ranking

**Step 1 — Metadata pre-filter (optional):** A lightweight keyword heuristic on the query applies ChromaDB `where` filters before embedding similarity runs. Example: query containing "kinase" → `{"target_family": "kinase"}`. Improves both speed and precision.

**Step 2 — Vector search:** Query embedded with same model used at index time (mismatched models produce incorrect results). ChromaDB returns top-25 by cosine similarity.

**Step 3 — Graph traversal:** Entities (gene names, compound IDs) extracted from top-25 results are looked up in `data/graph/graph.json`. Connected nodes (e.g., pathway proteins, related diseases) are fetched and added to the candidate set if not already present.

**Step 4 — Cross-encoder re-ranking:** All candidates scored as (query, document) pairs by `cross-encoder/ms-marco-MiniLM-L-12-v2`. Top-5 survivors passed to LLM.

---

## LLM Generation

System prompt instructs the model to:
- Answer using only the provided context
- Cite sources inline as `[Source N]`
- Include specific activity values (IC50, Ki, EC50) when present
- State explicitly when data is insufficient or conclusions are based on limited assay coverage

**LLM routing:**

```python
if config.llm.provider == "anthropic":
    # Anthropic SDK — claude-sonnet-4-6-20251001
elif config.llm.provider == "ollama":
    # requests to ollama REST API — no extra dependency
    # base_url: http://localhost:11434
```

`LLM_PROVIDER=ollama dti query "..."` switches at runtime without editing config.

**Graceful degradation:** If LLM call fails (network, quota), retrieved documents and their re-rank scores are still printed so retrieval results are never lost.

---

## Graphifyy Integration

`dti graph` runs as an optional fourth stage after `dti embed`:

1. Exports all chunked documents to `data/docs/` as individual `.md` files
2. Shells out to `graphify` CLI on that directory
3. Outputs are written to `data/graph/`:
   - `graph.json` — persistent knowledge graph used at query time for entity traversal
   - `graph.html` — interactive browser visualization
   - `GRAPH_REPORT.md` — core nodes and suggested queries

At query time, `retrieve.py` loads `graph.json` (if present) and traverses from matched entities to surface connected context that pure vector similarity would miss.

`dti graph` is re-runnable after adding new data. If `graph.json` is absent (stage not yet run), the query pipeline falls back to vector-only retrieval without error.

---

## Error Handling

| Stage | Behavior on failure |
|-------|-------------------|
| Ingest | Exponential backoff (3 retries) per API call. One failed target is logged to `data/raw/failed_targets.txt` and skipped — pipeline continues. |
| Embed | Failed batch retried once. Stage is idempotent — safe to re-run. |
| Graph | `graphify` subprocess failure is logged; pipeline continues without graph. |
| Query — retrieval | Errors surface immediately with full traceback. |
| Query — LLM | Retrieved docs + scores printed regardless. LLM error logged, answer omitted. |

---

## Testing

| File | What it tests | API calls |
|------|--------------|-----------|
| `tests/test_chunk.py` | All three document builders with fixture JSONLines | None |
| `tests/test_retrieve.py` | Vector search + graph traversal against ephemeral in-memory ChromaDB (20 known docs) | None |
| `tests/test_generate.py` | Prompt construction and LLM routing with mock LLM | None |
| `tests/test_cli.py` | Smoke tests for all four CLI commands via `click.testing.CliRunner` | None |

ChromaDB tests use a real ephemeral in-memory instance — not mocked — so retrieval logic is tested against actual vector search behavior.

---

## Dependencies (v1)

```toml
[project]
dependencies = [
  "click",                           # CLI
  "pyyaml",                          # config.yaml parsing
  "chembl-webresource-client",       # ChEMBL API
  "requests",                        # UniProt, PubChem, BindingDB, Ollama
  "chromadb",                        # Vector store
  "sentence-transformers",           # BioLORD embedding + cross-encoder
  "openai",                          # OpenAI embedding (optional)
  "anthropic",                       # LLM generation
  "graphifyy",                       # Knowledge graph
]
```

---

## What is Explicitly Out of Scope for v1

- Cross-encoder fine-tuning on selectivity pairs (v2)
- Streamlit frontend (can be added on top of `query.py` API)
- RAGAS-based evaluation harness (v2)
- DrugBank bulk download (API key path only in v1)
- Distributed ingestion or parallelism beyond batch embedding
