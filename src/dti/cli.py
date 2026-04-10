"""CLI entry point — four independently re-runnable pipeline stages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers",
                "transformers", "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config.yaml"

# Auto-load .env from the project root so API keys are always available
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
if _ENV_FILE.exists():
    import os
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


def _load_cfg(config_path: str) -> "Config":
    from .config import load
    return load(Path(config_path))


@click.group()
@click.version_option("0.1.0")
def cli() -> None:
    """Drug-Target Interaction Explorer — RAG pipeline over ChEMBL/UniProt and more."""


@cli.command()
@click.option(
    "--config", default=str(_DEFAULT_CONFIG), show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--sources", multiple=True,
    help="Override enabled sources (e.g. --sources chembl --sources uniprot)",
)
@click.option(
    "--families", multiple=True,
    help="Override target families (e.g. --families kinase --families gpcr)",
)
def ingest(config: str, sources: tuple, families: tuple) -> None:
    """Fetch data from enabled sources and write JSONLines to data/raw/."""
    from . import ingest as ingest_mod

    cfg = _load_cfg(config)

    if families:
        cfg.ingest.target_families = list(families)

    click.echo(f"Ingesting targets: {cfg.ingest.target_families}")
    click.echo(f"Sources: {list(sources) or cfg.enabled_sources}")

    try:
        ingest_mod.run(cfg, override_sources=list(sources) or None)
        click.echo("Ingest complete.")
    except Exception as exc:
        click.echo(f"Ingest failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config", default=str(_DEFAULT_CONFIG), show_default=True,
    help="Path to config.yaml",
)
def embed(config: str) -> None:
    """Read JSONLines, embed documents, and store in ChromaDB. Idempotent."""
    from . import embed as embed_mod

    cfg = _load_cfg(config)
    click.echo(f"Embedding provider: {cfg.embedding.provider}")

    try:
        embed_mod.run(cfg)
        click.echo("Embedding complete.")
    except Exception as exc:
        click.echo(f"Embed failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config", default=str(_DEFAULT_CONFIG), show_default=True,
    help="Path to config.yaml",
)
def graph(config: str) -> None:
    """Export documents to markdown and run graphify to build knowledge graph."""
    from . import graph as graph_mod

    cfg = _load_cfg(config)
    click.echo("Building knowledge graph via graphify...")

    try:
        graph_mod.run(cfg)
        html_path = cfg.data.graph_dir / "graph.html"
        if html_path.exists():
            click.echo(f"Interactive graph: {html_path}")
        click.echo("Graph stage complete.")
    except Exception as exc:
        click.echo(f"Graph stage failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("entity_a")
@click.argument("entity_b")
@click.option(
    "--config", default=str(_DEFAULT_CONFIG), show_default=True,
    help="Path to config.yaml",
)
@click.option("--depth", default=3, show_default=True, help="Max hops in graph search")
@click.option("--open-graph", is_flag=True, help="Open graph.html in browser after results")
def relations(entity_a: str, entity_b: str, config: str, depth: int, open_graph: bool) -> None:
    """Show knowledge graph paths between two entities (gene names or compound names).

    Examples:

        dti relations EGFR HER2

        dti relations Gefitinib EGFR

        dti relations EGFR "non-small cell lung"
    """
    import webbrowser

    from .graph import find_relations, load_graph

    cfg = _load_cfg(config)
    graph = load_graph(cfg)

    if graph is None:
        click.echo(
            "No knowledge graph found. Run `dti graph` first to build it.",
            err=True,
        )
        sys.exit(1)

    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))
    click.echo(f"\nKnowledge graph: {node_count} nodes, {edge_count} edges")
    click.echo(f"Finding paths between '{entity_a}' and '{entity_b}' (max depth={depth})...\n")

    paths = find_relations(graph, entity_a, entity_b, max_depth=depth)

    if not paths:
        click.echo(f"No paths found between '{entity_a}' and '{entity_b}' within {depth} hops.")
        click.echo("Try increasing --depth or check that both entities appear in the graph.")
    else:
        click.echo("=" * 60)
        click.echo(f"RELATIONSHIP PATHS: {entity_a} → {entity_b}")
        click.echo("=" * 60)
        for i, path in enumerate(paths, start=1):
            click.echo(f"\nPath {i} ({len(path)} hop{'s' if len(path) != 1 else ''}):")
            for step in path:
                click.echo(f"  → {step}")

    if open_graph:
        html_path = cfg.data.graph_dir / "graph.html"
        if html_path.exists():
            click.echo(f"\nOpening {html_path}")
            webbrowser.open(f"file://{html_path.resolve()}")
        else:
            click.echo("graph.html not found — run `dti graph` to generate it.")


@cli.command()
@click.argument("question")
@click.option(
    "--config", default=str(_DEFAULT_CONFIG), show_default=True,
    help="Path to config.yaml",
)
@click.option("--no-sources", is_flag=True, help="Suppress source documents in output")
def query(question: str, config: str, no_sources: bool) -> None:
    """Ask a natural language question and get a grounded answer with citations.

    QUESTION is the natural language query, e.g.:

        dti query "Which EGFR inhibitors are selective over HER2?"
    """
    from . import query as query_mod

    cfg = _load_cfg(config)

    try:
        result = query_mod.run(question, cfg)

        click.echo("\n" + "=" * 60)
        click.echo("ANSWER")
        click.echo("=" * 60)
        click.echo(result.answer)

        if not no_sources and result.sources:
            click.echo("\n" + "=" * 60)
            click.echo("SUPPORTING EVIDENCE")
            click.echo("=" * 60)
            for i, doc in enumerate(result.sources, start=1):
                click.echo(
                    f"\n[Source {i}] score={doc.rerank_score:.3f} "
                    f"type={doc.metadata.get('doc_type', 'unknown')}"
                )
                click.echo("-" * 40)
                click.echo(doc.text[:600] + ("..." if len(doc.text) > 600 else ""))

    except Exception as exc:
        click.echo(f"Query failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--config", default=str(_DEFAULT_CONFIG), show_default=True,
    help="Path to config.yaml",
)
@click.option("--no-sources", is_flag=True, help="Suppress source documents in output")
def chat(config: str, no_sources: bool) -> None:
    """Interactive query session — models loaded once, fast on every question.

    This is much faster than calling `dti query` in a loop because BioLORD
    and the cross-encoder reranker are loaded into memory once and stay warm.

    Type 'exit' or Ctrl-C to quit.
    """
    from .embed import build_embedder
    from .generate import generate
    from .graph import load_graph
    from .retrieve import retrieve
    from .store import get_collection

    cfg = _load_cfg(config)

    click.echo("")
    click.echo("=" * 60)
    click.echo("  Drug-Target Interaction Explorer — Interactive Mode")
    click.echo("  Type a question, 'help', or 'exit' to quit.")
    click.echo("=" * 60)

    # ── Load everything once ──────────────────────────────────
    click.echo("\nLoading models and database (one-time ~5s)...")
    try:
        embedder = build_embedder(cfg)
        collection = get_collection(cfg)
        graph = load_graph(cfg)
    except Exception as exc:
        click.echo(f"Startup failed: {exc}", err=True)
        sys.exit(1)

    doc_count = collection.count()
    graph_info = f"{len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} edges" if graph else "not built"

    click.echo(f"\nReady.")
    click.echo(f"  ChromaDB: {doc_count:,} documents at db/chromadb/")
    click.echo(f"  Knowledge graph: {graph_info}")
    click.echo(f"  Embedding: BioLORD (FremyCompany/BioLORD-2023)")
    click.echo(f"  Reranker: cross-encoder/ms-marco-MiniLM-L-12-v2")
    _llm_display = {
        "anthropic": cfg.llm.anthropic_model,
        "ollama": cfg.llm.ollama_model,
        "local": cfg.llm.local_model,
    }.get(cfg.llm.provider, cfg.llm.provider)
    click.echo(f"  LLM: {cfg.llm.provider} / {_llm_display}")
    click.echo("")
    click.echo("Hint: try  'dti relations GSK3B ITK'  to explore graph paths.")
    click.echo("")

    # ── Query loop ────────────────────────────────────────────
    while True:
        try:
            question = click.prompt("\033[36mQuestion\033[0m", prompt_suffix="> ")
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            click.echo("Goodbye.")
            break
        if question.lower() == "help":
            click.echo("  Ask anything about drug-target interactions.")
            click.echo("  Examples:")
            click.echo("    Which kinase inhibitors are most selective for GSK3B?")
            click.echo("    How is GSK3B related to MAP3K12?")
            click.echo("    What is the IC50 of staurosporine against GSK3B?")
            continue

        try:
            docs = retrieve(question, cfg, collection, graph=graph, embedder=embedder)
            answer = generate(question, docs, cfg)

            click.echo("\n" + "=" * 60)
            click.echo("ANSWER")
            click.echo("=" * 60)
            click.echo(answer)

            if not no_sources and docs:
                click.echo("\n" + "-" * 60)
                click.echo("SOURCES")
                for i, doc in enumerate(docs, start=1):
                    click.echo(
                        f"  [Source {i}] score={doc.rerank_score:.2f} "
                        f"({doc.metadata.get('doc_type', 'unknown')})"
                    )
            click.echo("")

        except Exception as exc:
            click.echo(f"Query error: {exc}", err=True)
