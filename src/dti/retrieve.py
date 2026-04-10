"""Retrieval: vector search + graph traversal + cross-encoder re-ranking.

Graph integration strategy:
  - Every query: graph traversal at depth=1 adds connected entity context
  - Relation queries (detected by keyword): depth=2 + relationship sentences
    formatted as proper RAG documents so the cross-encoder scores them well
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import chromadb
from sentence_transformers import CrossEncoder

from .config import Config

logger = logging.getLogger(__name__)

# Module-level cache — reranker is loaded once per process
_reranker_cache: dict[str, "CrossEncoder"] = {}


def _get_reranker(model_name: str) -> "CrossEncoder":
    if model_name not in _reranker_cache:
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


# Keyword → metadata filter for pre-narrowing vector search.
#
# ORDER MATTERS: first match wins. Potency/selectivity keywords are checked
# before family keywords — "10x potency against GSK3B kinase" should hit
# selectivity docs, not the kinase-family filter that would miss them.
#
# Doc types used for selectivity:
#   compound_selectivity      — per-compound ranked potency profiles (directional: most potent first)
#   target_selectivity_index  — per-target index in BOTH directions (T is best / T was outperformed)
#   selectivity_comparison    — pre-computed target-pair docs (only 1 exists: PDE3A vs PDE3B)
_SELECTIVITY_TYPES = {"$in": [
    "compound_selectivity",
    "target_selectivity_index",
    "selectivity_comparison",
]}

_KEYWORD_FILTERS = {
    # "over X" / "vs X" / ratio queries → target_selectivity_index is essential
    "highest selectiv": {"doc_type": _SELECTIVITY_TYPES},
    "over gsk":         {"doc_type": _SELECTIVITY_TYPES},
    "over egfr":        {"doc_type": _SELECTIVITY_TYPES},
    "over itk":         {"doc_type": _SELECTIVITY_TYPES},
    "over map":         {"doc_type": _SELECTIVITY_TYPES},
    "ratio for":        {"doc_type": _SELECTIVITY_TYPES},
    "ratio of":         {"doc_type": _SELECTIVITY_TYPES},
    # General selectivity / potency comparison keywords
    "selectiv":         {"doc_type": _SELECTIVITY_TYPES},
    "10x":              {"doc_type": _SELECTIVITY_TYPES},
    "greater potency":  {"doc_type": _SELECTIVITY_TYPES},
    "more potent":      {"doc_type": {"$in": [
                            "compound_selectivity",
                            "target_selectivity_index",
                            "selectivity_comparison",
                            "compound_activity"]}},
    "most potent":      {"doc_type": {"$in": [
                            "compound_selectivity",
                            "target_selectivity_index",
                            "selectivity_comparison",
                            "compound_activity"]}},
    "fold ":            {"doc_type": _SELECTIVITY_TYPES},
    "potenc":           {"doc_type": {"$in": ["compound_selectivity",
                                              "target_selectivity_index",
                                              "compound_activity"]}},
    # Target family narrowing — only applies when no potency keyword matches first
    "kinase":           {"target_family": "kinase"},
    "gpcr":             {"target_family": "gpcr"},
    "nuclear receptor": {"target_family": "nuclear_receptor"},
    "nuclear_receptor": {"target_family": "nuclear_receptor"},
}

# Keyword patterns that signal a biological context question — skip doc-type filter
# so graph_context docs (disease/GO) are eligible in the vector search
_BIO_CONTEXT_KEYWORDS = {
    "disease", "indication", "therapeutic area", "go term", "gene ontology",
    "biological process", "molecular function", "pathway", "mechanism",
    "signaling", "signalling",
}

# Evidence-quality keywords — skip doc-type filter so evidence_depth docs can surface
_EVIDENCE_KEYWORDS = {
    "evidence", "confidence", "validated", "reliable", "uncertainty", "weak",
    "supported", "assay count", "publication", "discrepan", "disagree",
    "inconsisten", "artifact", "bias", "sparse", "single-assay",
    "how many assays", "how many studies", "evidence base", "evidence depth",
    "audit", "trust", "quality",
}

# Research / biological context keywords — need diverse doc types (target profile,
# graph context, activity data) rather than just selectivity profiles
_RESEARCH_KEYWORDS = {
    "mechanism", "how does", "why", "explain", "role", "function",
    "therapeutic", "treatment", "drug development", "clinical",
    "advance", "candidate", "repurpos", "reposit",
    "disease", "indication", "pathway", "signaling", "signalling",
    "biological", "molecular", "cellular",
    "recommend", "suggest", "which would", "should",
    "compare", "contrast", "difference", "advantage",
    "safety", "toxicity", "side effect", "off-target",
    "resistance", "mutation", "biomarker",
    "combination", "synergy", "polypharmacology",
    "what targets", "what compounds", "what drugs",
    "what are the", "what inhibitors", "which inhibitors",
    "tell me about", "describe", "summarize", "overview",
}


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict
    vector_distance: float
    rerank_score: float


def retrieve(
    query: str,
    cfg: Config,
    collection: chromadb.Collection,
    graph: dict | None = None,
    embedder=None,
) -> list[RetrievedDoc]:
    """Full retrieval pipeline: embed → pre-filter → vector search → graph traversal → rerank.

    When a knowledge graph is available:
      - All queries get depth-1 traversal (direct neighbours)
      - Relation queries ("how is X connected to Y?") get depth-2 traversal
        and the graph context is formatted as rich relationship sentences that
        the cross-encoder can score meaningfully

    Args:
        query: natural language question
        cfg: loaded Config
        collection: ChromaDB collection
        graph: parsed graph.json dict — always pass this when available
        embedder: pre-built Embedder instance (built fresh if None)

    Returns:
        Top-k RetrievedDoc objects sorted by rerank score descending.
    """
    from .embed import build_embedder
    from .graph import is_relation_query, traverse
    from .store import query_collection

    if embedder is None:
        embedder = build_embedder(cfg)

    # Step 1: metadata pre-filter
    where = _detect_filter(query)

    # Step 2: embed query and run vector search
    query_vec = embedder.encode([query])[0]
    candidates = query_collection(
        collection,
        query_vec,
        n_results=cfg.retrieval.initial_k,
        where=where,
    )
    logger.debug("Vector search returned %d candidates (filter=%s)", len(candidates), where)

    # Step 2b: inject selectivity_comparison docs when query names two targets.
    # BioLORD can't distinguish "A over B" from "B over A" at the embedding level,
    # so directional selectivity docs often rank below the initial_k cutoff.
    # We inject them so the cross-encoder can score them properly.
    candidates = _inject_selectivity_comparisons(query, collection, candidates)

    # Step 2c: inject evidence_depth and assay_discrepancy docs for evidence-quality queries.
    # These doc types are semantically different from activity docs, so BioLORD often
    # ranks them below the initial_k cutoff. Injection lets the cross-encoder evaluate.
    candidates = _inject_evidence_docs(query, collection, candidates)

    # Step 3: graph traversal — enrich candidates with knowledge graph context
    if graph:
        relation_query = is_relation_query(query)
        depth = 2 if relation_query else 1
        entity_names = _extract_entities(query, candidates)

        if entity_names:
            graph_snippets = traverse(graph, entity_names, depth=depth)

            _MAX_GRAPH_SNIPPETS = 75
            graph_snippets = graph_snippets[:_MAX_GRAPH_SNIPPETS]

            # Consolidate graph snippets into batched docs by category
            # instead of one doc per sentence — this creates dense, high-value
            # docs that compete well against compound_selectivity in reranking
            existing_texts = {c["text"] for c in candidates}
            graph_candidates = _batch_graph_snippets(graph_snippets, entity_names, existing_texts)
            candidates.extend(graph_candidates)

            logger.info(
                "Graph traversal (depth=%d, relation_query=%s): added %d context docs for entities %s",
                depth, relation_query, len(graph_candidates), entity_names,
            )
        else:
            logger.debug("No entities extracted from query — skipping graph traversal")

    if not candidates:
        return []

    # Step 4: cross-encoder re-ranking (cached — loaded once per process)
    reranker = _get_reranker(cfg.retrieval.reranker_model)
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(candidates, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    top_k = cfg.retrieval.rerank_top_k
    results = [
        RetrievedDoc(
            doc_id=c["id"],
            text=c["text"],
            metadata=c["metadata"],
            vector_distance=c["distance"],
            rerank_score=float(score),
        )
        for c, score in ranked[:top_k]
    ]

    # Step 5: guarantee diversity — ensure critical doc types surface
    q_lower = query.lower()
    if any(kw in q_lower for kw in _EVIDENCE_KEYWORDS):
        results = _ensure_doc_type_diversity(results, ranked, top_k,
                                             required_types=["evidence_depth", "compound_activity"])

    # Research queries need biological context + activity data to synthesize answers
    if any(kw in q_lower for kw in _RESEARCH_KEYWORDS):
        results = _ensure_doc_type_diversity(results, ranked, top_k,
                                             required_types=["target_profile", "graph_context"])

    graph_docs_in_top = sum(1 for r in results if r.metadata.get("doc_type") == "graph_context")
    if graph_docs_in_top:
        logger.info(
            "%d/%d top results came from knowledge graph context",
            graph_docs_in_top, len(results),
        )

    return results


def _batch_graph_snippets(
    snippets: list[str],
    entity_names: list[str],
    existing_texts: set[str],
) -> list[dict]:
    """Consolidate graph snippets into batched docs by category.

    Instead of one candidate per sentence, we group by type (diseases, GO terms,
    drug interactions, family) into dense documents. This gives the cross-encoder
    and LLM richer context per doc, and prevents graph info from being diluted
    across dozens of tiny candidates.
    """
    diseases: list[str] = []
    go_terms: list[str] = []
    activities: list[str] = []
    pathways: list[str] = []
    mechanisms: list[str] = []
    similarity: list[str] = []
    family: list[str] = []
    other: list[str] = []

    for s in snippets:
        if s in existing_texts:
            continue
        existing_texts.add(s)
        s_lower = s.lower()
        if "disease" in s_lower or "therapeutic" in s_lower:
            diseases.append(s)
        elif "biological process" in s_lower or "molecular function" in s_lower or "cellular compartment" in s_lower or "located in" in s_lower:
            go_terms.append(s)
        elif "shared pathway" in s_lower or "same signaling" in s_lower:
            pathways.append(s)
        elif "acts via" in s_lower or "mechanism" in s_lower:
            mechanisms.append(s)
        elif "structurally similar" in s_lower or "tanimoto" in s_lower:
            similarity.append(s)
        elif "binds" in s_lower or "inhibits" in s_lower or "activates" in s_lower or "drug-target" in s_lower:
            activities.append(s)
        elif "family" in s_lower or "member of" in s_lower:
            family.append(s)
        else:
            other.append(s)

    entities_str = ", ".join(entity_names)
    candidates = []

    if diseases:
        candidates.append({
            "id": "graph_diseases",
            "text": (
                f"[Biological Knowledge Graph — Disease Associations for: {entities_str}]\n"
                + "\n".join(f"• {d}" for d in diseases[:20])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    if go_terms:
        candidates.append({
            "id": "graph_biology",
            "text": (
                f"[Biological Knowledge Graph — Biological Functions & Processes for: {entities_str}]\n"
                + "\n".join(f"• {g}" for g in go_terms[:20])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    if pathways:
        candidates.append({
            "id": "graph_pathways",
            "text": (
                f"[Biological Knowledge Graph — Signaling Pathway Connections for: {entities_str}]\n"
                + "\n".join(f"• {p}" for p in pathways[:15])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    if mechanisms:
        candidates.append({
            "id": "graph_mechanisms",
            "text": (
                f"[Biological Knowledge Graph — Mechanism of Action for: {entities_str}]\n"
                + "\n".join(f"• {m}" for m in mechanisms[:15])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    if similarity:
        candidates.append({
            "id": "graph_similarity",
            "text": (
                f"[Biological Knowledge Graph — Structurally Similar Compounds for: {entities_str}]\n"
                + "\n".join(f"• {s}" for s in similarity[:15])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    if activities:
        candidates.append({
            "id": "graph_activities",
            "text": (
                f"[Biological Knowledge Graph — Drug-Target Interactions for: {entities_str}]\n"
                + "\n".join(f"• {a}" for a in activities[:25])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    if family or other:
        combined = family + other
        candidates.append({
            "id": "graph_other",
            "text": (
                f"[Biological Knowledge Graph — Classification & Context for: {entities_str}]\n"
                + "\n".join(f"• {c}" for c in combined[:15])
            ),
            "metadata": {"doc_type": "graph_context", "gene": ",".join(entity_names), "source": "knowledge_graph"},
            "distance": 1.0,
        })

    return candidates


def _format_graph_context(snippet: str, entity_names: list[str]) -> str:
    """Wrap a graph relationship sentence in a context header.

    The header labels the snippet as knowledge-graph evidence so both the
    cross-encoder and the LLM treat it as structured biological context rather
    than raw retrieval text.
    """
    entities_str = ", ".join(entity_names)
    return (
        f"[Biological Knowledge Graph — entities: {entities_str}]\n"
        f"{snippet}"
    )


def _detect_filter(query: str) -> dict | None:
    q_lower = query.lower()
    # Biological-context, evidence, or research queries should not be pre-filtered
    # by doc_type — let everything compete and let the cross-encoder decide.
    if any(kw in q_lower for kw in _BIO_CONTEXT_KEYWORDS):
        return None
    if any(kw in q_lower for kw in _EVIDENCE_KEYWORDS):
        return None
    if any(kw in q_lower for kw in _RESEARCH_KEYWORDS):
        return None
    for keyword, filt in _KEYWORD_FILTERS.items():
        if keyword in q_lower:
            return filt
    return None


def _inject_evidence_docs(
    query: str,
    collection: chromadb.Collection,
    candidates: list[dict],
) -> list[dict]:
    """Inject evidence_depth and assay_discrepancy docs for evidence-quality queries.

    Evidence docs are semantically different from activity docs — BioLORD embeddings
    rank them poorly for questions about "evidence", "confidence", "uncertainty", etc.
    Injection ensures the cross-encoder can evaluate them.
    """
    q_lower = query.lower()
    if not any(kw in q_lower for kw in _EVIDENCE_KEYWORDS):
        return candidates

    existing_ids = {c["id"] for c in candidates}

    try:
        # Fetch evidence_depth and assay_discrepancy docs
        ev_docs = collection.get(
            where={"doc_type": {"$in": ["evidence_depth", "assay_discrepancy"]}},
            include=["documents", "metadatas"],
        )
        if not ev_docs["ids"]:
            return candidates

        # Extract gene names from query
        gene_pattern = re.compile(r"\b[A-Z]{2,8}[0-9]?[A-Z]?\b")
        query_genes = set(gene_pattern.findall(query))

        injected = 0
        for i, doc_id in enumerate(ev_docs["ids"]):
            if doc_id in existing_ids:
                continue
            meta = ev_docs["metadatas"][i]
            doc_genes = set(meta.get("gene", "").split(","))

            # Inject if genes match OR if it's the global discrepancy report
            if (query_genes & doc_genes) or meta.get("doc_type") == "assay_discrepancy":
                candidates.append({
                    "id": doc_id,
                    "text": ev_docs["documents"][i],
                    "metadata": meta,
                    "distance": 0.5,
                })
                injected += 1

        # Also inject compound_activity docs for mentioned genes so the LLM
        # has both potency data AND evidence quality data to compare
        if query_genes:
            try:
                act_docs = collection.get(
                    where={"doc_type": "compound_activity"},
                    include=["documents", "metadatas"],
                )
                act_injected = 0
                for j, act_id in enumerate(act_docs["ids"]):
                    if act_id in existing_ids:
                        continue
                    act_meta = act_docs["metadatas"][j]
                    act_genes = set(act_meta.get("gene", "").split(","))
                    if query_genes & act_genes:
                        candidates.append({
                            "id": act_id,
                            "text": act_docs["documents"][j],
                            "metadata": act_meta,
                            "distance": 0.5,
                        })
                        existing_ids.add(act_id)
                        act_injected += 1
                        if act_injected >= 20:  # cap to keep reranker fast
                            break
                if act_injected:
                    logger.info("Injected %d compound_activity docs for genes %s", act_injected, query_genes)
            except Exception as exc:
                logger.warning("Failed to inject compound_activity docs: %s", exc)

        if injected:
            logger.info("Injected %d evidence docs for query about evidence quality", injected)

    except Exception as exc:
        logger.warning("Failed to inject evidence docs: %s", exc)

    return candidates


def _inject_selectivity_comparisons(
    query: str,
    collection: chromadb.Collection,
    candidates: list[dict],
) -> list[dict]:
    """Inject selectivity_comparison docs when the query mentions two targets.

    BioLORD embeddings can't distinguish directionality ("A over B" ≈ "B over A"),
    so the correct directional doc may not appear in the initial vector search.
    By fetching all selectivity_comparison docs and injecting them, the cross-encoder
    can evaluate both directions and pick the right one.
    """
    # Only if the query looks like a selectivity/comparison question
    q_lower = query.lower()
    sel_keywords = {"selectiv", "over", "vs", "versus", "compared", "ratio", "fold"}
    if not any(kw in q_lower for kw in sel_keywords):
        return candidates

    existing_ids = {c["id"] for c in candidates}

    try:
        # Fetch selectivity_comparison AND target_selectivity_index docs
        sel_docs = collection.get(
            where={"doc_type": {"$in": ["selectivity_comparison", "target_selectivity_index"]}},
            include=["documents", "metadatas"],
        )
        if not sel_docs["ids"]:
            return candidates

        # Filter to docs that mention gene names found in the query
        gene_pattern = re.compile(r"\b[A-Z]{2,8}[0-9]?[A-Z]?\b")
        query_genes = set(gene_pattern.findall(query))

        injected = 0
        for i, doc_id in enumerate(sel_docs["ids"]):
            if doc_id in existing_ids:
                continue
            meta = sel_docs["metadatas"][i]
            doc_genes = set(meta.get("gene", "").split(","))
            # Inject if at least one query gene matches
            if query_genes & doc_genes:
                candidates.append({
                    "id": doc_id,
                    "text": sel_docs["documents"][i],
                    "metadata": meta,
                    "distance": 0.5,  # neutral distance; let cross-encoder decide
                })
                injected += 1

        if injected:
            logger.info("Injected %d selectivity_comparison docs for genes %s", injected, query_genes)

    except Exception as exc:
        logger.warning("Failed to inject selectivity docs: %s", exc)

    return candidates


def _ensure_doc_type_diversity(
    results: list["RetrievedDoc"],
    ranked: list[tuple[dict, float]],
    top_k: int,
    required_types: list[str],
) -> list["RetrievedDoc"]:
    """Ensure at least one doc of each required type appears in results.

    The ms-marco cross-encoder underscores statistical/report docs because it
    was trained on web-search passages. This function promotes the highest-scored
    doc of each missing type into the result set, replacing the lowest-scored
    result of the MOST COMMON type to avoid overwriting previously promoted docs.
    """
    existing_types = {r.metadata.get("doc_type") for r in results}
    result_ids = {r.doc_id for r in results}

    for req_type in required_types:
        if req_type in existing_types:
            continue
        # Find the best-scored candidate of this type in the full ranked list
        for c, score in ranked:
            if c["metadata"].get("doc_type") == req_type and c["id"] not in result_ids:
                promoted = RetrievedDoc(
                    doc_id=c["id"],
                    text=c["text"],
                    metadata=c["metadata"],
                    vector_distance=c["distance"],
                    rerank_score=float(score),
                )
                if len(results) >= top_k:
                    # Replace the lowest-scored result of the most common type
                    # so we don't overwrite previously promoted diversity docs
                    type_counts: dict[str, int] = {}
                    for r in results:
                        dt = r.metadata.get("doc_type", "")
                        type_counts[dt] = type_counts.get(dt, 0) + 1
                    most_common = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
                    # Find the lowest-scored result of the most common type
                    replace_idx = -1
                    replace_score = float("inf")
                    for idx, r in enumerate(results):
                        if r.metadata.get("doc_type") == most_common and r.rerank_score < replace_score:
                            replace_score = r.rerank_score
                            replace_idx = idx
                    if replace_idx >= 0:
                        results[replace_idx] = promoted
                    else:
                        results[-1] = promoted
                else:
                    results.append(promoted)
                result_ids.add(c["id"])
                existing_types.add(req_type)
                logger.info(
                    "Promoted %s doc (score=%.3f) into top results for diversity",
                    req_type, score,
                )
                break

    return results


def _extract_entities(query: str, candidates: list[dict]) -> list[str]:
    """Extract gene/compound names from the query and top candidate metadata."""
    gene_pattern = re.compile(r"\b[A-Z]{2,8}[0-9]?\b")
    entities: set[str] = set(gene_pattern.findall(query))
    for c in candidates[:3]:
        meta = c.get("metadata", {})
        gene = meta.get("gene", "")
        if gene:
            entities.update(g for g in gene.split(",") if g)
    return [e for e in entities if len(e) > 1]
