"""Build a biomedical knowledge graph from drug-target interaction data.

At query time, the knowledge graph is used in two ways:
  1. traverse()      — adds connected entity context to RAG candidates
  2. find_relations() — explicitly finds relationship paths between two entities

Graph is built directly from structured targets/compounds data (no external CLI needed).
An interactive HTML view is also generated using D3.js so users can explore visually.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import mlflow

from .config import Config

logger = logging.getLogger(__name__)

# Relation query keywords — trigger deeper graph traversal (depth=2)
_RELATION_KEYWORDS = {
    "relation", "related", "relationship", "connect", "connected",
    "pathway", "downstream", "upstream", "interact", "interaction",
    "between", "link", "linked", "associate", "associated",
    "mechanism", "signaling", "signalling", "network",
    "disease", "indication", "therapeutic", "treatment",
    "go term", "gene ontology", "biological process", "molecular function",
    "pathway member", "involved in",
}


def run(cfg: Config) -> None:
    """Build knowledge graph from ingested data and save graph.json + graph.html."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="graph"):
        from .ingest import read_jsonl

        targets = read_jsonl(cfg.data.raw_dir / "targets.jsonl")
        compounds = read_jsonl(cfg.data.raw_dir / "compounds.jsonl")

        if not targets and not compounds:
            logger.warning("No data in data/raw/ — run `dti ingest` first.")
            mlflow.log_param("graph_status", "no_data")
            return

        graph_dir = cfg.data.graph_dir
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_data = _build_graph(targets, compounds)
        node_count = len(graph_data["nodes"])
        edge_count = len(graph_data["edges"])
        logger.info("Built knowledge graph: %d nodes, %d edges", node_count, edge_count)

        graph_json_path = graph_dir / "graph.json"
        graph_json_path.write_text(json.dumps(graph_data, indent=2))
        logger.info("Saved graph.json to %s", graph_json_path)

        html_path = graph_dir / "graph.html"
        _write_graph_html(graph_data, html_path)
        logger.info("Saved interactive graph to %s", html_path)

        mlflow.log_metric("graph_nodes", node_count)
        mlflow.log_metric("graph_edges", edge_count)
        mlflow.log_metric("targets_in_graph", len(targets))
        mlflow.log_metric("compounds_in_graph", len(compounds))
        mlflow.log_param("graph_status", "success")
        mlflow.log_artifact(str(graph_json_path))
        mlflow.log_artifact(str(html_path))


def _build_graph(targets: list[dict], compounds: list[dict]) -> dict:
    """Build nodes and edges from structured drug-target data.

    Node types:
      - target     (gene/protein, coloured by target family)
      - compound   (drug/small molecule)
      - disease    (from target disease associations)
      - family     (kinase, gpcr, nuclear_receptor, …)
      - go_term    (biological process, molecular function, cellular component)
      - mechanism  (mechanism of action: binding, functional, covalent, allosteric)

    Edge types:
      - inhibits / activates / binds (compound → target, with IC50/Ki/EC50 as label)
      - associated_with (target → disease)
      - belongs_to_family (target → family)
      - involved_in / has_molecular_function / located_in (target → go_term)
      - shared_pathway (target ↔ target, via co-annotated GO biological processes)
      - structurally_similar (compound ↔ compound, Tanimoto ≥ 0.7)
      - acts_via (compound → mechanism)
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_seen: set[tuple] = set()

    def add_node(nid: str, label: str, node_type: str, description: str = "") -> None:
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": label, "type": node_type, "description": description}

    def add_edge(src: str, tgt: str, label: str) -> None:
        key = (src, tgt, label)
        if key not in edge_seen:
            edge_seen.add(key)
            edges.append({"source": src, "target": tgt, "label": label})

    # ── target nodes ──────────────────────────────────────────────────────────
    # Also collect GO biological process terms per target for pathway co-annotation
    _GO_RELATION = {"P": "involved in", "F": "has molecular function", "C": "located in"}
    target_go_processes: dict[str, set[str]] = {}  # tid → set of GO process labels

    for t in targets:
        tid = f"target:{t['id']}"
        gene = t.get("gene") or t.get("protein_name", "").split()[0] or t["id"]
        protein_name = t.get("protein_name", "")
        family = t.get("target_family", "")
        function_text = t.get("function", "")
        if protein_name and function_text:
            desc = f"{protein_name}. {function_text[:200]}"
        elif function_text:
            desc = function_text[:200]
        else:
            desc = protein_name or ""
        add_node(tid, gene, "target", description=desc)

        if family:
            fid = f"family:{family}"
            add_node(fid, family.replace("_", " ").title(), "family")
            add_edge(tid, fid, "belongs to family")

        for disease in t.get("diseases", [])[:5]:
            if not disease or not disease.strip():
                continue
            did = f"disease:{disease.lower().replace(' ', '_')[:50]}"
            add_node(did, disease.strip(), "disease")
            add_edge(tid, did, "associated with")

        go_procs: set[str] = set()
        for go_term in t.get("go_terms", [])[:15]:
            if not go_term or not go_term.strip():
                continue
            category = go_term[0] if go_term else "?"
            relation = _GO_RELATION.get(category, "annotated with")
            go_label = go_term[2:].strip() if len(go_term) > 2 else go_term
            go_id = f"go:{go_term.lower().replace(' ', '_').replace(':', '_')[:60]}"
            add_node(go_id, go_label, "go_term",
                     description=f"GO term ({category}): {go_label}")
            add_edge(tid, go_id, relation)
            if category == "P":
                go_procs.add(go_label)
        if go_procs:
            target_go_processes[tid] = go_procs

    # ── target-target pathway edges (GO biological process co-annotation) ────
    # Two targets sharing ≥2 biological processes are in the same signaling
    # cascade — connect them so the graph can answer pathway questions.
    target_ids = list(target_go_processes.keys())
    pathway_edges = 0
    for i in range(len(target_ids)):
        for j in range(i + 1, len(target_ids)):
            tid_a, tid_b = target_ids[i], target_ids[j]
            shared = target_go_processes[tid_a] & target_go_processes[tid_b]
            if len(shared) >= 2:
                top_processes = sorted(shared)[:3]
                label = f"shared pathway: {', '.join(top_processes)}"
                add_edge(tid_a, tid_b, label)
                pathway_edges += 1
    if pathway_edges:
        logger.info("Added %d target-target pathway edges via GO co-annotation", pathway_edges)

    # ── compound + activity edges ─────────────────────────────────────────────
    compound_by_name: dict[str, dict] = {}
    compound_activities: dict[str, list[dict]] = defaultdict(list)

    for c in compounds:
        name = c.get("name") or c["id"]
        cid = f"compound:{name.lower().replace(' ', '_')[:60]}"
        if name not in compound_by_name:
            compound_by_name[name] = {"cid": cid, "record": c}
        compound_activities[name].append(c)

    for name, info in compound_by_name.items():
        cid = info["cid"]
        rec = info["record"]
        smiles = rec.get("smiles", "")
        add_node(cid, name, "compound", description=f"SMILES: {smiles[:80]}" if smiles else "")

    for name, acts in compound_activities.items():
        cid = compound_by_name[name]["cid"]
        for act in acts:
            target_id = act.get("target_id", "")
            if not target_id:
                continue
            tid = f"target:{target_id}"
            if tid not in nodes:
                continue
            atype = act.get("activity_type", "binds")
            aval = act.get("activity_value", "")
            units = act.get("activity_units", "nM")
            edge_label = f"{atype}={aval}{units}" if aval else atype.lower()
            add_edge(cid, tid, edge_label)

    # ── mechanism-of-action nodes ─────────────────────────────────────────────
    # Extract MoA from assay_type codes and assay_description keywords.
    # Creates mechanism nodes and connects compounds to them.
    _MOA_KEYWORDS = {
        "covalent": ("covalent", "Covalent inhibitor — forms irreversible bond with target"),
        "allosteric": ("allosteric", "Allosteric modulator — binds outside the active site"),
        "competitive": ("competitive", "Competitive inhibitor — competes with substrate at active site"),
        "reversible": ("reversible", "Reversible inhibitor — non-covalent, dissociates from target"),
        "irreversible": ("irreversible", "Irreversible inhibitor — permanently modifies target"),
        "agonist": ("agonist", "Agonist — activates the target receptor"),
        "antagonist": ("antagonist", "Antagonist — blocks receptor activation"),
        "inverse agonist": ("inverse agonist", "Inverse agonist — produces opposite effect to agonist"),
        "partial agonist": ("partial agonist", "Partial agonist — partially activates the receptor"),
    }
    _ASSAY_TYPE_MOA = {
        "B": ("binding", "Binding assay — measures direct target binding affinity"),
        "F": ("functional", "Functional assay — measures biological/cellular effect"),
        "A": ("ADME", "ADME assay — absorption, distribution, metabolism, excretion"),
        "T": ("toxicity", "Toxicity assay — measures cytotoxicity or off-target effects"),
    }

    compound_moa_seen: set[tuple[str, str]] = set()
    for name, acts in compound_activities.items():
        cid = compound_by_name[name]["cid"]
        for act in acts:
            moa_found: list[tuple[str, str]] = []

            # Check assay_type code first
            assay_type = act.get("assay_type", "")
            if assay_type in _ASSAY_TYPE_MOA:
                moa_found.append(_ASSAY_TYPE_MOA[assay_type])

            # Search assay_description for mechanism keywords
            desc_lower = (act.get("assay_description") or "").lower()
            for keyword, (moa_label, moa_desc) in _MOA_KEYWORDS.items():
                if keyword in desc_lower:
                    moa_found.append((moa_label, moa_desc))

            for moa_label, moa_desc in moa_found:
                moa_id = f"mechanism:{moa_label.replace(' ', '_')}"
                add_node(moa_id, moa_label.title(), "mechanism", description=moa_desc)
                edge_key = (cid, moa_id)
                if edge_key not in compound_moa_seen:
                    compound_moa_seen.add(edge_key)
                    add_edge(cid, moa_id, "acts via")

    moa_count = len([n for n in nodes.values() if n["type"] == "mechanism"])
    if moa_count:
        logger.info("Added %d mechanism-of-action nodes with %d compound edges",
                     moa_count, len(compound_moa_seen))

    # ── compound similarity edges (Tanimoto on Morgan fingerprints) ───────────
    # Connect structurally similar compounds so the graph can answer
    # "Which compounds are similar to X?" and suggest analogs.
    similarity_edges = _build_compound_similarity_edges(compound_by_name, add_node, add_edge)
    if similarity_edges:
        logger.info("Added %d compound-compound structural similarity edges", similarity_edges)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def _build_compound_similarity_edges(
    compound_by_name: dict[str, dict],
    add_node,
    add_edge,
) -> int:
    """Connect compounds with Tanimoto similarity >= 0.7 on Morgan fingerprints.

    Uses RDKit Morgan fingerprints (radius=2, 1024 bits). Only considers
    compounds with valid SMILES. Caps at 5 similarity edges per compound
    to avoid cluttering the graph.
    """
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        logger.warning("rdkit not installed — skipping compound similarity edges. "
                       "Install with: pip install rdkit")
        return 0

    # Build fingerprints for compounds with valid SMILES
    fp_data: list[tuple[str, str, any]] = []  # (cid, name, fingerprint)
    for name, info in compound_by_name.items():
        smiles = info["record"].get("smiles", "")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_data.append((info["cid"], name, fp))

    if len(fp_data) < 2:
        return 0

    # Pairwise Tanimoto — cap comparisons for large datasets
    # For >2000 compounds, sample to keep runtime reasonable
    max_compounds = 2000
    if len(fp_data) > max_compounds:
        import random
        random.seed(42)
        fp_data = random.sample(fp_data, max_compounds)

    edge_count = 0
    edges_per_compound: dict[str, int] = defaultdict(int)
    _MAX_EDGES_PER_COMPOUND = 5
    _MIN_TANIMOTO = 0.7

    for i in range(len(fp_data)):
        cid_a, name_a, fp_a = fp_data[i]
        if edges_per_compound[cid_a] >= _MAX_EDGES_PER_COMPOUND:
            continue
        for j in range(i + 1, len(fp_data)):
            cid_b, name_b, fp_b = fp_data[j]
            if edges_per_compound[cid_b] >= _MAX_EDGES_PER_COMPOUND:
                continue

            sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
            if sim >= _MIN_TANIMOTO:
                add_edge(cid_a, cid_b, f"structurally similar (Tanimoto={sim:.2f})")
                edge_count += 1
                edges_per_compound[cid_a] += 1
                edges_per_compound[cid_b] += 1

    return edge_count


def _write_graph_html(graph_data: dict, output_path: Path) -> None:
    """Write a self-contained D3.js force-directed graph HTML file."""
    nodes_json = json.dumps(graph_data["nodes"])
    edges_json = json.dumps(graph_data["edges"])

    # Color palette per node type
    color_map = {
        "target": "#4e79a7",
        "compound": "#f28e2b",
        "disease": "#e15759",
        "family": "#76b7b2",
        "go_term": "#59a14f",
        "mechanism": "#b07aa1",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Drug-Target Interaction Knowledge Graph</title>
<style>
  body {{ margin: 0; background: #1a1a2e; font-family: Arial, sans-serif; color: #eee; }}
  #info {{ position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.7);
           padding: 10px 16px; border-radius: 8px; max-width: 320px; font-size: 13px; z-index: 10; }}
  #info h3 {{ margin: 0 0 6px 0; font-size: 15px; }}
  #info p  {{ margin: 2px 0; }}
  .legend {{ position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7);
             padding: 8px 14px; border-radius: 8px; font-size: 12px; z-index: 10; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  svg {{ width: 100vw; height: 100vh; }}
  .node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
  .node text {{ fill: #ddd; font-size: 10px; pointer-events: none; }}
  .link {{ stroke: #555; stroke-opacity: 0.6; }}
  .link-label {{ fill: #aaa; font-size: 8px; }}
</style>
</head>
<body>
<div id="info">
  <h3>Drug-Target Interaction Graph</h3>
  <p id="node-count">Nodes: {len(graph_data["nodes"])}</p>
  <p id="edge-count">Edges: {len(graph_data["edges"])}</p>
  <p id="selected" style="margin-top:8px; color:#f28e2b;"></p>
</div>
<div class="legend">
  <div class="legend-item"><span class="dot" style="background:#4e79a7"></span> Target (protein)</div>
  <div class="legend-item"><span class="dot" style="background:#f28e2b"></span> Compound (drug)</div>
  <div class="legend-item"><span class="dot" style="background:#e15759"></span> Disease</div>
  <div class="legend-item"><span class="dot" style="background:#76b7b2"></span> Target family</div>
  <div class="legend-item"><span class="dot" style="background:#59a14f"></span> GO term (biology)</div>
  <div class="legend-item"><span class="dot" style="background:#b07aa1"></span> Mechanism of action</div>
</div>
<svg></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const nodes = {nodes_json};
const links = {edges_json}.map(e => ({{...e, source: e.source, target: e.target}}));
const colorMap = {json.dumps(color_map)};
const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("svg");
const g = svg.append("g");

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-120))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide(18));

const link = g.append("g").selectAll("line")
  .data(links).join("line").attr("class","link").attr("stroke-width", 1);

const node = g.append("g").selectAll("g")
  .data(nodes).join("g").attr("class","node")
  .call(d3.drag()
    .on("start", (e,d) => {{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on("drag",  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on("end",   (e,d) => {{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}))
  .on("click", (e, d) => {{
    d3.select("#selected").text(d.label + (d.description ? ": " + d.description.slice(0,80) : ""));
  }});

node.append("circle")
  .attr("r", d => d.type==="target" ? 10 : d.type==="compound" ? 7 : d.type==="family" ? 12 : 6)
  .attr("fill", d => colorMap[d.type] || "#aaa");

node.append("text").attr("dx", 12).attr("dy", 4).text(d => d.label.slice(0, 20));

svg.call(d3.zoom().scaleExtent([0.1,6]).on("zoom", e => g.attr("transform", e.transform)));

sim.on("tick", () => {{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
      .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform", d=>`translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>"""
    output_path.write_text(html)



def load_graph(cfg: Config) -> dict | None:
    """Load graph.json if it exists. Returns None if graph stage has not been run."""
    graph_path = cfg.data.graph_dir / "graph.json"
    if not graph_path.exists():
        return None
    try:
        return json.loads(graph_path.read_text())
    except Exception as exc:
        logger.warning("Failed to load graph.json: %s", exc)
        return None


def is_relation_query(query: str) -> bool:
    """Return True if the query is asking about relationships between entities."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _RELATION_KEYWORDS)


def traverse(graph: dict, entity_names: list[str], depth: int = 1) -> list[str]:
    """Return relationship sentences for entities connected to the given names.

    Returns rich sentences like:
      "EGFR [inhibits] Non-small cell lung carcinoma: disease association"
      "Gefitinib [targets] EGFR: IC50 = 0.033 uM"

    instead of bare node labels — these score much better in cross-encoder reranking.
    """
    if not graph:
        return []

    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    # Build adjacency with edge labels: node_id → list of (neighbour_id, relation_label)
    adjacency: dict[str, list[tuple[str, str]]] = {nid: [] for nid in nodes}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        rel = edge.get("label", edge.get("type", "relates to"))
        if src in adjacency:
            adjacency[src].append((tgt, rel))
        if tgt in adjacency:
            adjacency[tgt].append((src, rel))

    # Find seed nodes matching entity names (case-insensitive substring match)
    seed_ids: set[str] = set()
    entity_lower = [e.lower() for e in entity_names]
    for nid, node in nodes.items():
        label = node.get("label", "").lower()
        if any(e in label for e in entity_lower):
            seed_ids.add(nid)

    if not seed_ids:
        return []

    # BFS up to `depth` hops, collecting relationship sentences
    visited: set[str] = set(seed_ids)
    frontier: set[str] = set(seed_ids)
    sentences: list[str] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for nid in frontier:
            src_node = nodes.get(nid, {})
            src_label = src_node.get("label", nid)
            src_desc = src_node.get("description", "")

            for neighbour_id, relation in adjacency.get(nid, []):
                tgt_node = nodes.get(neighbour_id, {})
                tgt_label = tgt_node.get("label", neighbour_id)
                tgt_desc = tgt_node.get("description", "")

                # Build a biologically descriptive sentence the cross-encoder can score
                sentence = _bio_sentence(src_label, relation, tgt_label,
                                         tgt_node.get("type", ""), tgt_desc)
                sentences.append(sentence)

                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    next_frontier.add(neighbour_id)

        frontier = next_frontier

    return sentences


def _bio_sentence(src: str, relation: str, tgt: str, tgt_type: str, tgt_desc: str) -> str:
    """Produce a natural-language biological sentence from a graph edge.

    The phrasing is chosen so the cross-encoder can score it against a
    biomedical question — plain-English is better than bracket notation.
    """
    if tgt_type == "disease":
        return (
            f"{src} is associated with the disease {tgt}. "
            f"This target-disease relationship indicates {src} may be a therapeutic target "
            f"for {tgt}."
        )
    if tgt_type == "go_term":
        if "molecular function" in relation:
            return f"{src} has molecular function: {tgt}."
        if "located in" in relation:
            return f"{src} is located in the {tgt} cellular compartment."
        return f"{src} is involved in the biological process: {tgt}."
    if tgt_type == "family":
        return f"{src} is a member of the {tgt} protein family."
    if tgt_type == "mechanism":
        return (
            f"{src} acts via {tgt} mechanism. "
            f"{tgt_desc}" if tgt_desc else f"{src} acts via {tgt} mechanism."
        )
    if tgt_type == "compound":
        # Compound-compound similarity edge
        if "structurally similar" in relation:
            return (
                f"{src} and {tgt} are structurally similar compounds ({relation}). "
                f"They may share pharmacological properties and target profiles."
            )
        return f"{tgt} {relation} {src}."
    if tgt_type == "target":
        # Compound → target activity edge: relation looks like "IC50=0.5nM"
        if "=" in relation:
            return (
                f"{src} binds {tgt} with {relation}. "
                f"This represents a direct drug-target interaction."
            )
        # Target-target shared pathway edge
        if "shared pathway" in relation:
            return (
                f"{src} and {tgt} participate in the same signaling pathway ({relation}). "
                f"Compounds targeting {src} may also affect {tgt}-dependent signaling."
            )
        return f"{src} {relation} {tgt}."
    # Generic fallback
    sentence = f"{src} {relation} {tgt}"
    if tgt_desc:
        sentence += f" — {tgt_desc[:120]}"
    return sentence + "."


def find_relations(
    graph: dict,
    entity_a: str,
    entity_b: str,
    max_depth: int = 3,
) -> list[list[str]]:
    """Find all paths between two entities in the knowledge graph.

    Returns a list of paths, each path being a list of relationship sentences
    describing the connection:
      ["EGFR [inhibits] Erlotinib: IC50=0.02uM", "Erlotinib [targets] VEGFR2: IC50=1.2uM"]

    Returns [] if no path found within max_depth hops.
    """
    if not graph:
        return []

    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])

    adjacency: dict[str, list[tuple[str, str]]] = {nid: [] for nid in nodes}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        rel = edge.get("label", edge.get("type", "relates to"))
        if src in adjacency:
            adjacency[src].append((tgt, rel))
        if tgt in adjacency:
            adjacency[tgt].append((src, rel))

    def find_node_ids(name: str) -> set[str]:
        name_lower = name.lower()
        return {
            nid for nid, node in nodes.items()
            if name_lower in node.get("label", "").lower()
        }

    seeds_a = find_node_ids(entity_a)
    seeds_b = find_node_ids(entity_b)

    if not seeds_a or not seeds_b:
        return []

    # BFS from each seed in A, find paths to any node in B
    all_paths: list[list[str]] = []

    for start_id in seeds_a:
        # BFS: queue of (current_node_id, path_of_sentences, visited_set)
        queue: list[tuple[str, list[str], set[str]]] = [
            (start_id, [], {start_id})
        ]
        while queue:
            current_id, path, path_visited = queue.pop(0)
            if len(path) > max_depth:
                continue

            current_node = nodes.get(current_id, {})
            current_label = current_node.get("label", current_id)

            for neighbour_id, relation in adjacency.get(current_id, []):
                if neighbour_id in path_visited:
                    continue

                tgt_node = nodes.get(neighbour_id, {})
                tgt_label = tgt_node.get("label", neighbour_id)
                tgt_desc = tgt_node.get("description", "")

                sentence = _bio_sentence(current_label, relation, tgt_label,
                                         tgt_node.get("type", ""), tgt_desc)

                new_path = path + [sentence]

                if neighbour_id in seeds_b:
                    all_paths.append(new_path)
                else:
                    queue.append((neighbour_id, new_path, path_visited | {neighbour_id}))

        if all_paths:
            break  # Stop after first seed that finds paths

    # Return shortest paths first
    return sorted(all_paths, key=len)[:5]
