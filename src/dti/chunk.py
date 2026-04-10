"""Build the three document types from raw JSONLines records."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class Document:
    text: str
    metadata: dict
    doc_id: str


def build_target_profiles(targets: list[dict], compounds: list[dict]) -> list[Document]:
    """One document per target, combining UniProt annotations + ligand activity summary."""
    activities_by_target: dict[str, list[dict]] = defaultdict(list)
    for c in compounds:
        activities_by_target[c["target_id"]].append(c)

    docs = []
    for t in targets:
        tid = t["id"]
        acts = activities_by_target.get(tid, [])

        ligand_summary = _summarize_activities(acts)

        go_str = ", ".join(t.get("go_terms", [])[:10]) or "None"
        disease_str = ", ".join(t.get("diseases", [])[:5]) or "None reported"

        text = (
            f"Target: {t.get('protein_name', t['id'])} ({t.get('gene', '')})\n"
            f"ID: {tid} | Source: {t['source']} | Organism: {t.get('organism', 'Homo sapiens')}\n"
            f"Family: {t.get('target_family', '')}\n"
            f"Function: {t.get('function', 'Not available')}\n"
            f"GO Terms: {go_str}\n"
            f"Diseases: {disease_str}\n"
            f"{ligand_summary}"
        ).strip()

        docs.append(Document(
            text=text,
            doc_id=f"target_{tid}",
            metadata={
                "doc_type": "target_profile",
                "gene": t.get("gene", ""),
                "uniprot_id": tid if t["source"] == "uniprot" else "",
                "compound_id": "",
                "targets": t.get("gene", tid),
                "target_family": t.get("target_family", ""),
                "sources": t["source"],
                "chunk_index": 0,
            },
        ))
    return docs


def build_compound_activity_docs(compounds: list[dict], targets: list[dict]) -> list[Document]:
    """One document per compound, listing activity across all tested targets.

    Compounds with >15 targets are split into chunks of 10, with compound ID
    and SMILES repeated in every chunk.
    """
    target_map = {t["id"]: t for t in targets}

    # Group activities by compound
    by_compound: dict[str, list[dict]] = defaultdict(list)
    for c in compounds:
        by_compound[c["id"]].append(c)

    docs = []
    for compound_id, acts in by_compound.items():
        if not acts:
            continue
        first = acts[0]
        smiles = first.get("smiles", "")
        name = first.get("name", compound_id)
        all_target_names = {a["target_id"] for a in acts}

        # Collect target family from any activity or target map
        family = first.get("target_family", "")
        if not family:
            for a in acts:
                t = target_map.get(a["target_id"])
                if t:
                    family = t.get("target_family", "")
                    break

        # Split into chunks of 10 if >15 targets
        chunks = _chunk_list(acts, 10) if len(acts) > 15 else [acts]

        for chunk_idx, chunk in enumerate(chunks):
            lines = [
                f"Compound: {name} ({compound_id})",
                f"SMILES: {smiles}",
            ]
            for a in chunk:
                tname = a.get("target_name") or a["target_id"]
                t_rec = target_map.get(a["target_id"])
                gene_sym = t_rec.get("gene", "") if t_rec else ""
                # Include gene symbol so embeddings match abbreviation queries (e.g. "GSK3B")
                tname_display = f"{tname} ({gene_sym})" if gene_sym and gene_sym not in tname else tname
                lines.append(
                    f"  vs {tname_display}: "
                    f"{a['activity_type']} = {a['activity_value']} {a.get('activity_units', 'nM')}"
                )

            docs.append(Document(
                text="\n".join(lines),
                doc_id=f"compound_{compound_id}_chunk{chunk_idx}",
                metadata={
                    "doc_type": "compound_activity",
                    "gene": "",
                    "uniprot_id": "",
                    "compound_id": compound_id,
                    "targets": ",".join(a["target_id"] for a in chunk),
                    "target_family": family,
                    "sources": first.get("source", ""),
                    "chunk_index": chunk_idx,
                },
            ))

    return docs


def build_selectivity_docs(compounds: list[dict], targets: list[dict]) -> list[Document]:
    """Precomputed selectivity comparisons for target pairs within the same family.

    Generates TWO documents per target pair — one for each direction:
      - "Selectivity for PDE3A over PDE3B" (compounds where PDE3A is more potent)
      - "Selectivity for PDE3B over PDE3A" (compounds where PDE3B is more potent)

    Each doc lists ALL compounds tested against both targets (not just ≥10x),
    sorted by selectivity ratio descending, with the top compound explicitly
    labelled as "HIGHEST SELECTIVITY RATIO" so the LLM can answer
    "which compound has the greatest selectivity?" directly.
    """
    # Group targets by family
    families: dict[str, list[dict]] = defaultdict(list)
    for t in targets:
        fam = t.get("target_family", "")
        if fam:
            families[fam].append(t)

    # Index activities: compound_id → target_id → best activity value
    act_index: dict[str, dict[str, tuple[float, str]]] = defaultdict(dict)
    compound_names: dict[str, str] = {}
    for c in compounds:
        cid = c["id"]
        tid = c["target_id"]
        val = c.get("activity_value")
        atype = c.get("activity_type", "IC50")
        if val is None:
            continue
        if cid not in compound_names:
            compound_names[cid] = c.get("name", cid)
        existing = act_index[cid].get(tid)
        if existing is None or val < existing[0]:
            act_index[cid][tid] = (float(val), atype)

    docs = []
    for family, fam_targets in families.items():
        pairs = _target_pairs(fam_targets, max_pairs=50)
        for t_a, t_b in pairs:
            # Generate BOTH directions for this pair
            for t_fav, t_other in [(t_a, t_b), (t_b, t_a)]:
                doc = _build_one_direction_selectivity_doc(
                    t_fav, t_other, act_index, compound_names, family,
                )
                if doc:
                    docs.append(doc)

    return docs


def _build_one_direction_selectivity_doc(
    t_fav: dict,
    t_other: dict,
    act_index: dict,
    compound_names: dict,
    family: str,
) -> Document | None:
    """Build one selectivity doc: compounds selective FOR t_fav OVER t_other.

    A compound is "selective for t_fav" when val_fav < val_other (lower = more potent).
    Ratio = val_other / val_fav (higher ratio = greater selectivity for t_fav).
    """
    id_fav, id_other = t_fav["id"], t_other["id"]
    gene_fav = t_fav.get("gene", id_fav)
    gene_other = t_other.get("gene", id_other)

    # Collect all compounds tested against both targets
    rows: list[tuple] = []  # (cid, cname, val_fav, atype_fav, val_other, atype_other, ratio)
    for cid, target_acts in act_index.items():
        v_fav = target_acts.get(id_fav)
        v_other = target_acts.get(id_other)
        if v_fav and v_other and v_fav[0] > 0 and v_other[0] > 0:
            ratio = v_other[0] / v_fav[0]
            if ratio > 1.0:  # only include if actually selective for t_fav
                cname = compound_names.get(cid, cid)
                rows.append((cid, cname, v_fav[0], v_fav[1], v_other[0], v_other[1], ratio))

    if not rows:
        return None

    rows.sort(key=lambda x: x[6], reverse=True)  # highest selectivity ratio first

    # Header with clear direction labelling
    lines = [
        f"Selectivity ranking: compounds selective FOR {gene_fav} OVER {gene_other} (family: {family})",
        f"[Direction: lower {gene_fav} value = more potent at {gene_fav} = selective FOR {gene_fav}]",
        f"[Ratio = {gene_other}_value / {gene_fav}_value — higher ratio = greater selectivity for {gene_fav}]",
        f"Total compounds tested against both targets: {len(rows)}",
        "",
    ]

    for i, (cid, cname, val_fav, atype_fav, val_other, atype_other, ratio) in enumerate(rows[:25]):
        assay_note = ""
        if atype_fav != atype_other:
            assay_note = f" [CAUTION: comparing {atype_fav} vs {atype_other} — different assay types]"

        label = "  ★ HIGHEST SELECTIVITY RATIO" if i == 0 else ""
        lines.append(
            f"  {i + 1}. {cname} ({cid}): "
            f"{gene_fav} {atype_fav}={val_fav:.4g} nM [LOWER=MORE POTENT] "
            f"vs {gene_other} {atype_other}={val_other:.4g} nM [HIGHER=LESS POTENT] "
            f"→ {ratio:.1f}x selectivity for {gene_fav}{assay_note}{label}"
        )

    # Highlight ≥10x selective compounds
    high_selective = [r for r in rows if r[6] >= 10]
    if high_selective:
        lines.append(f"\n{len(high_selective)} compound(s) with ≥10x selectivity for {gene_fav} over {gene_other}.")
    else:
        lines.append(f"\nNo compounds with ≥10x selectivity for {gene_fav} over {gene_other}.")

    pair_id = f"selectivity_{gene_fav}_over_{gene_other}"
    return Document(
        text="\n".join(lines),
        doc_id=pair_id,
        metadata={
            "doc_type": "selectivity_comparison",
            "gene": f"{gene_fav},{gene_other}",
            "uniprot_id": "",
            "compound_id": "",
            "targets": f"{id_fav},{id_other}",
            "target_family": family,
            "sources": "computed",
            "chunk_index": 0,
        },
    )


def build_compound_selectivity_docs(compounds: list[dict], targets: list[dict]) -> list[Document]:
    """One document per compound tested against ≥2 targets.

    Shows ranked potency across all tested targets with explicit selectivity ratios,
    so the LLM can directly answer "which compounds are ≥10x more potent against X vs Y?"
    without needing to compute anything.
    """
    target_map = {t["id"]: t for t in targets}

    # Collect best (lowest) activity value per compound-target pair
    best: dict[str, dict[str, tuple]] = defaultdict(dict)
    compound_meta: dict[str, dict] = {}

    for c in compounds:
        cid = c["id"]
        tid = c["target_id"]
        val = c.get("activity_value")
        if val is None:
            continue

        if cid not in compound_meta:
            compound_meta[cid] = {
                "name": c.get("name", cid),
                "smiles": c.get("smiles", ""),
                "source": c.get("source", ""),
            }

        t_rec = target_map.get(tid)
        gene = t_rec.get("gene", "") if t_rec else ""
        tname = c.get("target_name") or tid
        family = c.get("target_family") or (t_rec.get("target_family", "") if t_rec else "")
        units = c.get("activity_units", "nM")
        atype = c.get("activity_type", "IC50")

        existing = best[cid].get(tid)
        if existing is None or float(val) < existing[0]:
            best[cid][tid] = (float(val), atype, tname, gene, family, units)

    docs = []
    for cid, target_acts in best.items():
        if len(target_acts) < 2:
            continue  # Only meaningful for multi-target compounds

        meta = compound_meta.get(cid, {})
        name = meta.get("name", cid)
        smiles = meta.get("smiles", "")

        # Sort by ascending activity value (lowest = most potent)
        sorted_acts = sorted(target_acts.items(), key=lambda x: x[1][0])

        best_tid, (best_val, best_atype, best_tname, best_gene, _, best_units) = sorted_acts[0]
        best_label = best_gene or best_tname

        # Lead with a rich summary sentence so BioLORD embeds the semantic
        # "X is selective for Y" close to queries asking about Y's selective compounds.
        second_tid, (second_val, second_atype, second_tname, second_gene, _, second_units) = sorted_acts[1]
        second_label = second_gene or second_tname
        top_ratio = second_val / best_val if best_val > 0 else 0
        summary = (
            f"{name} ({cid}) is most potent against {best_label} "
            f"({best_atype}={best_val:.4g} {best_units}), "
            f"{top_ratio:.0f}x more potent than {second_label} "
            f"({second_atype}={second_val:.4g} {second_units})."
        )

        lines = [
            summary,
            f"Compound selectivity profile: {name} ({cid})",
            f"SMILES: {smiles}",
            f"Tested against {len(sorted_acts)} targets (ranked most-to-least potent):",
        ]

        selectivity_highlights: list[str] = []
        all_genes: list[str] = []
        all_tids: list[str] = []
        family_set: set[str] = set()

        for i, (tid_i, (val, atype, tname, gene, family, units)) in enumerate(sorted_acts):
            label = gene or tname
            display = f"{tname} ({gene})" if gene and gene not in tname else tname
            if family:
                family_set.add(family)
            all_genes.append(gene or tname)
            all_tids.append(tid_i)

            if i == 0:
                lines.append(f"  1. {display}: {atype} = {val:.4g} {units}  [most potent]")
            else:
                if best_val > 0:
                    ratio = val / best_val
                    ratio_str = f"  [{ratio:.1f}x less potent than {best_label}]"
                    if ratio >= 10:
                        selectivity_highlights.append(
                            f"{best_label} over {label}: "
                            f"{best_atype}={best_val:.4g} {best_units} vs "
                            f"{atype}={val:.4g} {units}  (ratio={ratio:.0f}x)"
                        )
                else:
                    ratio_str = ""
                lines.append(f"  {i + 1}. {display}: {atype} = {val:.4g} {units}{ratio_str}")

        if selectivity_highlights:
            lines.append("")
            lines.append("Selectivity highlights (≥10x potency difference):")
            for h in selectivity_highlights:
                lines.append(f"  - {h}")
        else:
            lines.append("")
            lines.append("No ≥10x selectivity found between any tested target pair.")

        docs.append(Document(
            text="\n".join(lines),
            doc_id=f"selectivity_profile_{cid}",
            metadata={
                "doc_type": "compound_selectivity",
                "gene": ",".join(all_genes),
                "uniprot_id": "",
                "compound_id": cid,
                "targets": ",".join(all_tids),
                "target_family": ",".join(family_set),
                "sources": meta.get("source", ""),
                "chunk_index": 0,
            },
        ))

    return docs


def build_target_selectivity_index_docs(compounds: list[dict], targets: list[dict]) -> list[Document]:
    """One document per target capturing BOTH selectivity directions.

    For target T this doc answers:
      - "Which compounds are selective FOR T?" — section A (T is most potent)
      - "Which compounds show higher potency at another target OVER T?" — section B (T was tested but outperformed)

    This is what lets the retriever find "highest selectivity ratio for X over GSK3B":
    the GSK3B index doc lists every compound where another target beat GSK3B, with explicit ratios.

    Assay-type caveats (IC50 vs Ki) are embedded in the text so the LLM sees them.
    """
    target_map = {t["id"]: t for t in targets}

    # Collect best (lowest) activity per compound-target pair
    # stored as (value, atype, tname, gene, units)
    best: dict[str, dict[str, tuple]] = defaultdict(dict)
    compound_meta: dict[str, dict] = {}

    for c in compounds:
        cid = c["id"]
        tid = c["target_id"]
        val = c.get("activity_value")
        if val is None:
            continue

        if cid not in compound_meta:
            compound_meta[cid] = {"name": c.get("name", cid)}

        t_rec = target_map.get(tid)
        gene = t_rec.get("gene", "") if t_rec else ""
        tname = c.get("target_name") or tid
        units = c.get("activity_units", "nM")
        atype = c.get("activity_type", "IC50")

        existing = best[cid].get(tid)
        if existing is None or float(val) < existing[0]:
            best[cid][tid] = (float(val), atype, tname, gene, units)

    # For each target T, partition multi-target compound data into:
    #   is_best[T]   — rows where T is ranked #1 in the compound's profile
    #   is_lower[T]  — rows where T was tested but another target outranked it
    #
    # Row format for is_best:  (cid, cname, T_val, T_atype, T_units, 2nd_gene, 2nd_val, 2nd_atype, 2nd_units, ratio)
    # Row format for is_lower: (cid, cname, best_gene, best_val, best_atype, best_units, T_val, T_atype, T_units, ratio)
    is_best: dict[str, list] = defaultdict(list)
    is_lower: dict[str, list] = defaultdict(list)

    # Also build pairwise lookup: (tid_a, tid_b) → list[(cid, cname, val_a, atype_a, val_b, atype_b, ratio)]
    # where ratio = val_b / val_a (i.e., selectivity for tid_a over tid_b)
    pairwise: dict[tuple[str, str], list] = defaultdict(list)

    for cid, acts in best.items():
        if len(acts) < 2:
            continue

        sorted_acts = sorted(acts.items(), key=lambda x: x[1][0])
        top_tid, (top_val, top_atype, top_tname, top_gene, top_units) = sorted_acts[0]
        second_tid, (sec_val, sec_atype, sec_tname, sec_gene, sec_units) = sorted_acts[1]
        ratio_top_vs_second = sec_val / top_val if top_val > 0 else 0

        cname = compound_meta.get(cid, {}).get("name", cid)

        # Register the best target
        is_best[top_tid].append((
            cid, cname,
            top_val, top_atype, top_units,
            sec_gene or sec_tname, sec_val, sec_atype, sec_units,
            ratio_top_vs_second,
        ))

        # Register all non-best targets
        for tid_i, (t_val, t_atype, t_tname, t_gene, t_units) in sorted_acts[1:]:
            ratio_i = t_val / top_val if top_val > 0 else 0
            is_lower[tid_i].append((
                cid, cname,
                top_gene or top_tname, top_val, top_atype, top_units,
                t_val, t_atype, t_units,
                ratio_i,
            ))

        # Build all pairwise comparisons for this compound
        act_list = list(acts.items())
        for i, (tid_a, (va, at_a, tn_a, gn_a, u_a)) in enumerate(act_list):
            for j, (tid_b, (vb, at_b, tn_b, gn_b, u_b)) in enumerate(act_list):
                if i == j or va <= 0 or vb <= 0:
                    continue
                if va < vb:  # tid_a is more potent (lower value)
                    ratio_ab = vb / va
                    pairwise[(tid_a, tid_b)].append(
                        (cid, cname, va, at_a, u_a, vb, at_b, u_b, ratio_ab)
                    )

    # Build a global "most potent compound per target" lookup across ALL compounds
    # (not just multi-target ones) so we can annotate the index doc header.
    all_acts_per_target: dict[str, list] = defaultdict(list)
    for cid, acts in best.items():
        cname_val = compound_meta.get(cid, {}).get("name", cid)
        for tid_i, (val, atype, tname_i, gene_i, units) in acts.items():
            all_acts_per_target[tid_i].append((val, atype, units, cname_val, cid))

    global_best: dict[str, tuple] = {}  # tid → (val, atype, units, cname, cid)
    for tid_i, rows in all_acts_per_target.items():
        rows.sort(key=lambda x: x[0])
        global_best[tid_i] = rows[0]

    docs = []
    for t in targets:
        tid = t["id"]
        gene = t.get("gene", tid)
        tname = t.get("protein_name", gene)
        family = t.get("target_family", "")

        best_rows = is_best.get(tid, [])
        lower_rows = is_lower.get(tid, [])

        if not best_rows and not lower_rows:
            continue

        display = f"{tname} ({gene})" if gene and gene not in tname else tname
        lines = [
            f"Target selectivity index: {display}",
            f"[POTENCY NOTE: lower IC50/Ki/EC50 value = stronger binding = more potent]",
        ]
        if family:
            lines.append(f"Target family: {family}")

        # ── Global most-potent header ──────────────────────────────────────────
        gb = global_best.get(tid)
        if gb:
            g_val, g_atype, g_units, g_cname, g_cid = gb
            lines.append(
                f"Most potent compound against {gene} in the entire dataset: "
                f"{g_cname} ({g_cid}) — {g_atype}={g_val:.4g} {g_units} "
                f"[this is the LOWEST recorded value = HIGHEST potency]"
            )

        # ── Section A: T is the most potent tested target for these compounds ─
        if best_rows:
            best_rows.sort(key=lambda x: x[9], reverse=True)  # highest ratio first
            lines.append(
                f"\nCompounds where {gene} is the most potent tested target "
                f"({len(best_rows)} compound{'s' if len(best_rows) != 1 else ''}, "
                f"sorted by selectivity ratio descending):"
            )
            for row in best_rows[:25]:
                cid_, cname_, t_val, t_atype, t_units, s_gene, s_val, s_atype, s_units, ratio = row
                assay_note = f" [CAUTION: comparing {t_atype} vs {s_atype} — different assay types]" if t_atype != s_atype else ""
                if ratio >= 2:
                    lines.append(
                        f"  - {cname_} ({cid_}): {gene} {t_atype}={t_val:.4g} {t_units} [LOWER]"
                        f" vs {s_gene} {s_atype}={s_val:.4g} {s_units} [HIGHER]"
                        f" → {gene} is {ratio:.0f}x more potent than {s_gene}{assay_note}"
                    )
                else:
                    lines.append(
                        f"  - {cname_} ({cid_}): {gene} {t_atype}={t_val:.4g} {t_units}"
                        f" ≈ comparable to {s_gene} {s_atype}={s_val:.4g} {s_units}{assay_note}"
                    )

        # ── Section B: T was tested but another target was more potent ────────
        if lower_rows:
            lower_rows.sort(key=lambda x: x[9], reverse=True)  # highest ratio first
            lines.append(
                f"\nCompounds tested on {gene} where another target was MORE potent than {gene} "
                f"({len(lower_rows)} compound{'s' if len(lower_rows) != 1 else ''}, "
                f"sorted by ratio descending):"
            )
            for row in lower_rows[:25]:
                cid_, cname_, b_gene, b_val, b_atype, b_units, t_val, t_atype, t_units, ratio = row
                assay_note = f" [CAUTION: comparing {b_atype} vs {t_atype}]" if b_atype != t_atype else ""
                lines.append(
                    f"  - {cname_} ({cid_}): {b_gene} {b_atype}={b_val:.4g} {b_units} [LOWER=MORE POTENT]"
                    f" vs {gene} {t_atype}={t_val:.4g} {t_units} [HIGHER=LESS POTENT]"
                    f" → {b_gene} is {ratio:.0f}x more potent than {gene}{assay_note}"
                )

        # ── Section C: Pairwise selectivity summary per other target ─────────
        # For each other target that shares compounds with T, show the compound
        # with the HIGHEST selectivity ratio for T over that target.
        pair_keys_for_t = [(k, v) for k, v in pairwise.items() if k[0] == tid]
        if pair_keys_for_t:
            lines.append(
                f"\nPairwise selectivity for {gene} over each other target "
                f"(showing compound with HIGHEST ratio for each pair):"
            )
            for (_, other_tid), pair_rows in sorted(pair_keys_for_t, key=lambda x: max(r[8] for r in x[1]), reverse=True):
                pair_rows_sorted = sorted(pair_rows, key=lambda x: x[8], reverse=True)
                top = pair_rows_sorted[0]
                cid_, cname_, va, at_a, u_a, vb, at_b, u_b, ratio = top
                other_gene = target_map.get(other_tid, {}).get("gene", other_tid)
                assay_note = f" [CAUTION: {at_a} vs {at_b}]" if at_a != at_b else ""
                lines.append(
                    f"  ★ {gene} over {other_gene}: best compound = {cname_} ({cid_}), "
                    f"{gene} {at_a}={va:.4g} {u_a} vs {other_gene} {at_b}={vb:.4g} {u_b} "
                    f"→ {ratio:.1f}x selectivity for {gene}{assay_note} "
                    f"(from {len(pair_rows_sorted)} compounds tested on both)"
                )

        docs.append(Document(
            text="\n".join(lines),
            doc_id=f"target_sel_index_{tid}",
            metadata={
                "doc_type": "target_selectivity_index",
                "gene": gene,
                "uniprot_id": tid if t.get("source") == "uniprot" else "",
                "compound_id": "",
                "targets": tid,
                "target_family": family,
                "sources": t.get("source", ""),
                "chunk_index": 0,
            },
        ))

    return docs


def build_evidence_depth_docs(compounds: list[dict], targets: list[dict]) -> list[Document]:
    """One document per target summarising evidence quality for its top compounds.

    For each target, aggregates:
      - Number of independent assays per compound-target pair
      - Number of independent publications
      - Assay type breakdown (Binding vs Functional vs ADME)
      - Activity value spread (min/max/median) as a consistency check
      - pChEMBL availability and range
      - Data validity flags
      - Year range of measurements
      - Species coverage
      - Evidence confidence score (composite of above)

    This enables the LLM to answer prompts about evidence strength,
    distinguishing well-validated compounds from single-assay hits.
    """
    target_map = {t["id"]: t for t in targets}

    # Group all activity records by (target_id, compound_id)
    pair_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in compounds:
        pair_records[(c["target_id"], c["id"])].append(c)

    # Build per-target evidence summaries
    docs = []
    target_evidence: dict[str, list[dict]] = defaultdict(list)

    for (tid, cid), records in pair_records.items():
        assay_ids = {r.get("assay_chembl_id", "") for r in records if r.get("assay_chembl_id")}
        doc_ids = {r.get("document_chembl_id", "") for r in records if r.get("document_chembl_id")}
        assay_types = [r.get("assay_type", "U") for r in records]
        values = [r["activity_value"] for r in records if r.get("activity_value") is not None]
        pchembl_vals = [r["pchembl_value"] for r in records if r.get("pchembl_value") is not None]
        years = [r.get("document_year") for r in records if r.get("document_year")]
        organisms = {r.get("target_organism", "Homo sapiens") for r in records}
        validity_flags = [r.get("data_validity_comment", "") for r in records if r.get("data_validity_comment")]
        act_types_used = {r.get("activity_type", "IC50") for r in records}

        n_assays = len(assay_ids - {""})
        n_docs = len(doc_ids - {""})
        n_records = len(records)

        # Activity consistency: coefficient of variation
        if len(values) >= 2:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            cv = std_val / mean_val if mean_val > 0 else 0
        else:
            cv = None

        # Evidence confidence score (0–100)
        # Rewards: multiple assays, publications, consistent values, pChEMBL, no flags
        score = 0
        score += min(n_assays * 10, 30)       # up to 30 pts for independent assays
        score += min(n_docs * 10, 20)          # up to 20 pts for publications
        score += min(n_records * 2, 10)        # up to 10 pts for measurement count
        if pchembl_vals:
            score += 10                        # 10 pts for having pChEMBL
        if cv is not None and cv < 0.5:
            score += 15                        # 15 pts for consistent values (CV<50%)
        elif cv is not None and cv < 1.0:
            score += 5                         # 5 pts for moderate consistency
        if not validity_flags:
            score += 10                        # 10 pts for no data quality flags
        if len(act_types_used) == 1:
            score += 5                         # 5 pts for single endpoint type

        best_val = min(values) if values else None
        median_val = statistics.median(values) if values else None

        summary = {
            "compound_id": cid,
            "compound_name": records[0].get("name", cid),
            "target_id": tid,
            "n_records": n_records,
            "n_assays": n_assays,
            "n_publications": n_docs,
            "assay_type_counts": dict(Counter(assay_types)),
            "act_types": sorted(act_types_used),
            "best_value": best_val,
            "median_value": median_val,
            "cv": cv,
            "pchembl_range": (min(pchembl_vals), max(pchembl_vals)) if pchembl_vals else None,
            "year_range": (min(years), max(years)) if years else None,
            "organisms": sorted(organisms),
            "validity_flags": validity_flags,
            "confidence_score": score,
        }
        target_evidence[tid].append(summary)

    for t in targets:
        tid = t["id"]
        gene = t.get("gene", tid)
        tname = t.get("protein_name", gene)
        family = t.get("target_family", "")

        summaries = target_evidence.get(tid, [])
        if not summaries:
            continue

        # Sort by confidence score descending, then by best_value ascending
        summaries.sort(key=lambda s: (-s["confidence_score"],
                                       s["best_value"] if s["best_value"] is not None else 1e9))

        display = f"{tname} ({gene})" if gene and gene not in tname else tname
        lines = [
            f"Evidence depth report: {display}",
            f"Total compound-target pairs with activity data: {len(summaries)}",
            f"[METHODOLOGY: confidence_score = f(n_assays, n_publications, value_consistency, "
            f"data_quality). Score 0–100; >60 = well-validated, 30–60 = moderate, <30 = weak]",
            "",
        ]

        # Flag the "strongest-looking but weakest evidence" candidates
        if summaries:
            potent_weak = [s for s in summaries
                           if s["best_value"] is not None and s["best_value"] < 100
                           and s["confidence_score"] < 30]
            if potent_weak:
                top_pw = potent_weak[0]
                lines.append(
                    f"⚠ EVIDENCE CAUTION: {top_pw['compound_name']} ({top_pw['compound_id']}) "
                    f"appears very potent (best={top_pw['best_value']:.4g} nM) but has weak "
                    f"evidence support (confidence={top_pw['confidence_score']}/100, "
                    f"assays={top_pw['n_assays']}, pubs={top_pw['n_publications']})"
                )
                lines.append("")

        # Top compounds by confidence
        lines.append("Top compounds ranked by evidence confidence (not just potency):")
        for i, s in enumerate(summaries[:15]):
            assay_str = "/".join(f"{k}={v}" for k, v in s["assay_type_counts"].items())
            cv_str = f", CV={s['cv']:.2f}" if s["cv"] is not None else ""
            pch_str = (f", pChEMBL={s['pchembl_range'][0]:.1f}–{s['pchembl_range'][1]:.1f}"
                       if s["pchembl_range"] else "")
            year_str = (f", years={s['year_range'][0]}–{s['year_range'][1]}"
                        if s["year_range"] else "")
            org_str = f", species={','.join(s['organisms'])}" if len(s["organisms"]) > 1 else ""
            flag_str = f"  ⚠ FLAGS: {'; '.join(s['validity_flags'])}" if s["validity_flags"] else ""
            endpoints = "/".join(s["act_types"])

            lines.append(
                f"  {i+1}. {s['compound_name']} ({s['compound_id']}): "
                f"confidence={s['confidence_score']}/100, "
                f"best {endpoints}={s['best_value']:.4g} nM, "
                f"records={s['n_records']}, assays={s['n_assays']}, pubs={s['n_publications']}, "
                f"assay_types=[{assay_str}]{cv_str}{pch_str}{year_str}{org_str}"
            )
            if flag_str:
                lines.append(flag_str)

        # Bottom of the confidence list — "weakest evidence" candidates
        weak = [s for s in summaries if s["confidence_score"] < 30]
        if weak:
            lines.append(f"\n{len(weak)} compound(s) with weak evidence (confidence <30):")
            for s in weak[:5]:
                lines.append(
                    f"  - {s['compound_name']} ({s['compound_id']}): "
                    f"confidence={s['confidence_score']}/100, "
                    f"best={s['best_value']:.4g} nM, "
                    f"assays={s['n_assays']}, pubs={s['n_publications']}"
                )

        docs.append(Document(
            text="\n".join(lines),
            doc_id=f"evidence_{tid}",
            metadata={
                "doc_type": "evidence_depth",
                "gene": gene,
                "uniprot_id": "",
                "compound_id": "",
                "targets": tid,
                "target_family": family,
                "sources": "computed",
                "chunk_index": 0,
            },
        ))

    return docs


def build_assay_comparison_docs(compounds: list[dict], targets: list[dict]) -> list[Document]:
    """Detect compound-target pairs with substantial disagreement across assays.

    Flags pairs where activity values vary by >10x across measurements,
    and documents the assay types, years, and possible reasons for the discrepancy.
    """
    target_map = {t["id"]: t for t in targets}

    # Group by (compound_id, target_id)
    pair_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in compounds:
        if c.get("activity_value") is not None:
            pair_records[(c["id"], c["target_id"])].append(c)

    docs = []
    discrepancies: list[tuple] = []

    for (cid, tid), records in pair_records.items():
        if len(records) < 2:
            continue
        values = [r["activity_value"] for r in records]
        max_val = max(values)
        min_val = min(values)
        if min_val <= 0:
            continue
        fold_range = max_val / min_val
        if fold_range < 10:
            continue

        t_rec = target_map.get(tid, {})
        gene = t_rec.get("gene", tid)
        cname = records[0].get("name", cid)
        discrepancies.append((cid, cname, tid, gene, records, fold_range))

    if not discrepancies:
        return docs

    # Sort by fold_range descending
    discrepancies.sort(key=lambda x: x[5], reverse=True)

    lines = [
        "Assay discrepancy report: compound-target pairs with >10x value disagreement",
        f"Total discrepant pairs found: {len(discrepancies)}",
        "[These pairs require careful interpretation — do not assume any single value is definitive]",
        "",
    ]

    for cid, cname, tid, gene, records, fold_range in discrepancies[:20]:
        values = sorted(records, key=lambda r: r["activity_value"])
        min_rec = values[0]
        max_rec = values[-1]

        # Analyze possible reasons
        reasons = []
        act_types_used = {r.get("activity_type") for r in records}
        if len(act_types_used) > 1:
            reasons.append(f"mixed endpoint types ({'/'.join(sorted(act_types_used))})")
        assay_types_used = {r.get("assay_type", "U") for r in records}
        if len(assay_types_used) > 1:
            reasons.append(f"different assay formats ({'/'.join(sorted(assay_types_used))})")
        organisms = {r.get("target_organism", "?") for r in records}
        if len(organisms) > 1:
            reasons.append(f"different species ({'/'.join(sorted(organisms))})")
        years = [r.get("document_year") for r in records if r.get("document_year")]
        if years and max(years) - min(years) > 5:
            reasons.append(f"measurements span {min(years)}–{max(years)}")
        flagged = [r for r in records if r.get("data_validity_comment")]
        if flagged:
            reasons.append(f"data quality flags present")

        reason_str = "; ".join(reasons) if reasons else "no obvious methodological explanation"

        lines.append(
            f"  {cname} ({cid}) vs {gene} ({tid}): "
            f"{fold_range:.0f}x range — "
            f"min={min_rec['activity_type']} {min_rec['activity_value']:.4g} nM, "
            f"max={max_rec['activity_type']} {max_rec['activity_value']:.4g} nM "
            f"(n={len(records)} records). "
            f"Possible reasons: {reason_str}"
        )

    docs.append(Document(
        text="\n".join(lines),
        doc_id="assay_discrepancy_report",
        metadata={
            "doc_type": "assay_discrepancy",
            "gene": "",
            "uniprot_id": "",
            "compound_id": "",
            "targets": "",
            "target_family": "",
            "sources": "computed",
            "chunk_index": 0,
        },
    ))

    return docs


def build_all(targets: list[dict], compounds: list[dict]) -> list[Document]:
    """Build all document types and return combined list."""
    docs = []
    docs.extend(build_target_profiles(targets, compounds))
    docs.extend(build_compound_activity_docs(compounds, targets))
    docs.extend(build_selectivity_docs(compounds, targets))
    docs.extend(build_compound_selectivity_docs(compounds, targets))
    docs.extend(build_target_selectivity_index_docs(compounds, targets))
    docs.extend(build_evidence_depth_docs(compounds, targets))
    docs.extend(build_assay_comparison_docs(compounds, targets))
    return docs


# ── helpers ──────────────────────────────────────────────────────────────────

def _summarize_activities(acts: list[dict]) -> str:
    if not acts:
        return "Known ligands: None in dataset."
    values = [a["activity_value"] for a in acts if a.get("activity_value") is not None]
    if not values:
        return f"Known ligands: {len(acts)} records, no numeric activity values."
    mn, mx = min(values), max(values)
    unit = acts[0].get("activity_units", "nM")
    # Find most potent
    best = min(acts, key=lambda a: a.get("activity_value", float("inf")))
    return (
        f"Known ligands: {len(acts)} compounds tested. "
        f"Activity range: {mn:.4g}–{mx:.4g} {unit}. "
        f"Most potent: {best.get('name', best['id'])} "
        f"({best['activity_type']} = {best['activity_value']:.4g} {unit})"
    )


def _chunk_list(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _target_pairs(targets: list[dict], max_pairs: int) -> list[tuple[dict, dict]]:
    pairs = []
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            pairs.append((targets[i], targets[j]))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs
