"""LLM answer generation — Anthropic | Ollama | Local HuggingFace (offline)."""

from __future__ import annotations

import logging

import requests

from .config import Config
from .retrieve import RetrievedDoc

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a drug discovery research assistant with deep expertise in pharmacology, medicinal \
chemistry, structural biology, and molecular medicine.

Answer questions about drug-target interactions using ONLY the provided context documents, \
which include experimental bioactivity data, evidence depth reports, and knowledge-graph \
derived biological context.

General rules:
- Cite sources inline as [Source N] (e.g. "Gefitinib has IC50 = 0.033 nM [Source 1]")
- Always include specific activity values (IC50, Ki, EC50, Kd) when present in the context
- If the context is insufficient to answer, say so clearly — do not hallucinate data
- Prefer quantitative statements over qualitative ones

═══ EVIDENCE vs RELEVANCE (critical distinction) ═══

The "relevance" score on each source is a RETRIEVAL RELEVANCE score — it measures how \
semantically close the document is to the query. It is NOT a measure of experimental evidence \
strength. A document with relevance=0.95 but confidence_score=15/100 is a well-retrieved \
but poorly supported claim.

When answering questions about evidence strength, confidence, or reliability:
- Use the confidence_score from evidence_depth documents (0–100 scale)
- Report n_assays (independent experiments), n_publications (independent papers), \
  and value consistency (CV) when available
- A compound with 1 assay and 1 publication is WEAKLY supported regardless of its potency
- A compound with 5+ assays, 3+ publications, and CV<0.5 is WELL supported
- Flag data_validity_comments when present — these are ChEMBL quality warnings

═══ METHODOLOGICAL RIGOR ═══

8. Unit and endpoint comparisons:
   - All values are normalised to nM. If the original was µM, it has been converted.
   - IC50 and Ki measure different things. IC50 is assay-dependent; Ki is \
     theoretically assay-independent. Comparing them is an approximation.
   - pChEMBL values (-log10 M) enable cross-endpoint comparison when available.
   - If a question asks to "normalize across" endpoints, use pChEMBL if available \
     and explain limitations of direct nM comparison.

9. Assay type awareness:
   - B = Binding assay, F = Functional assay, A = ADME, T = Toxicity
   - Binding data (B) is not evidence of functional inhibition
   - Functional data (F) is stronger evidence of biological effect
   - When distinguishing "binders" from "inhibitors", look at assay_type

10. Target-level vs compound-level reasoning:
    - When asked "which TARGET has the weakest/strongest evidence", compare \
      targets against each other, not compounds within one target
    - When asked "which COMPOUND", compare compounds
    - Never answer a target question with a compound answer or vice versa

11. Disagreement and inconsistency:
    - If an assay_discrepancy report is present, use it to flag unreliable values
    - A >10x range in reported values for the same compound-target pair means \
      the "best" value is uncertain — report the range, not just the minimum
    - Possible reasons: mixed endpoint types, different species, different assay formats

12. Species and organism:
    - Default is Homo sapiens. If the context mentions other species, note it.
    - "Active against human JAK2" and "active against mouse Jak2" are different claims.

═══ RESEARCH SYNTHESIS (for multi-faceted questions) ═══

When the question asks you to evaluate, recommend, compare, or reason about drug
candidates, synthesize ALL available evidence types into a structured assessment:
  1. POTENCY: cite specific IC50/Ki/EC50 values from activity/selectivity docs
  2. SELECTIVITY: note off-target activity, fold differences between targets
  3. EVIDENCE QUALITY: use confidence_score, n_assays, n_publications if available
  4. BIOLOGICAL RATIONALE: use knowledge graph context (disease associations,
     GO terms, pathway involvement) to explain WHY a target matters
  5. RISKS: flag data quality warnings, assay discrepancies, off-target liabilities

Weigh all five dimensions — a potent compound with no biological rationale or weak
evidence is a poor candidate. A less potent compound targeting a validated disease
pathway with strong evidence may be better.

Knowledge graph sources (labelled "[Biological Knowledge Graph]"):
- Use disease associations to explain therapeutic relevance
- Use GO biological process annotations to explain mechanism of action
- Use GO molecular function annotations to characterise enzyme class
- Use family membership to contextualise targets
- Combine graph context with quantitative data for biological narrative
- These provide the "why" behind the numbers — always integrate them into answers

═══ ANTI-HALLUCINATION RULES (strictly enforced) ═══

1. NEVER invent or estimate activity values.

2. Selectivity ratios require BOTH values from the SAME source document.

3. Direction check before every selectivity claim:
   - Lower IC50/Ki/EC50/Kd = MORE potent (tighter binding).
   - selectivity ratio = HIGHER_value / LOWER_value.
   - State both values and which is lower.

4. "Most potent in the dataset" requires explicit label in the source.
   Comparing two compounds does NOT prove either is the global champion.

5. If a source flags "[CAUTION: comparing IC50 vs Ki]", echo that caveat.

6. For "highest selectivity" questions, scan ALL sources for pre-computed ratios.

7. SMILES strings represent structure only — do not infer potency from SMILES.
"""

# Condensed prompt for small local models (Phi-3-mini-4k, ~600 tokens vs ~1100)
_LOCAL_SYSTEM_PROMPT = """\
You are a drug discovery research assistant. Answer using ONLY the provided context documents.

Rules:
- Cite sources as [Source N] with specific activity values (IC50, Ki, EC50, Kd in nM)
- If context is insufficient, say so — NEVER invent data or compound IDs
- Lower IC50/Ki = MORE potent. Selectivity ratio = higher_value / lower_value
- "Relevance" score = retrieval similarity, NOT evidence strength
- For evidence questions, use confidence_score, n_assays, n_publications from context

Research synthesis:
- When sources include [Biological Knowledge Graph] data, USE it to explain:
  • WHY a target matters (disease associations, biological processes)
  • HOW a compound works (mechanism of action, molecular function)
  • THERAPEUTIC CONTEXT (what diseases could be treated, clinical relevance)
- Combine quantitative data (IC50/Ki values) with biological context for complete answers
- For "which would you advance" questions: weigh potency, selectivity, evidence quality,
  AND biological rationale (disease relevance, target function)
- Only reference compound IDs that appear EXACTLY in a [Source N] document
"""

# ── module-level cache so the local model loads once per process ──────────────
_local_llm_cache: dict[str, "_LocalLLM"] = {}


class _LocalLLM:
    """HuggingFace generative LLM loaded locally — no API key, runs on MPS/CUDA/CPU."""

    def __init__(self, model_name: str, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens

        # Resolve device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info("Loading local LLM '%s' on %s…", model_name, device)
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        logger.info("Local LLM ready on %s", device)

    def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        result = self._pipe(messages)
        # Extract only the newly generated assistant turn
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return str(generated)


def _get_local_llm(cfg: Config) -> _LocalLLM:
    key = cfg.llm.local_model
    if key not in _local_llm_cache:
        _local_llm_cache[key] = _LocalLLM(
            model_name=cfg.llm.local_model,
            device=cfg.llm.local_device,
            max_new_tokens=cfg.llm.local_max_new_tokens,
        )
    return _local_llm_cache[key]


# ── query classification ──────────────────────────────────────────────────────

_EVIDENCE_KW = {
    "evidence", "confidence", "supported", "weak", "reliable",
    "assay count", "publication", "trust", "quality",
}

_RESEARCH_KW = {
    "mechanism", "how does", "why", "explain", "role", "function",
    "therapeutic", "treatment", "clinical", "advance", "candidate",
    "disease", "indication", "pathway", "signaling", "recommend",
    "which would", "should", "compare", "repurpos", "combination",
    "tell me about", "describe", "summarize", "overview",
    "what targets", "what compounds", "safety", "toxicity",
    "off-target", "resistance", "biomarker", "polypharmacology",
}


def _select_docs_for_small_ctx(
    query: str, docs: list[RetrievedDoc],
) -> list[RetrievedDoc]:
    """Pick up to 3 diverse docs for the 4k context model.

    Strategy by query type:
      Evidence: evidence_depth first, then activity/selectivity
      Research: target_profile/graph_context + activity + selectivity (diverse)
      Simple:   top-1 by rerank score (usually compound_selectivity)
    """
    q_lower = query.lower()
    is_evidence = any(kw in q_lower for kw in _EVIDENCE_KW)
    is_research = any(kw in q_lower for kw in _RESEARCH_KW)

    if is_evidence or is_research:
        # Both evidence and research queries benefit from diverse doc types.
        # Categorize available docs and pick the best from each category.
        evidence = [d for d in docs if d.metadata.get("doc_type") in
                    ("evidence_depth", "assay_discrepancy")]
        bio = [d for d in docs if d.metadata.get("doc_type") in
               ("target_profile", "graph_context")]
        activity = [d for d in docs if d.metadata.get("doc_type") in
                    ("compound_selectivity", "target_selectivity_index",
                     "selectivity_comparison", "compound_activity")]

        selected: list[RetrievedDoc] = []
        selected_ids: set[str] = set()

        def _pick(pool: list[RetrievedDoc]) -> None:
            for d in pool:
                if d.doc_id not in selected_ids:
                    selected.append(d)
                    selected_ids.add(d.doc_id)
                    return

        # Priority order depends on query type
        if is_evidence:
            _pick(evidence)   # evidence first
            _pick(activity)   # then potency data
            _pick(bio)        # then biological context
        else:
            _pick(bio)        # biological context first for research
            _pick(activity)   # then potency data
            _pick(evidence)   # then evidence quality

        # Fill remaining slots (up to 3) from top-ranked docs
        for d in docs:
            if len(selected) >= 3:
                break
            if d.doc_id not in selected_ids:
                selected.append(d)
                selected_ids.add(d.doc_id)
        return selected or docs[:1]

    # Simple query — just top-1
    return docs[:1]


# ── public API ─────────────────────────────────────────────────────────────────

def build_context(docs: list[RetrievedDoc], max_chars: int = 0) -> str:
    """Build context string from retrieved docs.

    Args:
        docs: ranked documents
        max_chars: if >0, truncate each doc and total context to fit
                   within this character budget (rough proxy for tokens)
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        text = doc.text
        if max_chars > 0:
            # Per-doc budget: leave room for all docs
            per_doc = max(400, max_chars // max(len(docs), 1))
            if len(text) > per_doc:
                text = text[:per_doc] + "\n[…truncated]"
        parts.append(f"[Source {i}] (relevance: {doc.rerank_score:.3f})\n{text}")

    context = "\n\n---\n\n".join(parts)
    if max_chars > 0 and len(context) > max_chars:
        context = context[:max_chars] + "\n[…context truncated to fit model window]"
    return context


def generate(
    query: str,
    docs: list[RetrievedDoc],
    cfg: Config,
) -> str:
    """Generate an answer from retrieved documents.

    Provider routing:
      anthropic → Anthropic Claude API (requires ANTHROPIC_API_KEY + credits)
      ollama    → Local Ollama server  (requires `ollama serve` running)
      local     → HuggingFace model downloaded to ~/.cache/huggingface (offline, no key)
    """
    if not docs:
        return "No relevant documents found. Run `dti ingest` and `dti embed` first."

    # Context budget for local models:
    # Phi-3-mini-4k has 4096 token limit. Budget breakdown:
    #   _LOCAL_SYSTEM_PROMPT ≈ 200 tokens
    #   Question ≈ 50 tokens
    #   max_new_tokens (generation) = from config (default 1000)
    #   → context budget = 4096 - 200 - 50 - max_new_tokens
    # For 4k model with 1000 gen tokens: ~2800 tokens ≈ 8500 chars
    # Also limit to top-2 docs to keep context tight for small models.
    if cfg.llm.provider == "local":
        is_small_ctx = "4k" in cfg.llm.local_model.lower()
        if is_small_ctx:
            gen_tokens = cfg.llm.local_max_new_tokens
            available_tokens = 4096 - 250 - 50 - gen_tokens
            max_chars = min(7000, max(1500, available_tokens * 3))
            docs = _select_docs_for_small_ctx(query, docs)
        else:
            max_chars = 60000
    else:
        max_chars = 0
    context = build_context(docs, max_chars=max_chars)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        if cfg.llm.provider == "anthropic":
            return _call_anthropic(user_message, cfg)
        elif cfg.llm.provider == "ollama":
            return _call_ollama(user_message, cfg)
        elif cfg.llm.provider == "local":
            return _call_local(user_message, cfg)
        else:
            raise ValueError(f"Unknown LLM provider: {cfg.llm.provider!r}. "
                             "Set provider to 'anthropic', 'ollama', or 'local'.")
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        fallback = ["LLM generation failed. Retrieved evidence:\n"]
        for i, doc in enumerate(docs, start=1):
            fallback.append(f"[Source {i}] score={doc.rerank_score:.3f}\n{doc.text[:400]}...")
        return "\n\n".join(fallback)


def _call_anthropic(user_message: str, cfg: Config) -> str:
    import os
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    response = client.messages.create(
        model=cfg.llm.anthropic_model,
        max_tokens=2000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _call_ollama(user_message: str, cfg: Config) -> str:
    url = f"{cfg.llm.ollama_base_url}/api/chat"
    payload = {
        "model": cfg.llm.ollama_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_local(user_message: str, cfg: Config) -> str:
    llm = _get_local_llm(cfg)
    # Use full system prompt for 128k models; fall back to condensed for 4k models
    is_small_ctx = "4k" in cfg.llm.local_model.lower()
    prompt = _LOCAL_SYSTEM_PROMPT if is_small_ctx else _SYSTEM_PROMPT
    return llm.generate(prompt, user_message)
