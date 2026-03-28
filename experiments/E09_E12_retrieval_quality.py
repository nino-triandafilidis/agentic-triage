"""Retrieval quality experiments E09–E12.

E09 — Multi-facet query decomposition: 3 separate queries, merge top-5
E10 — LLM re-ranking: retrieve 20, LLM scores relevance, keep top-5
E11 — Distance-gated retrieval: fall back to LLM-only above threshold
E12 — Adaptive re-querying: rewrite + re-retrieve on poor initial results

All experiments use the same 150 dev_tune rows, 8k context_chars,
front-matter stripping, and gemini-2.5-flash for generation.

Usage:
    .venv/bin/python experiments/E09_E12_retrieval_quality.py \
        --experiment E11 --input data/splits/dev_tune.csv --n-rows 150

    # Run all:
    .venv/bin/python experiments/E09_E12_retrieval_quality.py \
        --experiment all --input data/splits/dev_tune.csv --n-rows 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from google.genai.types import GenerateContentConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config
from experiments import tracking
from experiments.eval_triage import evaluate, find_gt_column
from src.rag.retrieval import search_pubmed_articles, _embed_query, _load_faiss_index
from src.rag.triage_core import (
    GENERATION_TEMPERATURE,
    build_context_block,
    parse_triage,
    get_prompt_template,
    fetch_vertex_pricing,
    FALLBACK_PRICING,
)
from src.rag.query_agents import _concat_fields
from src.schemas import TriagePrediction

RUNS_DIR = ROOT / "data" / "runs"

# ── Front-matter stripping (from E04) ────────────────────────────────────

_SECTION_MARKERS = [
    "==== Body", "Introduction", "INTRODUCTION",
    "Background", "BACKGROUND", "Abstract", "ABSTRACT",
    "Methods", "METHODS", "Objective", "Objectives",
    "Purpose", "Simple Summary", "Results", "RESULTS",
]


def strip_front_matter(text: str) -> str:
    """Strip XML/metadata front-matter from article text."""
    best_pos = len(text)
    for marker in _SECTION_MARKERS:
        pos = text.find(f"\n{marker}")
        if pos == -1:
            pos = text.find(marker)
        if pos != -1 and pos < best_pos:
            best_pos = pos
    return text[best_pos:] if best_pos < len(text) else text


def build_context_block_stripped(articles: pd.DataFrame, context_chars: int) -> str:
    """Build context block with front-matter stripping applied."""
    if articles.empty:
        return "No relevant articles found."
    parts = []
    for i, row in articles.iterrows():
        text = strip_front_matter(str(row["article_text"]))[:context_chars]
        parts.append(f"--- Article {i + 1} (PMID {row['pmid']}) ---\n{text}")
    return "\n\n".join(parts) if parts else "No relevant articles found."


# ── Shared LLM call with retry ───────────────────────────────────────────

@retry(
    retry=retry_if_exception(lambda e: "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _llm_call(prompt: str, model_id: str, temperature: float = 0.0,
              json_schema: dict | None = None) -> Any:
    """Call Gemini with retry on rate limits."""
    gen_config = GenerateContentConfig(temperature=temperature)
    if json_schema:
        gen_config = GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=json_schema,
        )
    return config.genai_client.models.generate_content(
        model=model_id, contents=[prompt], config=gen_config,
    )


def _extract_usage(resp) -> dict:
    """Extract token usage from a Gemini response."""
    if not resp.usage_metadata:
        return {"prompt": 0, "completion": 0, "thinking": 0}
    return {
        "prompt": resp.usage_metadata.prompt_token_count or 0,
        "completion": resp.usage_metadata.candidates_token_count or 0,
        "thinking": getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0,
    }


# ── E11: Distance-gated retrieval ────────────────────────────────────────

def run_e11(df: pd.DataFrame, model_id: str, pricing: dict | None,
            context_chars: int, top_k: int, thresholds: list[float],
            run_dir: Path, ts: str, per_row_diag: bool,
            prompt_template: str = "default") -> list[dict]:
    """E11 — Distance-gated: fall back to LLM-only when top1_distance > threshold."""
    prompts = get_prompt_template(prompt_template)
    all_summaries = []

    for threshold in thresholds:
        label = f"E11_gate_{threshold:.2f}"
        print(f"\n{'='*60}\n{label}\n{'='*60}")

        results = []
        total_prompt_tok = 0
        total_comp_tok = 0
        total_think_tok = 0
        fallback_count = 0

        for idx in tqdm(range(len(df)), desc=label):
            row = df.iloc[idx]
            case = row.to_dict()
            query_text = _concat_fields(case)

            articles = search_pubmed_articles(query_text, top_k=top_k)
            top1_dist = float(articles["score"].iloc[0]) if len(articles) > 0 else 999.0
            mean_dist = float(articles["score"].mean()) if len(articles) > 0 else 999.0

            hpi = str(case.get("HPI", ""))
            patient_info = str(case.get("patient_info", ""))
            initial_vitals = str(case.get("initial_vitals", ""))

            use_rag = top1_dist <= threshold
            if use_rag:
                context = build_context_block_stripped(articles, context_chars)
                prompt = prompts["rag"].format(
                    context=context, hpi=hpi,
                    patient_info=patient_info, initial_vitals=initial_vitals,
                )
            else:
                fallback_count += 1
                prompt = prompts["llm"].format(
                    hpi=hpi, patient_info=patient_info,
                    initial_vitals=initial_vitals,
                )

            try:
                resp = _llm_call(prompt, model_id,
                                 json_schema=TriagePrediction.model_json_schema())
                raw_text = resp.text
                usage = _extract_usage(resp)
                total_prompt_tok += usage["prompt"]
                total_comp_tok += usage["completion"]
                total_think_tok += usage["thinking"]
            except Exception as e:
                raw_text = None
                print(f"WARNING row {idx}: {e}")

            parsed = parse_triage(raw_text)
            results.append({
                "triage_RAG_raw": raw_text,
                "triage_RAG": parsed,
                "top1_distance": round(top1_dist, 6),
                "mean_topk_distance": round(mean_dist, 6),
                "used_rag": use_rag,
                "gate_threshold": threshold,
            })

        results_df = df.copy()
        for col in ["triage_RAG_raw", "triage_RAG", "top1_distance",
                     "mean_topk_distance", "used_rag", "gate_threshold"]:
            results_df[col] = [r[col] for r in results]

        csv_path = run_dir / f"{label}_{ts}.csv"
        results_df.to_csv(csv_path, index=False)

        gt_col = find_gt_column(results_df, None)
        metrics = evaluate(results_df[gt_col], results_df["triage_RAG"], label=label)

        cost = _compute_run_cost(pricing, total_prompt_tok, total_comp_tok, total_think_tok)
        fallback_rate = fallback_count / len(df)

        sidecar = {
            "experiment": "E11", "label": label,
            "threshold": threshold, "fallback_rate": round(fallback_rate, 4),
            "fallback_count": fallback_count,
            "n_rows": len(df), "model": model_id,
            "context_chars": context_chars, "top_k": top_k,
            "metrics": metrics, "cost_usd": cost,
            "prompt_tokens": total_prompt_tok,
            "completion_tokens": total_comp_tok,
            "thinking_tokens": total_think_tok,
        }
        sidecar_path = csv_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        print(f"Saved: {csv_path.name}")
        print(f"  κ={metrics.get('quadratic_kappa', 'N/A'):.3f}  "
              f"exact={metrics.get('exact_accuracy', 'N/A'):.3f}  "
              f"fallback_rate={fallback_rate:.1%}  cost=${cost:.3f}")

        all_summaries.append({**sidecar, "csv_path": str(csv_path)})

    return all_summaries


# ── E10: LLM Re-ranking ──────────────────────────────────────────────────

_RERANK_PROMPT = (
    "Rate how useful this medical article excerpt is for determining the "
    "ESI triage level of the patient described below. "
    "Output ONLY a single integer from 1 (not useful) to 5 (very useful).\n\n"
    "Patient: {patient_summary}\n\n"
    "Article excerpt (first 1000 chars):\n{article_excerpt}\n\n"
    "Relevance score (1-5):"
)


def _rerank_articles(articles: pd.DataFrame, case: dict,
                     model_id: str, context_chars: int) -> pd.DataFrame:
    """Score each article with LLM and return sorted by relevance desc, then distance asc."""
    hpi = str(case.get("HPI", ""))[:300]
    patient_info = str(case.get("patient_info", ""))[:100]
    patient_summary = f"{hpi} | {patient_info}"

    scores = []
    for _, row in articles.iterrows():
        stripped = strip_front_matter(str(row["article_text"]))[:1000]
        prompt = _RERANK_PROMPT.format(
            patient_summary=patient_summary,
            article_excerpt=stripped,
        )
        try:
            resp = _llm_call(prompt, model_id, temperature=0.0)
            text = (resp.text or "").strip()
            match = re.search(r'[1-5]', text)
            score = int(match.group()) if match else 1
        except Exception:
            score = 1
        scores.append(score)

    articles = articles.copy()
    articles["llm_relevance"] = scores
    return articles.sort_values(
        ["llm_relevance", "score"], ascending=[False, True]
    ).reset_index(drop=True)


def run_e10(df: pd.DataFrame, model_id: str, pricing: dict | None,
            context_chars: int, retrieve_k: int, keep_k: int,
            run_dir: Path, ts: str, per_row_diag: bool,
            prompt_template: str = "default") -> list[dict]:
    """E10 — LLM re-ranking: retrieve top-20, rerank, keep top-5."""
    prompts = get_prompt_template(prompt_template)
    label = f"E10_rerank_k{retrieve_k}_keep{keep_k}"
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    results = []
    diagnostics = []
    total_prompt_tok = 0
    total_comp_tok = 0
    total_think_tok = 0
    rerank_prompt_tok = 0

    for idx in tqdm(range(len(df)), desc=label):
        row = df.iloc[idx]
        case = row.to_dict()
        query_text = _concat_fields(case)

        all_articles = search_pubmed_articles(query_text, top_k=retrieve_k)
        original_top1 = float(all_articles["score"].iloc[0]) if len(all_articles) > 0 else 999.0

        reranked = _rerank_articles(all_articles, case, model_id, context_chars)
        kept = reranked.head(keep_k).reset_index(drop=True)

        reranked_top1 = float(kept["score"].iloc[0]) if len(kept) > 0 else 999.0
        reranked_mean = float(kept["score"].mean()) if len(kept) > 0 else 999.0

        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = str(case.get("initial_vitals", ""))

        context = build_context_block_stripped(kept, context_chars)
        prompt = prompts["rag"].format(
            context=context, hpi=hpi,
            patient_info=patient_info, initial_vitals=initial_vitals,
        )

        try:
            resp = _llm_call(prompt, model_id,
                             json_schema=TriagePrediction.model_json_schema())
            raw_text = resp.text
            usage = _extract_usage(resp)
            total_prompt_tok += usage["prompt"]
            total_comp_tok += usage["completion"]
            total_think_tok += usage["thinking"]
        except Exception as e:
            raw_text = None
            print(f"WARNING row {idx}: {e}")

        parsed = parse_triage(raw_text)

        overlap = set(all_articles.head(keep_k)["pmc_id"]) & set(kept["pmc_id"])
        overlap_rate = len(overlap) / keep_k if keep_k > 0 else 0

        results.append({
            "triage_RAG_raw": raw_text,
            "triage_RAG": parsed,
            "top1_distance_original": round(original_top1, 6),
            "top1_distance_reranked": round(reranked_top1, 6),
            "mean_topk_distance_reranked": round(reranked_mean, 6),
            "overlap_with_top5": round(overlap_rate, 4),
            "llm_relevance_scores": reranked["llm_relevance"].tolist()[:keep_k],
        })

        if per_row_diag:
            diagnostics.append({
                "row_index": idx,
                "original_top5_pmc_ids": all_articles.head(keep_k)["pmc_id"].tolist(),
                "reranked_top5_pmc_ids": kept["pmc_id"].tolist(),
                "relevance_scores_all": reranked["llm_relevance"].tolist(),
                "distances_all": reranked["score"].tolist(),
            })

    results_df = df.copy()
    for col in ["triage_RAG_raw", "triage_RAG", "top1_distance_original",
                 "top1_distance_reranked", "mean_topk_distance_reranked",
                 "overlap_with_top5"]:
        results_df[col] = [r[col] for r in results]

    csv_path = run_dir / f"{label}_{ts}.csv"
    results_df.to_csv(csv_path, index=False)

    gt_col = find_gt_column(results_df, None)
    metrics = evaluate(results_df[gt_col], results_df["triage_RAG"], label=label)

    cost = _compute_run_cost(pricing, total_prompt_tok, total_comp_tok, total_think_tok)
    mean_overlap = results_df["overlap_with_top5"].mean()

    sidecar = {
        "experiment": "E10", "label": label,
        "retrieve_k": retrieve_k, "keep_k": keep_k,
        "mean_overlap_rate": round(float(mean_overlap), 4),
        "n_rows": len(df), "model": model_id,
        "context_chars": context_chars,
        "metrics": metrics, "cost_usd": cost,
        "prompt_tokens": total_prompt_tok,
        "completion_tokens": total_comp_tok,
        "thinking_tokens": total_think_tok,
    }
    sidecar_path = csv_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    if per_row_diag and diagnostics:
        diag_path = csv_path.with_suffix("").with_suffix(".diagnostics.jsonl")
        with open(diag_path, "w") as f:
            for d in diagnostics:
                f.write(json.dumps(d, default=str) + "\n")

    print(f"Saved: {csv_path.name}")
    print(f"  κ={metrics.get('quadratic_kappa', 'N/A'):.3f}  "
          f"exact={metrics.get('exact_accuracy', 'N/A'):.3f}  "
          f"mean_overlap={mean_overlap:.1%}  cost=${cost:.3f}")

    return [sidecar]


# ── E09: Multi-facet query decomposition ─────────────────────────────────

def _build_facet_queries(case: dict) -> list[str]:
    """Decompose a patient case into 3 facet queries."""
    hpi = str(case.get("HPI", "")).strip()
    chief = str(case.get("chiefcomplaint", "")).strip()
    patient_info = str(case.get("patient_info", "")).strip()
    past_med = str(case.get("past_medical", "")).strip()
    vitals = str(case.get("initial_vitals", "")).strip()
    pain = str(case.get("pain", "")).strip()

    facet_1 = f"{chief} {hpi}".strip()
    facet_2 = f"{past_med} {patient_info}".strip() if past_med else patient_info
    facet_3 = f"{vitals} {pain}".strip() if vitals else pain

    return [f for f in [facet_1, facet_2, facet_3] if f]


def run_e09(df: pd.DataFrame, model_id: str, pricing: dict | None,
            context_chars: int, top_k: int, per_facet_k: int,
            run_dir: Path, ts: str, per_row_diag: bool,
            prompt_template: str = "default") -> list[dict]:
    """E09 — Multi-facet: 3 queries per case, merge, deduplicate, keep top-5."""
    prompts = get_prompt_template(prompt_template)
    label = f"E09_multifacet_k{top_k}"
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    results = []
    diagnostics = []
    total_prompt_tok = 0
    total_comp_tok = 0
    total_think_tok = 0

    for idx in tqdm(range(len(df)), desc=label):
        row = df.iloc[idx]
        case = row.to_dict()
        facets = _build_facet_queries(case)

        all_hits = {}
        for fq in facets:
            articles = search_pubmed_articles(fq, top_k=per_facet_k)
            for _, art_row in articles.iterrows():
                pmc_id = art_row["pmc_id"]
                dist = float(art_row["score"])
                if pmc_id not in all_hits or dist < all_hits[pmc_id]["score"]:
                    all_hits[pmc_id] = art_row.to_dict()
                    all_hits[pmc_id]["score"] = dist

        merged = pd.DataFrame(list(all_hits.values()))
        if not merged.empty:
            merged = merged.sort_values("score").head(top_k).reset_index(drop=True)

        single_query = _concat_fields(case)
        single_articles = search_pubmed_articles(single_query, top_k=top_k)
        single_ids = set(single_articles["pmc_id"]) if not single_articles.empty else set()
        merged_ids = set(merged["pmc_id"]) if not merged.empty else set()
        overlap = len(single_ids & merged_ids) / top_k if top_k > 0 else 0

        top1_dist = float(merged["score"].iloc[0]) if len(merged) > 0 else 999.0
        mean_dist = float(merged["score"].mean()) if len(merged) > 0 else 999.0

        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = str(case.get("initial_vitals", ""))

        context = build_context_block_stripped(merged, context_chars)
        prompt = prompts["rag"].format(
            context=context, hpi=hpi,
            patient_info=patient_info, initial_vitals=initial_vitals,
        )

        try:
            resp = _llm_call(prompt, model_id,
                             json_schema=TriagePrediction.model_json_schema())
            raw_text = resp.text
            usage = _extract_usage(resp)
            total_prompt_tok += usage["prompt"]
            total_comp_tok += usage["completion"]
            total_think_tok += usage["thinking"]
        except Exception as e:
            raw_text = None
            print(f"WARNING row {idx}: {e}")

        parsed = parse_triage(raw_text)
        results.append({
            "triage_RAG_raw": raw_text,
            "triage_RAG": parsed,
            "top1_distance": round(top1_dist, 6),
            "mean_topk_distance": round(mean_dist, 6),
            "n_facets": len(facets),
            "n_unique_articles": len(all_hits),
            "overlap_with_single": round(overlap, 4),
        })

        if per_row_diag:
            diagnostics.append({
                "row_index": idx,
                "facet_queries": facets,
                "merged_pmc_ids": merged["pmc_id"].tolist() if not merged.empty else [],
                "single_pmc_ids": single_articles["pmc_id"].tolist() if not single_articles.empty else [],
            })

    results_df = df.copy()
    for col in ["triage_RAG_raw", "triage_RAG", "top1_distance",
                 "mean_topk_distance", "n_facets", "n_unique_articles",
                 "overlap_with_single"]:
        results_df[col] = [r[col] for r in results]

    csv_path = run_dir / f"{label}_{ts}.csv"
    results_df.to_csv(csv_path, index=False)

    gt_col = find_gt_column(results_df, None)
    metrics = evaluate(results_df[gt_col], results_df["triage_RAG"], label=label)

    cost = _compute_run_cost(pricing, total_prompt_tok, total_comp_tok, total_think_tok)
    mean_overlap = results_df["overlap_with_single"].mean()

    sidecar = {
        "experiment": "E09", "label": label,
        "per_facet_k": per_facet_k, "final_k": top_k,
        "mean_unique_articles": float(results_df["n_unique_articles"].mean()),
        "mean_overlap_with_single": round(float(mean_overlap), 4),
        "n_rows": len(df), "model": model_id,
        "context_chars": context_chars,
        "metrics": metrics, "cost_usd": cost,
        "prompt_tokens": total_prompt_tok,
        "completion_tokens": total_comp_tok,
        "thinking_tokens": total_think_tok,
    }
    sidecar_path = csv_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    if per_row_diag and diagnostics:
        diag_path = csv_path.with_suffix("").with_suffix(".diagnostics.jsonl")
        with open(diag_path, "w") as f:
            for d in diagnostics:
                f.write(json.dumps(d, default=str) + "\n")

    print(f"Saved: {csv_path.name}")
    print(f"  κ={metrics.get('quadratic_kappa', 'N/A'):.3f}  "
          f"exact={metrics.get('exact_accuracy', 'N/A'):.3f}  "
          f"mean_overlap_single={mean_overlap:.1%}  cost=${cost:.3f}")

    return [sidecar]


# ── E12: Adaptive re-querying ────────────────────────────────────────────

_REQUERY_PROMPT = (
    "The following medical articles were retrieved for a patient case but "
    "may not be relevant to emergency triage. Extract 3-5 key medical terms "
    "from the patient presentation that should appear in relevant emergency "
    "medicine literature for triage decision support. "
    "Output ONLY the search terms separated by spaces, nothing else.\n\n"
    "Patient presentation:\n{patient_text}\n\n"
    "Search terms:"
)


def run_e12(df: pd.DataFrame, model_id: str, pricing: dict | None,
            context_chars: int, top_k: int,
            requery_threshold: float, fallback_threshold: float,
            run_dir: Path, ts: str, per_row_diag: bool,
            prompt_template: str = "default") -> list[dict]:
    """E12 — Adaptive re-query: rewrite + re-retrieve on poor initial results."""
    prompts = get_prompt_template(prompt_template)
    label = f"E12_adaptive_rq{requery_threshold:.2f}_fb{fallback_threshold:.2f}"
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    results = []
    diagnostics = []
    total_prompt_tok = 0
    total_comp_tok = 0
    total_think_tok = 0
    requery_count = 0
    fallback_count = 0

    for idx in tqdm(range(len(df)), desc=label):
        row = df.iloc[idx]
        case = row.to_dict()
        query_text = _concat_fields(case)

        articles = search_pubmed_articles(query_text, top_k=top_k)
        initial_mean = float(articles["score"].mean()) if len(articles) > 0 else 999.0
        requeried = False
        requery_text = None

        if initial_mean > requery_threshold:
            requery_count += 1
            hpi = str(case.get("HPI", ""))[:500]
            patient_info = str(case.get("patient_info", ""))[:200]
            patient_text = f"{hpi} {patient_info}".strip()

            prompt_rq = _REQUERY_PROMPT.format(patient_text=patient_text)
            try:
                resp_rq = _llm_call(prompt_rq, model_id, temperature=0.0)
                requery_text = (resp_rq.text or "").strip()
                usage_rq = _extract_usage(resp_rq)
                total_prompt_tok += usage_rq["prompt"]
                total_comp_tok += usage_rq["completion"]
                total_think_tok += usage_rq["thinking"]

                if requery_text:
                    new_articles = search_pubmed_articles(requery_text, top_k=top_k)
                    all_hits = {}
                    for _, art_row in articles.iterrows():
                        all_hits[art_row["pmc_id"]] = art_row.to_dict()
                    for _, art_row in new_articles.iterrows():
                        pmc_id = art_row["pmc_id"]
                        dist = float(art_row["score"])
                        if pmc_id not in all_hits or dist < all_hits[pmc_id]["score"]:
                            all_hits[pmc_id] = art_row.to_dict()
                            all_hits[pmc_id]["score"] = dist
                    merged = pd.DataFrame(list(all_hits.values()))
                    articles = merged.sort_values("score").head(top_k).reset_index(drop=True)
                    requeried = True
            except Exception as e:
                print(f"WARNING requery row {idx}: {e}")

        final_mean = float(articles["score"].mean()) if len(articles) > 0 else 999.0
        top1_dist = float(articles["score"].iloc[0]) if len(articles) > 0 else 999.0

        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = str(case.get("initial_vitals", ""))

        use_rag = final_mean <= fallback_threshold
        if use_rag:
            context = build_context_block_stripped(articles, context_chars)
            prompt = prompts["rag"].format(
                context=context, hpi=hpi,
                patient_info=patient_info, initial_vitals=initial_vitals,
            )
        else:
            fallback_count += 1
            prompt = prompts["llm"].format(
                hpi=hpi, patient_info=patient_info,
                initial_vitals=initial_vitals,
            )

        try:
            resp = _llm_call(prompt, model_id,
                             json_schema=TriagePrediction.model_json_schema())
            raw_text = resp.text
            usage = _extract_usage(resp)
            total_prompt_tok += usage["prompt"]
            total_comp_tok += usage["completion"]
            total_think_tok += usage["thinking"]
        except Exception as e:
            raw_text = None
            print(f"WARNING row {idx}: {e}")

        parsed = parse_triage(raw_text)
        results.append({
            "triage_RAG_raw": raw_text,
            "triage_RAG": parsed,
            "top1_distance": round(top1_dist, 6),
            "mean_topk_distance": round(final_mean, 6),
            "initial_mean_distance": round(initial_mean, 6),
            "requeried": requeried,
            "used_rag": use_rag,
            "requery_text": requery_text,
        })

        if per_row_diag:
            diagnostics.append({
                "row_index": idx, "requeried": requeried,
                "initial_mean": initial_mean, "final_mean": final_mean,
                "requery_text": requery_text, "used_rag": use_rag,
            })

    results_df = df.copy()
    for col in ["triage_RAG_raw", "triage_RAG", "top1_distance",
                 "mean_topk_distance", "initial_mean_distance",
                 "requeried", "used_rag"]:
        results_df[col] = [r[col] for r in results]

    csv_path = run_dir / f"{label}_{ts}.csv"
    results_df.to_csv(csv_path, index=False)

    gt_col = find_gt_column(results_df, None)
    metrics = evaluate(results_df[gt_col], results_df["triage_RAG"], label=label)

    cost = _compute_run_cost(pricing, total_prompt_tok, total_comp_tok, total_think_tok)

    sidecar = {
        "experiment": "E12", "label": label,
        "requery_threshold": requery_threshold,
        "fallback_threshold": fallback_threshold,
        "requery_count": requery_count,
        "requery_rate": round(requery_count / len(df), 4),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(df), 4),
        "n_rows": len(df), "model": model_id,
        "context_chars": context_chars, "top_k": top_k,
        "metrics": metrics, "cost_usd": cost,
        "prompt_tokens": total_prompt_tok,
        "completion_tokens": total_comp_tok,
        "thinking_tokens": total_think_tok,
    }
    sidecar_path = csv_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    if per_row_diag and diagnostics:
        diag_path = csv_path.with_suffix("").with_suffix(".diagnostics.jsonl")
        with open(diag_path, "w") as f:
            for d in diagnostics:
                f.write(json.dumps(d, default=str) + "\n")

    print(f"Saved: {csv_path.name}")
    print(f"  κ={metrics.get('quadratic_kappa', 'N/A'):.3f}  "
          f"exact={metrics.get('exact_accuracy', 'N/A'):.3f}  "
          f"requery_rate={requery_count/len(df):.1%}  "
          f"fallback_rate={fallback_count/len(df):.1%}  cost=${cost:.3f}")

    return [sidecar]


# ── Helpers ───────────────────────────────────────────────────────────────

def _compute_run_cost(pricing: dict | None, prompt_tok: int,
                      comp_tok: int, think_tok: int) -> float:
    if not pricing:
        return 0.0
    think_rate = pricing.get("thinking") or pricing.get("output") or 0
    return round(
        prompt_tok * (pricing.get("input") or 0)
        + comp_tok * (pricing.get("output") or 0)
        + think_tok * think_rate, 6
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Retrieval quality experiments E09–E12.")
    p.add_argument("--experiment", required=True,
                   choices=["E09", "E10", "E11", "E12", "all"],
                   help="Which experiment to run")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--n-rows", type=int, default=150)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--context-chars", type=int, default=8000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--per-row-diagnostics", action="store_true")
    p.add_argument("--prompt-template", type=str, default="default",
                   help="Prompt template name from PROMPT_TEMPLATES (default: 'default')")
    # E10 specific
    p.add_argument("--retrieve-k", type=int, default=20,
                   help="E10: articles to retrieve before reranking")
    p.add_argument("--keep-k", type=int, default=5,
                   help="E10: articles to keep after reranking")
    # E11 specific
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.25, 0.27, 0.30],
                   help="E11: distance thresholds to sweep")
    # E12 specific
    p.add_argument("--requery-threshold", type=float, default=0.28,
                   help="E12: mean_topk_distance above which to re-query")
    p.add_argument("--fallback-threshold", type=float, default=0.30,
                   help="E12: mean_topk_distance above which to fall back to LLM-only")
    # E09 specific
    p.add_argument("--per-facet-k", type=int, default=10,
                   help="E09: articles per facet query")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    config.setup_clients()
    model_id = args.model or config.MODEL_ID
    pricing, pricing_source = fetch_vertex_pricing(model_id)
    context_chars = args.context_chars

    df = pd.read_csv(args.input, nrows=args.n_rows)
    print(f"Loaded {len(df)} rows from {args.input.name}")

    experiments = [args.experiment] if args.experiment != "all" else ["E11", "E10", "E09", "E12"]
    all_results = []

    for exp in experiments:
        run_dir = RUNS_DIR / f"{exp}_retrieval_quality"
        run_dir.mkdir(parents=True, exist_ok=True)

        if exp == "E11":
            summaries = run_e11(
                df, model_id, pricing, context_chars, args.top_k,
                args.thresholds, run_dir, ts, args.per_row_diagnostics,
                prompt_template=args.prompt_template,
            )
            all_results.extend(summaries)

        elif exp == "E10":
            summaries = run_e10(
                df, model_id, pricing, context_chars,
                args.retrieve_k, args.keep_k,
                run_dir, ts, args.per_row_diagnostics,
                prompt_template=args.prompt_template,
            )
            all_results.extend(summaries)

        elif exp == "E09":
            summaries = run_e09(
                df, model_id, pricing, context_chars, args.top_k,
                args.per_facet_k, run_dir, ts, args.per_row_diagnostics,
                prompt_template=args.prompt_template,
            )
            all_results.extend(summaries)

        elif exp == "E12":
            summaries = run_e12(
                df, model_id, pricing, context_chars, args.top_k,
                args.requery_threshold, args.fallback_threshold,
                run_dir, ts, args.per_row_diagnostics,
                prompt_template=args.prompt_template,
            )
            all_results.extend(summaries)

    # Print combined summary
    print(f"\n{'='*70}")
    print("COMBINED SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        m = r.get("metrics", {})
        print(f"  {r['label']:40s}  κ={m.get('quadratic_kappa', 0):.3f}  "
              f"exact={m.get('exact_accuracy', 0):.3f}  "
              f"range={m.get('range_accuracy', 0):.3f}  "
              f"cost=${r.get('cost_usd', 0):.3f}")


if __name__ == "__main__":
    main()
