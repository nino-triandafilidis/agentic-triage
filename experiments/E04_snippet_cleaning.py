"""E04: Snippet cleaning — test if stripping PMC front-matter improves RAG triage.

PMC article_text begins with XML front-matter (journal names, ISSNs, DOIs,
authors, dates, license boilerplate).  With 2000-char storage truncation,
65% of articles are *entirely* front-matter — the LLM sees zero clinical content.
This experiment tests two cleaning strategies:

  strip:  Strip front-matter when a section marker is found.
          Articles without a marker are passed as-is.  All articles included.

  skip:   Only include articles where a section marker was found (front-matter
          successfully stripped).  Articles without a marker are excluded.
          If no articles pass the filter for a row, fall back to LLM-only prompt.

Both conditions share the same retrieval pass (one embedding + FAISS lookup per
row), then build two different prompts and call generation twice.

Controls (matching E02a / E00.5v2 baselines):
  - Input: dev_tune.csv, 150 rows
  - Model: gemini-2.5-flash
  - Retrieval: concat query, top_k=5, FAISS (cached)
  - context_chars: 2000 (applied after stripping)
  - Generation: temperature=0, JSON schema output

Note: article_text is pre-truncated to 2000 chars at storage level.
After stripping, remaining content is typically 200-1500 chars.

Usage:
    .venv/bin/python experiments/E04_snippet_cleaning.py
    .venv/bin/python experiments/E04_snippet_cleaning.py --n-rows 10  # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config
from experiments import tracking
from experiments.eval_triage import evaluate, find_gt_column, log_result
from src.rag.query_agents import get_agent
from src.rag.retrieval import search_pubmed_articles
from src.rag.triage_core import (
    GENERATION_TEMPERATURE,
    TRIAGE_PROMPT_LLM_ONLY,
    TRIAGE_PROMPT_WITH_RAG,
    fetch_vertex_pricing,
    parse_triage,
)
from src.schemas import TriagePrediction

CACHE_PATH = ROOT / "data" / "cache" / "retrieval_cache.parquet"
RUNS_DIR = ROOT / "data" / "runs"


# ── Front-matter stripping ───────────────────────────────────────────────

_SECTION_MARKERS = [
    "\n==== Body",
    "\nIntroduction\n",
    "\nINTRODUCTION\n",
    "\nBackground\n",
    "\nBACKGROUND\n",
    "\nAbstract\n",
    "\nABSTRACT\n",
    "\nMethods\n",
    "\nMETHODS\n",
    "\nObjective\n",
    "\nOBJECTIVE\n",
    "\nObjectives\n",
    "\nOBJECTIVES\n",
    "\nPurpose\n",
    "\nPURPOSE\n",
    "\nSimple Summary\n",
    "\nResults\n",
    "\nRESULTS\n",
]


def strip_front_matter(text: str) -> tuple[str, str | None]:
    """Strip PMC XML front-matter by finding the earliest section marker.

    Returns (cleaned_text, matched_marker_name) or (original_text, None).
    """
    best_pos = len(text)
    best_marker = None
    for marker in _SECTION_MARKERS:
        pos = text.find(marker)
        if 0 < pos < best_pos:
            best_pos = pos
            best_marker = marker.strip()
    if best_marker is not None:
        return text[best_pos:], best_marker
    return text, None


def _preprocess_articles(articles: pd.DataFrame):
    """Strip front-matter for each article once.  Returns list of dicts."""
    processed = []
    for _, art in articles.iterrows():
        raw = str(art["article_text"])
        cleaned, marker = strip_front_matter(raw)
        processed.append({
            "pmc_id": str(art.get("pmc_id", "")),
            "pmid": art.get("pmid"),
            "raw": raw,
            "cleaned": cleaned,
            "marker": marker,
            "chars_stripped": len(raw) - len(cleaned) if marker else 0,
        })
    return processed


# ── Context builders (operate on preprocessed list) ──────────────────────

def _build_context_strip(processed: list[dict], context_chars: int):
    """Strip condition: use cleaned text for all articles."""
    parts = []
    for i, art in enumerate(processed):
        snippet = art["cleaned"][:context_chars]
        parts.append(f"--- Article {i + 1} (PMID {art['pmid']}) ---\n{snippet}")
    return "\n\n".join(parts) if parts else "No relevant articles found."


def _build_context_skip(processed: list[dict], context_chars: int):
    """Skip condition: only include articles where a marker was found.

    Returns (context_block | None, n_articles_included).
    None signals no usable articles → LLM-only fallback.
    """
    parts = []
    for art in processed:
        if art["marker"] is not None:
            snippet = art["cleaned"][:context_chars]
            parts.append(
                f"--- Article {len(parts) + 1} (PMID {art['pmid']}) ---\n{snippet}"
            )
    if not parts:
        return None, 0
    return "\n\n".join(parts), len(parts)


# ── Generation ───────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception(
        lambda e: "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
    ),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _generate(prompt: str, model_id: str):
    from src.llm import generate as llm_generate
    from src.llm.types import GenerationConfig

    return llm_generate(
        prompt,
        model_id=model_id,
        config=GenerationConfig(
            temperature=GENERATION_TEMPERATURE,
            response_mime_type="application/json",
            response_json_schema=TriagePrediction.model_json_schema(),
        ),
    )


def _call_and_parse(prompt: str, model_id: str):
    """Call LLM, extract tokens and parsed ESI. Returns dict."""
    raw_text = None
    prompt_tokens = 0
    completion_tokens = 0
    thinking_tokens = 0
    try:
        resp = _generate(prompt, model_id)
        raw_text = resp.text
        prompt_tokens = resp.prompt_tokens
        completion_tokens = resp.completion_tokens
        thinking_tokens = resp.thinking_tokens
    except Exception as e:
        print(f"  WARNING generation: {type(e).__name__}: {e}")
    return {
        "raw": raw_text,
        "parsed": parse_triage(raw_text),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
    }


# ── Main loop ────────────────────────────────────────────────────────────

def run_single_condition(
    condition: str,
    df_input: pd.DataFrame,
    retrieval_cache: dict[str, pd.DataFrame],
    model_id: str,
    top_k: int,
    context_chars: int,
    pricing: dict | None,
) -> tuple[list[dict], list[dict], dict]:
    """Run one condition: retrieve, build context, generate.

    Returns (results_per_row, diagnostics_per_row, aggregate_stats).
    """
    agent = get_agent("concat")

    results: list[dict] = []
    diagnostics: list[dict] = []
    tok = {"prompt": 0, "completion": 0, "thinking": 0}
    marker_counter: Counter = Counter()
    n_marker_hit = 0
    n_marker_miss = 0
    n_llm_fallback = 0

    for idx in tqdm(range(len(df_input)), desc=condition):
        row = df_input.iloc[idx]
        case = row.to_dict()

        # 1. Build query (concat — same as E02a)
        query_result = agent.build_query(case, {})
        qhash = query_result.query_hash

        # 2. Retrieve
        if qhash in retrieval_cache:
            articles = retrieval_cache[qhash].head(top_k)
        else:
            articles = search_pubmed_articles(
                query_result.query_text, top_k=top_k
            )

        # 3. Preprocess articles (strip front-matter)
        processed = _preprocess_articles(articles) if not articles.empty else []

        # Track marker stats
        for art in processed:
            if art["marker"]:
                n_marker_hit += 1
                marker_counter[art["marker"]] += 1
            else:
                n_marker_miss += 1

        # 4. Build context + prompt
        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = str(case.get("initial_vitals", ""))
        use_rag = True
        n_articles_used = len(processed)

        if condition == "strip":
            if processed:
                context_block = _build_context_strip(processed, context_chars)
            else:
                context_block = "No relevant articles found."
            article_stats = [
                {
                    "rank": i + 1,
                    "pmc_id": a["pmc_id"],
                    "marker": a["marker"],
                    "chars_stripped": a["chars_stripped"],
                    "snippet_len": len(a["cleaned"][:context_chars]),
                }
                for i, a in enumerate(processed)
            ]
        elif condition == "skip":
            context_block, n_articles_used = _build_context_skip(
                processed, context_chars
            )
            if context_block is None:
                use_rag = False
                n_llm_fallback += 1
            article_stats = [
                {
                    "rank": i + 1,
                    "pmc_id": a["pmc_id"],
                    "marker": a["marker"],
                    "included": a["marker"] is not None,
                    "chars_stripped": a["chars_stripped"],
                }
                for i, a in enumerate(processed)
            ]
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if use_rag:
            prompt = TRIAGE_PROMPT_WITH_RAG.format(
                context=context_block,
                hpi=hpi,
                patient_info=patient_info,
                initial_vitals=initial_vitals,
            )
        else:
            prompt = TRIAGE_PROMPT_LLM_ONLY.format(
                hpi=hpi,
                patient_info=patient_info,
                initial_vitals=initial_vitals,
            )

        # 5. Generate
        gen = _call_and_parse(prompt, model_id)
        tok["prompt"] += gen["prompt_tokens"]
        tok["completion"] += gen["completion_tokens"]
        tok["thinking"] += gen["thinking_tokens"]

        results.append({
            "triage_RAG_raw": gen["raw"],
            "triage_RAG": gen["parsed"],
            "triage_query_hash": qhash,
            "n_articles_used": n_articles_used,
            "used_rag_prompt": use_rag,
        })
        diagnostics.append({
            "row_index": idx,
            "stay_id": int(row.get("stay_id", 0)),
            "ground_truth": (
                int(row.get("triage", 0))
                if pd.notna(row.get("triage"))
                else None
            ),
            "predicted": gen["parsed"],
            "n_articles_used": n_articles_used,
            "used_rag_prompt": use_rag,
            "article_stats": article_stats,
        })

    # Aggregate stats
    total_articles = n_marker_hit + n_marker_miss
    cost_usd = None
    if pricing:
        thinking_rate = pricing.get("thinking") or pricing.get("output") or 0
        cost_usd = (
            tok["prompt"] * (pricing.get("input") or 0)
            + tok["completion"] * (pricing.get("output") or 0)
            + tok["thinking"] * thinking_rate
        )

    agg_stats = {
        "total_articles_seen": total_articles,
        "marker_hit": n_marker_hit,
        "marker_miss": n_marker_miss,
        "marker_hit_pct": (
            round(100 * n_marker_hit / total_articles, 1) if total_articles else 0
        ),
        "marker_breakdown": dict(marker_counter.most_common()),
        "n_llm_fallback": n_llm_fallback,
        "prompt_tokens": tok["prompt"],
        "completion_tokens": tok["completion"],
        "thinking_tokens": tok["thinking"],
        "cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
    }

    return results, diagnostics, agg_stats


# ── CLI + Main ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="E04: Snippet cleaning experiment")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "splits" / "dev_tune.csv",
    )
    p.add_argument("--n-rows", type=int, default=150)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--context-chars", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument(
        "--condition",
        choices=["strip", "skip", "both"],
        default="both",
        help="Run one condition or both sequentially (default: both)",
    )
    p.add_argument(
        "--output-prefix", type=str, default="E04_snippet_cleaning"
    )
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or tracking.generate_run_id(prefix="E04")
    context_chars = args.context_chars or config.RETRIEVAL_CONTEXT_CHARS

    config.setup_clients()
    model_id = args.model or config.MODEL_ID
    pricing, pricing_source = fetch_vertex_pricing(model_id)

    # Load retrieval cache
    retrieval_cache: dict[str, pd.DataFrame] = {}
    if CACHE_PATH.exists():
        raw_cache = pd.read_parquet(CACHE_PATH)
        raw_cache = raw_cache[raw_cache["top_k"] >= args.top_k]
        if "context_chars" in raw_cache.columns:
            preferred = raw_cache[raw_cache["context_chars"] >= context_chars]
            if not preferred.empty:
                raw_cache = preferred
        cols = ["pmc_id", "pmid", "distance", "article_text", "article_citation"]
        for qhash, group in raw_cache.groupby("query_hash"):
            retrieval_cache[qhash] = (
                group[cols].head(args.top_k).reset_index(drop=True)
            )
        print(f"Retrieval cache loaded: {len(retrieval_cache)} queries")
    else:
        print("WARNING: No retrieval cache — retrieval will be called per row")

    df_input = pd.read_csv(args.input, nrows=args.n_rows)
    print(f"Loaded {len(df_input)} rows from {args.input.name}")
    input_sha256 = tracking.sha256_file(args.input)

    git_state = tracking.collect_git_state(ROOT)
    code_files, _ = tracking.compute_file_hashes(
        ROOT, tracking.DEFAULT_TRACKED_FILES
    )

    run_dir = RUNS_DIR / args.output_prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    conditions = (
        ["strip", "skip"] if args.condition == "both" else [args.condition]
    )

    # ── Run each condition ────────────────────────────────────────
    all_summaries = []

    for condition in conditions:
        results, diagnostics, agg_stats = run_single_condition(
            condition=condition,
            df_input=df_input,
            retrieval_cache=retrieval_cache,
            model_id=model_id,
            top_k=args.top_k,
            context_chars=context_chars,
            pricing=pricing,
        )

        print(f"\n{'=' * 60}")
        print(f"Condition: {condition}")
        print(f"{'=' * 60}")

        # Build output DataFrame
        pred_col = "triage_RAG"
        results_df = df_input.copy()
        results_df["triage_RAG_raw"] = [r["triage_RAG_raw"] for r in results]
        results_df[pred_col] = [r["triage_RAG"] for r in results]
        results_df["n_articles_used"] = [r["n_articles_used"] for r in results]
        results_df["used_rag_prompt"] = [r["used_rag_prompt"] for r in results]

        csv_path = run_dir / f"{args.output_prefix}_{condition}_{ts}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path.name}")

        # Evaluate
        metrics: dict = {}
        try:
            gt_col = find_gt_column(results_df, None)
            metrics = evaluate(
                results_df[gt_col], results_df[pred_col], label=condition
            )
        except (ValueError, KeyError):
            print("WARNING: Could not evaluate (no ground-truth column found)")

        # Print snippet stats
        print(f"\n── Snippet stats ({condition}) ──")
        print(f"  Total articles seen:    {agg_stats['total_articles_seen']}")
        print(
            f"  Marker hit:             {agg_stats['marker_hit']} "
            f"({agg_stats['marker_hit_pct']}%)"
        )
        print(f"  Marker miss:            {agg_stats['marker_miss']}")
        if condition == "skip":
            print(
                f"  LLM-only fallback rows: {agg_stats['n_llm_fallback']}"
                f"/{len(df_input)}"
            )
        print("  Marker breakdown:")
        for marker, count in sorted(
            agg_stats["marker_breakdown"].items(), key=lambda x: -x[1]
        ):
            print(f"    {count:4d}  {marker}")
        if agg_stats["cost_usd"] is not None:
            print(f"  Cost: ${agg_stats['cost_usd']:.4f}")

        # Save diagnostics
        diag_path = csv_path.with_suffix(".diagnostics.jsonl")
        with open(diag_path, "w") as f:
            for d in diagnostics:
                f.write(json.dumps(d, default=str) + "\n")
        print(f"Diagnostics: {diag_path.name}")

        # Sidecar manifest
        sidecar = {
            "manifest_version": "1.2",
            "run_id": run_id,
            "experiment": "E04_snippet_cleaning",
            "condition": condition,
            "timestamp_utc": ts,
            "model": model_id,
            "mode": "rag",
            "top_k": args.top_k,
            "context_chars": context_chars,
            "retrieval_backend": config.RETRIEVAL_BACKEND,
            "query_agent_name": "concat",
            "n_rows": len(df_input),
            "input_file": args.input.name,
            "input_sha256": input_sha256,
            "snippet_stats": agg_stats,
            "prompt_tokens": agg_stats["prompt_tokens"],
            "completion_tokens": agg_stats["completion_tokens"],
            "thinking_tokens": agg_stats["thinking_tokens"],
            "cost_usd": agg_stats["cost_usd"],
            "pricing_source": pricing_source,
            "git_hash": git_state["git_hash"],
            "git_dirty": git_state["git_dirty"],
            "code_files": code_files,
            "code_fingerprint": tracking.sha256_json(code_files),
            "prompt_hashes": {
                "rag": tracking.sha256_text(TRIAGE_PROMPT_WITH_RAG),
                "llm": tracking.sha256_text(TRIAGE_PROMPT_LLM_ONLY),
            },
        }
        sidecar_path = csv_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2))

        # Log to experiment_log.csv
        log_result({
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "run_id": run_id,
            "base_run_id": None,
            "diff_id": None,
            "diff_file": None,
            "split": "dev_tune",
            "model": model_id,
            "mode": f"rag_{condition}",
            "top_k": args.top_k,
            "retrieval_backend": config.RETRIEVAL_BACKEND,
            "n_evaluated": metrics.get("n_evaluated"),
            "n_skipped": metrics.get("n_skipped"),
            "pred_col": pred_col,
            "exact_accuracy": metrics.get("exact_accuracy"),
            "adj1_accuracy": metrics.get("adj1_accuracy"),
            "range_accuracy": metrics.get("range_accuracy"),
            "mae": metrics.get("mae"),
            "quadratic_kappa": metrics.get("quadratic_kappa"),
            "git_hash": git_state["git_hash"],
            "git_dirty": git_state["git_dirty"],
            "code_fingerprint": tracking.sha256_json(code_files),
            "prompt_hash_rag": tracking.sha256_text(TRIAGE_PROMPT_WITH_RAG),
            "prompt_hash_llm": tracking.sha256_text(TRIAGE_PROMPT_LLM_ONLY),
            "input_sha256": input_sha256,
            "source_file": args.input.name,
            "input_file": csv_path.name,
            "prompt_tokens": agg_stats["prompt_tokens"],
            "completion_tokens": agg_stats["completion_tokens"],
            "cost_usd": agg_stats["cost_usd"],
            "pricing_source": pricing_source,
            "notes": f"E04 {condition}: snippet front-matter cleaning",
        })

        all_summaries.append({
            "condition": condition,
            **{
                k: v
                for k, v in metrics.items()
                if k
                in [
                    "exact_accuracy",
                    "adj1_accuracy",
                    "range_accuracy",
                    "mae",
                    "quadratic_kappa",
                ]
            },
            "marker_hit_pct": agg_stats["marker_hit_pct"],
            "n_llm_fallback": agg_stats["n_llm_fallback"],
            "cost_usd": agg_stats["cost_usd"],
        })

    # Final comparison table
    print(f"\n{'=' * 60}")
    print("E04 Summary")
    print(f"{'=' * 60}")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))
    print("\nBaseline comparisons (same 150 dev_tune rows):")
    print("  E02a concat (with front-matter):  56.0% exact, κ=0.2315")
    print("  E00.5v2 LLM-only (no articles):   58.67% exact, κ=0.3375")

    summary_path = run_dir / f"{args.output_prefix}_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path.name}")


if __name__ == "__main__":
    main()
