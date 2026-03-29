"""Unified triage experiment runner.

Supports both RAG (with query strategy sweep) and LLM-only modes.
Produces eval-compatible CSVs + sidecar manifests.

Usage:
    # RAG mode (default): sweep query strategies
    .venv/bin/python experiments/query_strategy_sweep.py \
        --input data/splits/scratch.csv --n-rows 5 \
        --strategies concat hpi_only --top-k 5 \
        --output-prefix sweep_test

    # LLM-only mode: no retrieval
    .venv/bin/python experiments/query_strategy_sweep.py \
        --mode llm --input data/splits/dev_tune.csv --n-rows 150 \
        --output-prefix E00.5v3_llm_baseline
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Prevent indefinite SSL read hangs — applies to all sockets in this process
socket.setdefaulttimeout(300)

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
logger = logging.getLogger(__name__)

import src.config as config
from experiments import tracking
from experiments.eval_triage import evaluate
from src.rag.agentic_pipeline import (
    CostBudget,
    CostLimitExceeded,
    TriageAgenticPipeline,
)
from src.rag.query_agents import (
    RewriteResult,
    get_agent,
)
from src.rag.esi_handbook import load_esi_handbook_prefix
from src.rag.query_agents import _concat_fields
from src.rag.retrieval import search_pubmed_articles
from src.rag.triage_core import (
    TRIAGE_PROMPT_WITH_RAG,
    TRIAGE_PROMPT_LLM_ONLY,
    TRIAGE_PROMPT_HANDBOOK_RAG,
    TRIAGE_PROMPT_HANDBOOK_ONLY,
    PROMPT_TEMPLATES,
    get_prompt_template,
    FALLBACK_PRICING,
    fetch_vertex_pricing,
)

CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "data" / "runs"

# Columns extracted from result dicts into the output CSV
_RESULT_COLS = [
    "triage_query_agent", "triage_query_text", "triage_query_hash",
    "n_articles_retrieved", "top1_distance", "mean_topk_distance",
    "top1_score", "mean_topk_score", "score_type",
    "tool_call_count",
    "case_bank_call_count", "pmc_call_count", "tool_call_sequence",
    "uncertainty_gate_result",
    "multi_role_nurse_acuity", "multi_role_nurse_confidence",
    "multi_role_reader_signal",
    "critic_used_case_bank", "critic_used_pmc",
    "pmc_trigger_reason", "final_changed_from_nurse",
    "reranker_selected_count", "reranker_selected_indices",
    "fallback_reason",
    "boundary_review_changed", "vitals_guardrail_fired", "vitals_flags",
    "distance_gate_action", "gate_top1_distance",
    "error",
]


def _write_partial_csv(
    df_input: pd.DataFrame,
    results: list[dict],
    pred_col: str,
    is_llm_only: bool,
    path: Path,
) -> None:
    """Write completed rows to a partial CSV for crash recovery."""
    n = len(results)
    raw_col = "triage_LLM_raw" if is_llm_only else "triage_RAG_raw"
    partial = df_input.iloc[:n].copy()
    partial[raw_col] = [r.get("triage_RAG_raw") for r in results]
    partial[pred_col] = [r.get("triage_RAG") for r in results]
    for col in _RESULT_COLS:
        partial[col] = [r.get(col) for r in results]
    partial.to_csv(path, index=False)


class APITimeoutError(Exception):
    """Raised by SIGALRM when an API call exceeds the wall-clock deadline."""


def _sigalrm_handler(signum, frame):
    raise APITimeoutError("API call exceeded wall-clock timeout (SIGALRM)")


def _empty_result(error: str | None = None) -> dict[str, object]:
    return {
        "triage_RAG": None,
        "triage_RAG_raw": None,
        "error": error,
    }


def _reset_provider_client(model_id: str) -> None:
    """Clear the cached client for the provider serving *model_id*."""
    from src.llm.registry import resolve_provider

    provider = resolve_provider(model_id)
    if hasattr(provider, "_client"):
        provider._client = None


def _is_timeout_error(exc: Exception) -> bool:
    err_name = type(exc).__name__.lower()
    return (
        isinstance(exc, APITimeoutError)
        or "timeout" in err_name
        or "readtimeout" in err_name
    )


@contextlib.contextmanager
def _installed_sigalrm_handler(handler):
    prev_handler = signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, prev_handler)


def _validate_args(args: argparse.Namespace) -> None:
    if args.prompt_template == "multi_role":
        if args.mode == "llm":
            sys.exit("FATAL: prompt_template='multi_role' requires --mode rag.")
        if args.top_k <= 0:
            sys.exit("FATAL: prompt_template='multi_role' requires --top-k > 0.")
        if args.distance_gate is not None:
            sys.exit("FATAL: prompt_template='multi_role' does not support --distance-gate.")
        if args.uncertainty_gate:
            sys.exit("FATAL: prompt_template='multi_role' does not support --uncertainty-gate.")
        if args.handbook_prefix:
            sys.exit("FATAL: prompt_template='multi_role' does not support --handbook-prefix.")
        return

    if args.prompt_template in {
        "tool_use_pmc",
        "tool_use_dual",
        "two_role_case_bank",
        "two_role_case_bank_pmc_conditional",
    }:
        if args.mode != "llm":
            sys.exit(f"FATAL: prompt_template='{args.prompt_template}' requires --mode llm.")
        if args.distance_gate is not None:
            sys.exit(f"FATAL: prompt_template='{args.prompt_template}' does not support --distance-gate.")
        if args.uncertainty_gate:
            sys.exit(f"FATAL: prompt_template='{args.prompt_template}' does not support --uncertainty-gate.")
        if args.handbook_prefix:
            sys.exit(f"FATAL: prompt_template='{args.prompt_template}' does not support --handbook-prefix.")
        return

    if args.prompt_template == "three_role_rerank_critic":
        if args.mode == "llm":
            sys.exit("FATAL: prompt_template='three_role_rerank_critic' requires --mode rag.")
        if args.top_k <= 0:
            sys.exit("FATAL: prompt_template='three_role_rerank_critic' requires --top-k > 0.")
        if args.distance_gate is not None:
            sys.exit("FATAL: prompt_template='three_role_rerank_critic' does not support --distance-gate.")
        if args.uncertainty_gate:
            sys.exit("FATAL: prompt_template='three_role_rerank_critic' does not support --uncertainty-gate.")
        if args.handbook_prefix:
            sys.exit("FATAL: prompt_template='three_role_rerank_critic' does not support --handbook-prefix.")
        return


def _cache_path_for_backend(backend: str) -> Path:
    """Return the cache parquet path for a given retrieval backend."""
    if backend == "faiss":
        return CACHE_DIR / "retrieval_cache.parquet"
    return CACHE_DIR / f"retrieval_cache_{backend}.parquet"


# ── Rewrite function factory ─────────────────────────────────────────────

def make_rewrite_fn(model_id: str):
    """Return a rewrite callable that dispatches via the LLM interface."""
    from src.llm import generate as llm_generate
    from src.llm.types import GenerationConfig

    @retry(
        retry=retry_if_exception(lambda e: "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _rewrite(raw_query: str, instruction: str) -> RewriteResult:
        resp = llm_generate(
            instruction,
            model_id=model_id,
            config=GenerationConfig(temperature=0.0),
        )
        return RewriteResult(
            text=resp.text or "",
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            model_id=model_id,
            thinking_tokens=resp.thinking_tokens,
        )

    return _rewrite


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unified triage experiment runner (RAG + LLM-only).")
    p.add_argument("--mode", choices=["rag", "llm"], default="rag",
                   help="rag = retrieval-augmented (default), llm = LLM-only (no retrieval)")
    p.add_argument("--input", type=Path, default=ROOT / "data" / "splits" / "scratch.csv")
    p.add_argument("--n-rows", type=int, default=10)
    p.add_argument("--skip-rows", type=int, default=0,
                   help="Skip first N rows of input before selecting --n-rows")
    p.add_argument("--top-k", type=int, default=5,
                   help="Articles to retrieve per row (ignored in llm mode)")
    p.add_argument("--context-chars", type=int, default=None,
                   help="Chars per article (default: config.RETRIEVAL_CONTEXT_CHARS)")
    p.add_argument("--strategies", nargs="+", default=["concat"],
                   choices=["concat", "hpi_only", "rewrite"],
                   help="Query strategies to sweep (ignored in llm mode)")
    p.add_argument("--rewrite-model", type=str, default="gemini-2.5-flash",
                   help="Model for rewrite agent (default: gemini-2.5-flash)")
    p.add_argument("--model", type=str, default=None,
                   help="Generation model (default: config.MODEL_ID)")
    p.add_argument("--fast-model", type=str, default=None,
                   help="Fast model for reranking/assessment (default: same as --model)")
    p.add_argument("--max-generation-cost-usd", type=float, default=None)
    p.add_argument("--max-rewrite-cost-usd", type=float, default=None)
    p.add_argument("--max-total-cost-usd", type=float, default=None)
    p.add_argument("--output-prefix", type=str, default="sweep")
    p.add_argument("--handbook-prefix", action="store_true",
                   help="Prepend full ESI Handbook text as system-level context")
    p.add_argument("--handbook-path", type=Path, default=None,
                   help="Custom path to ESI handbook text file (default: data/corpus/esi_handbook_clean.txt)")
    p.add_argument("--prompt-template", type=str, default="default",
                   help="Prompt template to use (default: 'default'). "
                   f"Available: {sorted(PROMPT_TEMPLATES.keys())}")
    p.add_argument("--distance-gate", type=float, default=None,
                   help="Distance gate threshold (float). When set in RAG mode, "
                   "rows whose top-1 retrieval distance exceeds this value fall "
                   "back to LLM-only (no context). Ignored in llm mode.")
    p.add_argument("--rerank", action="store_true",
                   help="Enable LLM-based reranking (over-retrieve 4x, rerank, keep top_k)")
    p.add_argument("--uncertainty-gate", action="store_true",
                   help="Enable uncertainty-gated two-pass: skip retrieval when fast model is confident")
    p.add_argument("--max-pmc-calls-per-row", type=int, default=None,
                   help="Limit PMC tool calls per row in tool-use modes (default: unlimited)")
    p.add_argument("--boundary-review", action="store_true",
                   help="Enable ESI-2/3 boundary self-review pass (second LLM call "
                   "when first pass predicts ESI-2 or ESI-3)")
    p.add_argument("--vitals-guardrail", action="store_true",
                   help="Enable LLM-based vital-sign danger zone check. "
                   "Uptriages ESI-3+ to ESI-2 when vitals exceed age-adjusted "
                   "ESI handbook thresholds (Decision Point D).")
    p.add_argument("--multi-role-evidence", choices=["pmc", "pmc_case_bank"], default="pmc",
                   help="Evidence source for prompt_template=multi_role (default: pmc)")
    p.add_argument("--requests-per-minute", type=float, default=None,
                   help="Throttle: max rows per minute (sleep between rows). "
                   "Note: each row may issue 1-3 API calls (rewrite + generation + "
                   "retry), so set this conservatively below the provider RPM limit.")
    p.add_argument("--thinking-level", type=str, default=None,
                   choices=["LOW", "MEDIUM", "HIGH"],
                   help="Set thinking level for Gemini 3 models (LOW/MEDIUM/HIGH). "
                   "Default: unset (model default, i.e. full thinking).")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Write _partial.csv every N rows (0 to disable). Default: 50.")
    p.add_argument("--per-row-diagnostics", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    is_llm_only = args.mode == "llm"
    run_id = args.run_id or tracking.generate_run_id(prefix="sweep")
    context_chars = args.context_chars or config.RETRIEVAL_CONTEXT_CHARS

    # LLM-only overrides
    if is_llm_only:
        args.top_k = 0
        args.strategies = ["noop"]
        print("LLM-only mode: retrieval disabled, using noop query agent.")

    _validate_args(args)

    config.setup_clients()
    model_id = args.model or config.MODEL_ID
    fast_model_id = args.fast_model or model_id

    # Auth preflight — fail immediately if credentials are missing
    from src.llm.registry import resolve_provider
    provider = resolve_provider(model_id)
    if not provider.is_available():
        sys.exit(f"FATAL: provider for {model_id!r} is not available "
                 f"(missing SDK or credentials). Fix before running.")

    gen_pricing, gen_pricing_source = fetch_vertex_pricing(model_id)

    fast_pricing = gen_pricing
    fast_pricing_source = gen_pricing_source
    needs_fast_model = (
        args.rerank
        or args.uncertainty_gate
        or args.vitals_guardrail
        or args.prompt_template in {
            "multi_role",
            "two_role_case_bank",
            "two_role_case_bank_pmc_conditional",
            "three_role_rerank_critic",
        }
    )
    if needs_fast_model and fast_model_id != model_id:
        fast_provider = resolve_provider(fast_model_id)
        if not fast_provider.is_available():
            sys.exit(f"FATAL: fast provider for {fast_model_id!r} is not available "
                     f"(missing SDK or credentials). Fix before running.")
        fast_pricing, fast_pricing_source = fetch_vertex_pricing(fast_model_id)

    # Rewrite pricing (may differ from generation model)
    rewrite_pricing = None
    rewrite_pricing_source = "unavailable"
    if "rewrite" in args.strategies:
        rewrite_provider = resolve_provider(args.rewrite_model)
        if not rewrite_provider.is_available():
            sys.exit(f"FATAL: rewrite provider for {args.rewrite_model!r} is not available "
                     f"(missing SDK or credentials). Fix before running.")
        rewrite_pricing, rewrite_pricing_source = fetch_vertex_pricing(args.rewrite_model)

    # Load retrieval cache (skipped in LLM-only mode)
    retrieval_cache: dict[str, pd.DataFrame] = {}
    cache_path = _cache_path_for_backend(config.RETRIEVAL_BACKEND)
    if not is_llm_only and cache_path.exists():
        raw_cache = pd.read_parquet(cache_path)
        raw_cache = raw_cache[raw_cache["top_k"] >= args.top_k]
        if "context_chars" in raw_cache.columns:
            preferred = raw_cache[raw_cache["context_chars"] >= context_chars]
            if not preferred.empty:
                raw_cache = preferred
        from src.rag.retrieval import RESULT_COLS
        available = [c for c in RESULT_COLS if c in raw_cache.columns]
        # Legacy caches may have "distance" instead of "score"
        if "distance" in raw_cache.columns and "distance" not in available:
            available.append("distance")
        for qhash, group in raw_cache.groupby("query_hash"):
            retrieval_cache[qhash] = group[available].head(args.top_k).reset_index(drop=True)
        print(f"Retrieval cache loaded ({config.RETRIEVAL_BACKEND}): {len(retrieval_cache)} queries")
    elif not is_llm_only:
        print("No retrieval cache — retrieval will be called per row")

    # Load input data
    if args.skip_rows > 0:
        df_input = pd.read_csv(args.input, skiprows=range(1, args.skip_rows + 1),
                               nrows=args.n_rows)
        print(f"Loaded {len(df_input)} rows from {args.input.name} "
              f"(skipped first {args.skip_rows})")
    else:
        df_input = pd.read_csv(args.input, nrows=args.n_rows)
        print(f"Loaded {len(df_input)} rows from {args.input.name}")
    input_sha256 = tracking.sha256_file(args.input)

    # Build rewrite fn once
    rewrite_fn = None
    if "rewrite" in args.strategies:
        rewrite_fn = make_rewrite_fn(args.rewrite_model)

    # Load ESI Handbook for system-level prefix (optional)
    handbook_text = None
    if args.handbook_prefix:
        handbook_text = load_esi_handbook_prefix(args.handbook_path)
        print(f"ESI Handbook loaded: {len(handbook_text):,} chars")

    # Git state + code hashes (shared across strategies)
    git_state = tracking.collect_git_state(ROOT)
    code_files, missing_code_files = tracking.compute_file_hashes(ROOT, tracking.DEFAULT_TRACKED_FILES)

    run_dir = RUNS_DIR / args.output_prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for strategy in args.strategies:
        label = "LLM-only" if is_llm_only else strategy
        print(f"\n{'='*60}")
        print(f"Strategy: {label}")
        print(f"{'='*60}")

        agent = get_agent(strategy, rewrite_fn=rewrite_fn)
        cost_budget = CostBudget(
            max_generation_usd=args.max_generation_cost_usd,
            max_rewrite_usd=args.max_rewrite_cost_usd,
            max_total_usd=args.max_total_cost_usd,
        )

        pipeline = TriageAgenticPipeline(
            query_agent=agent,
            top_k=args.top_k,
            context_chars=context_chars,
            model_id=model_id,
            pricing=gen_pricing,
            fast_pricing=fast_pricing,
            cost_budget=cost_budget,
            retrieval_cache=retrieval_cache,
            run_id=run_id,
            handbook_text=handbook_text,
            prompt_template=args.prompt_template,
            fast_model_id=fast_model_id,
            rerank=args.rerank,
            uncertainty_gate=args.uncertainty_gate,
            max_pmc_calls_per_row=args.max_pmc_calls_per_row,
            multi_role_evidence=args.multi_role_evidence,
            boundary_review=args.boundary_review,
            vitals_guardrail=args.vitals_guardrail,
            thinking_level=args.thinking_level,
        )

        # Distance gate: create a second LLM-only pipeline for fallback rows
        distance_gate = args.distance_gate
        use_distance_gate = distance_gate is not None and not is_llm_only
        llm_fallback_pipeline = None
        if use_distance_gate:
            from src.rag.query_agents import NoopQueryAgent
            llm_fallback_pipeline = TriageAgenticPipeline(
                query_agent=NoopQueryAgent(),
                top_k=0,
                context_chars=context_chars,
                model_id=model_id,
                pricing=gen_pricing,
                fast_pricing=fast_pricing,
                cost_budget=cost_budget,
                retrieval_cache={},
                run_id=run_id,
                handbook_text=handbook_text,
                prompt_template=args.prompt_template,
                fast_model_id=fast_model_id,
                max_pmc_calls_per_row=args.max_pmc_calls_per_row,
                multi_role_evidence=args.multi_role_evidence,
                boundary_review=args.boundary_review,
                vitals_guardrail=args.vitals_guardrail,
                thinking_level=args.thinking_level,
            )
            print(f"Distance gate enabled: threshold={distance_gate:.4f}")

        results = []
        diagnostics = []
        cost_limit_hit = False
        gate_fallback_count = 0

        pred_col = "triage_LLM" if is_llm_only else "triage_RAG"
        api_timeout_secs = 300

        # RPM throttle: minimum seconds between rows
        rpm_delay = 60.0 / args.requests_per_minute if args.requests_per_minute else 0
        last_request_time = 0.0

        # Incremental checkpoint path
        checkpoint_n = args.checkpoint_every
        csv_label = "llm" if is_llm_only else strategy
        partial_csv_path = run_dir / f"{args.output_prefix}_{csv_label}_{ts}_partial.csv"

        try:
            with _installed_sigalrm_handler(_sigalrm_handler):
                for idx in tqdm(range(len(df_input)), desc=label):
                    if rpm_delay and idx > 0:
                        elapsed = time.monotonic() - last_request_time
                        if elapsed < rpm_delay:
                            time.sleep(rpm_delay - elapsed)

                    row = df_input.iloc[idx]
                    case = row.to_dict()
                    gated_to_llm = False
                    gate_top1_distance = None
                    active_pipeline = pipeline

                    if use_distance_gate:
                        query_text = _concat_fields(case)
                        articles = search_pubmed_articles(query_text, top_k=args.top_k)
                        dist_col = "distance" if "distance" in articles.columns else "score"
                        gate_top1_distance = float(articles[dist_col].iloc[0]) if len(articles) > 0 else 999.0
                        if gate_top1_distance > distance_gate:
                            gated_to_llm = True
                            gate_fallback_count += 1
                            active_pipeline = llm_fallback_pipeline

                    try:
                        last_request_time = time.monotonic()
                        signal.alarm(api_timeout_secs)
                        result = active_pipeline.predict_one(case)
                        signal.alarm(0)
                    except Exception as e:
                        signal.alarm(0)
                        err_name = type(e).__name__
                        if _is_timeout_error(e):
                            logger.warning("Timeout on row %d (%s), retrying with fresh connection...", idx, err_name)
                            try:
                                _reset_provider_client(model_id)
                            except Exception:
                                pass
                            if rpm_delay:
                                time.sleep(rpm_delay)
                            try:
                                signal.alarm(api_timeout_secs)
                                result = active_pipeline.predict_one(case)
                                signal.alarm(0)
                            except Exception as e2:
                                signal.alarm(0)
                                logger.error("Retry also failed on row %d: %s", idx, e2)
                                result = _empty_result(str(e2))
                        else:
                            logger.error("Error on row %d: %s: %s", idx, err_name, e)
                            result = _empty_result(str(e))

                    if use_distance_gate:
                        result["distance_gate_action"] = "fallback_llm" if gated_to_llm else "used_rag"
                    if gate_top1_distance is not None:
                        result["gate_top1_distance"] = round(gate_top1_distance, 6)
                    results.append(result)

                    if idx == 0 and result.get("triage_RAG") is None:
                        err_msg = result.get("error", "null prediction with no error message")
                        raise RuntimeError(
                            f"First row failed: {err_msg}. "
                            f"Aborting to avoid wasting {len(df_input) - 1} more API calls."
                        )

                    if checkpoint_n and (idx + 1) % checkpoint_n == 0:
                        _write_partial_csv(df_input, results, pred_col, is_llm_only, partial_csv_path)
                        logger.info("Checkpoint: %d/%d rows → %s", idx + 1, len(df_input), partial_csv_path.name)

                    if args.per_row_diagnostics:
                        diag = {
                            "row_index": idx,
                            **result,
                            "events": [e.to_dict() for e in active_pipeline.events[-2:]],
                        }
                        diagnostics.append(diag)

                    if result.get("cost_limit_hit"):
                        cost_limit_hit = True
                        print(f"\nCost limit hit at row {idx}: {result['cost_limit_hit']}")
                        # Fill remaining rows with None
                        for remaining_idx in range(idx + 1, len(df_input)):
                            results.append({
                                "triage_RAG_raw": None,
                                "triage_RAG": None,
                                "triage_query_agent": strategy,
                                "triage_query_text": None,
                                "triage_query_hash": None,
                                "n_articles_retrieved": 0,
                                "cost_limit_hit": result["cost_limit_hit"],
                            })
                        break
        except Exception:
            if results:
                _write_partial_csv(df_input, results, pred_col, is_llm_only, partial_csv_path)
                logger.exception(
                    "Fatal error after %d/%d rows; emergency checkpoint written to %s",
                    len(results), len(df_input), partial_csv_path.name,
                )
            raise

        if use_distance_gate:
            print(f"Distance gate summary: {gate_fallback_count}/{len(df_input)} rows "
                  f"fell back to LLM-only ({gate_fallback_count/len(df_input):.1%})")

        # Build output DataFrame
        raw_col = "triage_LLM_raw" if is_llm_only else "triage_RAG_raw"
        results_df = df_input.copy()
        results_df[raw_col] = [r.get("triage_RAG_raw") for r in results]
        results_df[pred_col] = [r.get("triage_RAG") for r in results]
        for col in _RESULT_COLS:
            results_df[col] = [r.get(col) for r in results]

        # Save final CSV and remove partial checkpoint
        csv_path = run_dir / f"{args.output_prefix}_{csv_label}_{ts}.csv"
        results_df.to_csv(csv_path, index=False)
        if partial_csv_path.exists():
            partial_csv_path.unlink()
        print(f"Saved: {csv_path.name}")

        # Evaluate
        from experiments.eval_triage import find_gt_column
        try:
            gt_col = find_gt_column(results_df, None)
            metrics = evaluate(results_df[gt_col], results_df[pred_col], label=f"{label}")
        except (ValueError, KeyError):
            metrics = {}
            print(f"WARNING: Could not evaluate {strategy} (no ground-truth column found)")

        # Compute rewrite prompt hash if applicable
        rewrite_prompt_hash = None
        if strategy == "rewrite":
            from src.rag.query_agents import RewriteQueryAgent
            rewrite_prompt_hash = tracking.sha256_text(RewriteQueryAgent._REWRITE_INSTRUCTION)

        # Sidecar manifest
        prompts = get_prompt_template(args.prompt_template)
        if args.prompt_template == "multi_role":
            prompt_hashes = {
                "nurse": tracking.sha256_text(prompts["nurse"]),
                "reader": tracking.sha256_text(prompts["reader"]),
                "adjudicator": tracking.sha256_text(prompts["adjudicator"]),
            }
        elif args.prompt_template in {"two_role_case_bank", "two_role_case_bank_pmc_conditional"}:
            prompt_hashes = {
                "nurse": tracking.sha256_text(prompts["nurse"]),
                "critic": tracking.sha256_text(prompts["critic"]),
            }
        elif args.prompt_template == "three_role_rerank_critic":
            prompt_hashes = {
                "nurse": tracking.sha256_text(prompts["nurse"]),
                "reranker": tracking.sha256_text(prompts["reranker"]),
                "critic": tracking.sha256_text(prompts["critic"]),
            }
        elif handbook_text:
            prompt_hashes = {
                "rag": tracking.sha256_text(prompts["handbook_rag"]),
                "llm": tracking.sha256_text(prompts["handbook_only"]),
            }
        else:
            prompt_hashes = {
                "rag": tracking.sha256_text(prompts["rag"]),
                "llm": tracking.sha256_text(prompts["llm"]),
            }

        # Aggregate costs from both pipelines when distance gating is active
        generation_cost = round(pipeline.generation_cost_usd, 6)
        rewrite_cost = round(pipeline.rewrite_cost_usd, 6)
        rerank_cost = round(pipeline.rerank_cost_usd, 6)
        assessment_cost = round(pipeline.assessment_cost_usd, 6)
        if llm_fallback_pipeline is not None:
            generation_cost = round(generation_cost + llm_fallback_pipeline.generation_cost_usd, 6)
        total_cost = round(
            generation_cost + rewrite_cost + rerank_cost + assessment_cost,
            6,
        )

        total_prompt_tokens = (
            pipeline.generation_prompt_tokens + pipeline.rewrite_prompt_tokens
            + pipeline.rerank_prompt_tokens + pipeline.assessment_prompt_tokens
        )
        total_completion_tokens = (
            pipeline.generation_completion_tokens + pipeline.rewrite_completion_tokens
            + pipeline.rerank_completion_tokens + pipeline.assessment_completion_tokens
        )
        total_thinking_tokens = (
            pipeline.generation_thinking_tokens + pipeline.rewrite_thinking_tokens
            + pipeline.rerank_thinking_tokens + pipeline.assessment_thinking_tokens
        )
        if llm_fallback_pipeline is not None:
            total_prompt_tokens += llm_fallback_pipeline.generation_prompt_tokens
            total_completion_tokens += llm_fallback_pipeline.generation_completion_tokens
            total_thinking_tokens += llm_fallback_pipeline.generation_thinking_tokens
        n_evaluated = len(df_input)
        cost_per_row = round(total_cost / n_evaluated, 6) if gen_pricing and n_evaluated > 0 else None

        sidecar = {
            "manifest_version": "1.2",
            "run_id": run_id,
            "timestamp_start_utc": ts,
            "timestamp_end_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": model_id,
            "mode": args.mode,
            "top_k": args.top_k,
            "retrieval_backend": "none" if is_llm_only else config.RETRIEVAL_BACKEND,
            "n_rows": len(df_input),
            "skip_rows": args.skip_rows,
            "context_chars": context_chars,
            "rerank_chars": pipeline.rerank_chars,
            "input_file": args.input.name,
            "input_path": str(args.input.resolve()),
            "input_sha256": input_sha256,
            "output_file": csv_path.name,
            "output_path": str(csv_path.resolve()),
            "temperature": 0.0,
            "prompt_hashes": prompt_hashes,
            "code_files": code_files,
            "missing_code_files": missing_code_files,
            "code_fingerprint": tracking.sha256_json(code_files),
            "git_hash": git_state["git_hash"],
            "git_dirty": git_state["git_dirty"],
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "thinking_tokens": total_thinking_tokens,
            "cost_usd": total_cost if gen_pricing else None,
            "pricing_source": gen_pricing_source,
            "prompt_template": args.prompt_template,
            "multi_role_evidence": args.multi_role_evidence if args.prompt_template == "multi_role" else None,
            "max_pmc_calls_per_row": args.max_pmc_calls_per_row,
            "query_agent_name": strategy,
            "generation_cost_usd": generation_cost if gen_pricing else None,
            "rewrite_cost_usd": rewrite_cost if rewrite_pricing or strategy != "rewrite" else None,
            "rewrite_model": args.rewrite_model if strategy == "rewrite" else None,
            "rewrite_prompt_hash": rewrite_prompt_hash,
            "rewrite_pricing_source": rewrite_pricing_source if strategy == "rewrite" else None,
            "fast_model": fast_model_id,
            "fast_pricing_source": fast_pricing_source,
            "rerank_cost_usd": rerank_cost if gen_pricing else None,
            "assessment_cost_usd": assessment_cost if gen_pricing else None,
            "handbook_prefix": handbook_text is not None,
            "handbook_chars": len(handbook_text) if handbook_text else 0,
            "handbook_sha256": tracking.sha256_text(handbook_text) if handbook_text else None,
            "metrics": {
                "exact_accuracy": metrics.get("exact_accuracy"),
                "range_accuracy": metrics.get("range_accuracy"),
                "quadratic_kappa": metrics.get("quadratic_kappa"),
                "mae": metrics.get("mae"),
                "under_triage_rate": metrics.get("under_triage_rate"),
                "over_triage_rate": metrics.get("over_triage_rate"),
                "parse_fail_rate": metrics.get("parse_fail_rate"),
            },
            "cost_per_row_usd": cost_per_row,
            "distance_gate": distance_gate,
            "distance_gate_fallback_count": gate_fallback_count if use_distance_gate else None,
            "distance_gate_fallback_rate": round(gate_fallback_count / len(df_input), 4) if use_distance_gate else None,
            "boundary_review": args.boundary_review,
            "vitals_guardrail": args.vitals_guardrail,
            "execution_events": [e.to_dict() for e in pipeline.events],
            "cost_limit_hit": cost_limit_hit,
        }

        sidecar_path = csv_path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"Metadata: {sidecar_path.name}")

        # Diagnostics
        if args.per_row_diagnostics and diagnostics:
            diag_path = csv_path.with_suffix("").with_suffix(".diagnostics.jsonl")
            with open(diag_path, "w") as f:
                for d in diagnostics:
                    f.write(json.dumps(d, default=str) + "\n")
            print(f"Diagnostics: {diag_path.name}")

        # Summary row
        summary_rows.append({
            "strategy": label,
            "prompt_template": args.prompt_template,
            "multi_role_evidence": args.multi_role_evidence if args.prompt_template == "multi_role" else None,
            "handbook_prefix": handbook_text is not None,
            "n_rows": n_evaluated,
            "exact_accuracy": metrics.get("exact_accuracy"),
            "adj1_accuracy": metrics.get("adj1_accuracy"),
            "range_accuracy": metrics.get("range_accuracy"),
            "mae": metrics.get("mae"),
            "quadratic_kappa": metrics.get("quadratic_kappa"),
            "under_triage_rate": metrics.get("under_triage_rate"),
            "over_triage_rate": metrics.get("over_triage_rate"),
            "parse_fail_rate": metrics.get("parse_fail_rate"),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "thinking_tokens": total_thinking_tokens,
            "generation_cost_usd": generation_cost if gen_pricing else None,
            "rewrite_cost_usd": rewrite_cost if rewrite_pricing or strategy != "rewrite" else None,
            "total_cost_usd": total_cost if gen_pricing else None,
            "cost_per_row_usd": cost_per_row,
            "cost_limit_hit": cost_limit_hit,
        })

    # Write aggregate summary
    if len(summary_rows) > 1:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = run_dir / f"{args.output_prefix}_summary_{ts}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary: {summary_path.name}")
        print(summary_df.to_string(index=False))
    elif summary_rows:
        print(f"\nSingle strategy result:")
        for k, v in summary_rows[0].items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
