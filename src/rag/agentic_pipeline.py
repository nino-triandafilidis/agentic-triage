"""Linear agentic orchestrator for triage RAG.

Chain: query_agent -> retrieval -> generation (hardcoded linear, no DAG).
Tracks per-instance token counts and costs with split budgets for
generation vs rewrite.  Appends ExecutionEvent traces for each step.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

import src.config as config
from src.llm import generate as llm_generate, generate_with_tools as llm_generate_with_tools
from src.llm.types import GenerationConfig, ToolDefinition
from src.rag.query_agents import (
    ExecutionEvent,
    QueryAgent,
    QueryResult,
    _new_step_id,
)
from src.rag.text_cleaning import prepare_article_excerpt
from src.rag.retrieval import search_pubmed_articles
from src.rag.triage_core import (
    GENERATION_TEMPERATURE,
    TRIAGE_PROMPT_LLM_ONLY,
    TRIAGE_PROMPT_WITH_RAG,
    TRIAGE_PROMPT_HANDBOOK_RAG,
    TRIAGE_PROMPT_HANDBOOK_ONLY,
    BATCH_RERANK_PROMPT,
    UNCERTAINTY_ASSESSMENT_PROMPT,
    UNCERTAINTY_ASSESSMENT_SCHEMA,
    ESI_BOUNDARY_REVIEW_PROMPT,
    VITALS_DANGER_ZONE_PROMPT,
    PROMPT_TEMPLATES,
    format_vitals_for_prompt,
    get_prompt_template,
    build_context_block,
    parse_triage,
)
from src.schemas import NurseDraft, ReaderBrief, RerankerSelection, TriagePrediction, VitalsDangerZone


# ── Cost budget ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class CostBudget:
    """Split cost caps (USD). None = unlimited."""
    max_generation_usd: float | None = None
    max_rewrite_usd: float | None = None
    max_total_usd: float | None = None


class CostLimitExceeded(RuntimeError):
    """Raised when a cost budget is exhausted."""
    def __init__(self, category: str, current: float, limit: float):
        self.category = category
        self.current = current
        self.limit = limit
        super().__init__(
            f"Cost limit exceeded for '{category}': "
            f"${current:.6f} >= ${limit:.6f}"
        )


# ── Tool-use config ──────────────────────────────────────────────────
MAX_TOOL_TURNS = 3
TOOL_USE_PROMPT_TEMPLATES = {"tool_use", "tool_use_rag", "tool_use_pmc", "tool_use_dual", "tool_use_decision_tree"}
TWO_ROLE_PROMPT_TEMPLATES = {"two_role_case_bank", "two_role_case_bank_pmc_conditional"}

# ── Pipeline ─────────────────────────────────────────────────────────────

class TriageAgenticPipeline:
    """Linear orchestrator: query_agent -> retrieval -> generation."""

    def __init__(
        self,
        query_agent: QueryAgent,
        top_k: int,
        context_chars: int,
        model_id: str,
        pricing: dict[str, float] | None,
        fast_pricing: dict[str, float] | None = None,
        rerank_chars: int = 1500,
        cost_budget: CostBudget | None = None,
        retrieval_cache: dict[str, pd.DataFrame] | None = None,
        run_id: str = "",
        handbook_text: str | None = None,
        prompt_template: str = "default",
        fast_model_id: str | None = None,
        rerank: bool = False,
        uncertainty_gate: bool = False,
        max_pmc_calls_per_row: int | None = None,
        multi_role_evidence: str = "pmc",
        boundary_review: bool = False,
        vitals_guardrail: bool = False,
        thinking_level: str | None = None,
    ):
        self.query_agent = query_agent
        self.top_k = top_k
        self.context_chars = context_chars
        self.rerank_chars = rerank_chars
        self.model_id = model_id
        self.fast_model_id = fast_model_id or model_id
        self.pricing = pricing
        self.fast_pricing = fast_pricing or pricing
        self.cost_budget = cost_budget or CostBudget()
        self.retrieval_cache = retrieval_cache or {}
        self.run_id = run_id
        self.handbook_text = handbook_text
        self.prompt_template_name = prompt_template
        self._prompts = get_prompt_template(prompt_template)
        self.rerank = rerank
        self.uncertainty_gate = uncertainty_gate
        self.max_pmc_calls_per_row = max_pmc_calls_per_row
        self.multi_role_evidence = multi_role_evidence
        self.boundary_review = boundary_review
        self.vitals_guardrail = vitals_guardrail
        self.thinking_level = thinking_level

        if prompt_template == "three_role_rerank_critic" and top_k <= 0:
            raise ValueError("prompt_template='three_role_rerank_critic' requires top_k > 0")

        # Validate case bank availability at startup (fail-fast)
        if prompt_template in (
            "tool_use",
            "tool_use_rag",
            "tool_use_dual",
            "tool_use_decision_tree",
            "two_role_case_bank",
            "two_role_case_bank_pmc_conditional",
            "three_role_rerank_critic",
        ):
            from src.rag.case_bank import get_case_bank
            get_case_bank()  # raises FileNotFoundError if missing

        # Per-instance accumulators
        self.generation_prompt_tokens = 0
        self.generation_completion_tokens = 0
        self.generation_thinking_tokens = 0
        self.rewrite_prompt_tokens = 0
        self.rewrite_completion_tokens = 0
        self.rewrite_thinking_tokens = 0
        self.generation_cost_usd = 0.0
        self.rewrite_cost_usd = 0.0
        # Rerank accumulators
        self.rerank_prompt_tokens = 0
        self.rerank_completion_tokens = 0
        self.rerank_thinking_tokens = 0
        self.rerank_cost_usd = 0.0
        # Assessment accumulators (uncertainty gate first-pass)
        self.assessment_prompt_tokens = 0
        self.assessment_completion_tokens = 0
        self.assessment_thinking_tokens = 0
        self.assessment_cost_usd = 0.0
        self.events: list[ExecutionEvent] = []
        # Tool-use accumulators (per predict_one call, reset each time)
        self.tool_call_count = 0
        self.tool_call_details: list[dict] = []

    @property
    def total_cost_usd(self) -> float:
        return (
            self.generation_cost_usd
            + self.rewrite_cost_usd
            + self.rerank_cost_usd
            + self.assessment_cost_usd
        )

    def _compute_cost(
        self, prompt_tokens: int, completion_tokens: int, thinking_tokens: int = 0,
    ) -> float:
        return self._compute_cost_with_pricing(
            self.pricing, prompt_tokens, completion_tokens, thinking_tokens,
        )

    @staticmethod
    def _compute_cost_with_pricing(
        pricing: dict[str, float] | None,
        prompt_tokens: int,
        completion_tokens: int,
        thinking_tokens: int = 0,
    ) -> float:
        if pricing is None:
            return 0.0
        thinking_rate = pricing.get("thinking") or pricing.get("output") or 0
        return (
            prompt_tokens * (pricing.get("input") or 0)
            + completion_tokens * (pricing.get("output") or 0)
            + thinking_tokens * thinking_rate
        )

    def _check_budget(self, category: str) -> None:
        """Raise CostLimitExceeded if the given category is over budget."""
        b = self.cost_budget
        if category == "rewrite" and b.max_rewrite_usd is not None:
            if self.rewrite_cost_usd >= b.max_rewrite_usd:
                raise CostLimitExceeded("rewrite", self.rewrite_cost_usd, b.max_rewrite_usd)
        if category == "generation" and b.max_generation_usd is not None:
            if self.generation_cost_usd >= b.max_generation_usd:
                raise CostLimitExceeded("generation", self.generation_cost_usd, b.max_generation_usd)
        if b.max_total_usd is not None:
            if self.total_cost_usd >= b.max_total_usd:
                raise CostLimitExceeded("total", self.total_cost_usd, b.max_total_usd)

    def _record_event(
        self,
        agent_name: str,
        agent_version: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        status: str,
        error: str | None,
        start_time: float,
        parent_step_id: str | None = None,
        thinking_tokens: int = 0,
    ) -> ExecutionEvent:
        ev = ExecutionEvent(
            run_id=self.run_id,
            step_id=_new_step_id(),
            parent_step_id=parent_step_id,
            agent_name=agent_name,
            agent_version=agent_version,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            cost_usd=round(cost_usd, 8),
            status=status,
            error=error,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_ms=int((time.monotonic() - start_time) * 1000),
        )
        self.events.append(ev)
        return ev

    @staticmethod
    def _base_result(
        *,
        query_result: QueryResult,
        qhash: str | None,
        n_articles: int,
        top1_distance: float | None = None,
        mean_topk_distance: float | None = None,
        top1_score: float | None = None,
        mean_topk_score: float | None = None,
        score_type: str | None = None,
        uncertainty_gate_result: str | None = None,
    ) -> dict[str, Any]:
        return {
            "triage_query_agent": query_result.agent_name,
            "triage_query_text": query_result.query_text,
            "triage_query_hash": qhash,
            "n_articles_retrieved": n_articles,
            "top1_distance": top1_distance,
            "mean_topk_distance": mean_topk_distance,
            "top1_score": top1_score,
            "mean_topk_score": mean_topk_score,
            "score_type": score_type,
            "cost_limit_hit": None,
            "uncertainty_gate_result": uncertainty_gate_result,
            "tool_call_count": None,
            "case_bank_call_count": None,
            "pmc_call_count": None,
            "tool_call_sequence": None,
            "tool_call_details": None,
            "multi_role_nurse_acuity": None,
            "multi_role_nurse_confidence": None,
            "multi_role_reader_signal": None,
            "critic_used_case_bank": None,
            "critic_used_pmc": None,
            "pmc_trigger_reason": None,
            "final_changed_from_nurse": None,
            "reranker_selected_count": None,
            "reranker_selected_indices": None,
            "fallback_reason": None,
            "boundary_review_changed": None,
            "vitals_guardrail_fired": None,
            "vitals_flags": None,
            "error": None,
        }

    @staticmethod
    def _tool_log_summary(tool_call_log: list[dict]) -> dict[str, Any]:
        tool_names = [str(tc.get("tool_name")) for tc in tool_call_log if tc.get("tool_name")]
        case_bank_count = sum(name == "search_esi_case_bank" for name in tool_names)
        pmc_count = sum(name == "search_pmc_articles" for name in tool_names)
        return {
            "critic_used_case_bank": case_bank_count > 0,
            "critic_used_pmc": pmc_count > 0,
            "case_bank_call_count": case_bank_count,
            "pmc_call_count": pmc_count,
            "tool_call_sequence": json.dumps(tool_names) if tool_names else None,
        }

    def _build_default_prediction_prompt(
        self,
        *,
        hpi: str,
        patient_info: str,
        initial_vitals: str,
        articles: pd.DataFrame,
    ) -> str:
        prompts = get_prompt_template("default")
        if self.top_k > 0:
            context_block = build_context_block(articles, self.context_chars)
            if self.handbook_text:
                return prompts["handbook_rag"].format(
                    esi_handbook=self.handbook_text,
                    context=context_block,
                    hpi=hpi,
                    patient_info=patient_info,
                    initial_vitals=initial_vitals,
                )
            return prompts["rag"].format(
                context=context_block,
                hpi=hpi,
                patient_info=patient_info,
                initial_vitals=initial_vitals,
            )

        if self.handbook_text:
            return prompts["handbook_only"].format(
                esi_handbook=self.handbook_text,
                hpi=hpi,
                patient_info=patient_info,
                initial_vitals=initial_vitals,
            )
        return prompts["llm"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
        )

    @staticmethod
    def _single_agent_pmc_trigger_reason(
        prompt_template_name: str,
        tool_summary: dict[str, Any],
    ) -> str | None:
        if not tool_summary.get("critic_used_pmc"):
            return "none"
        if prompt_template_name in {"tool_use_pmc", "tool_use_dual", "tool_use_rag"}:
            return "always_available"
        return None

    @retry(
        retry=retry_if_exception(lambda e: "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_model_generation(
        self,
        prompt: str,
        *,
        model_id: str,
        config: GenerationConfig,
    ):
        return llm_generate(
            prompt,
            model_id=model_id,
            config=config,
        )

    def _call_generation(self, prompt: str):
        return self._call_model_generation(
            prompt,
            model_id=self.model_id,
            config=GenerationConfig(
                temperature=GENERATION_TEMPERATURE,
                response_mime_type="application/json",
                response_json_schema=TriagePrediction.model_json_schema(),
                thinking_level=self.thinking_level,
            ),
        )

    def _call_fast_generation(self, prompt: str, **kwargs):
        """Call the fast model (used for reranking and uncertainty assessment)."""
        gen_config = kwargs.pop("config", None) or GenerationConfig(
            temperature=GENERATION_TEMPERATURE,
            thinking_disabled=self.no_thinking,
        )
        return self._call_model_generation(
            prompt,
            model_id=self.fast_model_id,
            config=gen_config,
        )

    def _rerank_articles(
        self, articles_df: pd.DataFrame, patient_summary: str,
    ) -> pd.DataFrame:
        """Batch-rerank articles using the fast model.

        Sends all article excerpts in one LLM call, parses relevance scores,
        and returns top-k articles sorted by relevance (ties broken by FAISS
        distance / original score).
        """
        if articles_df.empty:
            return articles_df

        rerank_start = time.monotonic()

        # Keep reranking cheaper than generation by using a smaller excerpt budget.
        parts = []
        for idx, (_, row) in enumerate(articles_df.iterrows()):
            text = prepare_article_excerpt(
                row.get("article_text"),
                self.rerank_chars,
                context_text=row.get("context_text"),
            )
            parts.append(f"[Article {idx}] (PMID {row.get('pmid', 'N/A')})\n{text}")
        articles_block = "\n\n".join(parts)

        prompt = BATCH_RERANK_PROMPT.format(
            patient_summary=patient_summary,
            articles_block=articles_block,
        )

        try:
            resp = self._call_fast_generation(prompt)
            raw_text = (resp.text or "").strip()

            # Track rerank tokens/cost
            self.rerank_prompt_tokens += resp.prompt_tokens
            self.rerank_completion_tokens += resp.completion_tokens
            self.rerank_thinking_tokens += resp.thinking_tokens
            rerank_cost = self._compute_cost_with_pricing(
                self.fast_pricing,
                resp.prompt_tokens, resp.completion_tokens, resp.thinking_tokens,
            )
            self.rerank_cost_usd += rerank_cost

            self._record_event(
                agent_name="rerank",
                agent_version=f"rerank/{self.fast_model_id}",
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=rerank_cost,
                status="success",
                error=None,
                start_time=rerank_start,
                thinking_tokens=resp.thinking_tokens,
            )

            # Parse JSON array of scores from response
            # Extract JSON array even if wrapped in markdown code block
            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                score_list = json.loads(json_match.group())
            else:
                score_list = json.loads(raw_text)

            # Map article_index -> score
            score_map: dict[int, int] = {}
            for entry in score_list:
                idx = entry.get("article_index")
                score = entry.get("score", 1)
                if idx is not None:
                    score_map[int(idx)] = max(1, min(5, int(score)))

            # Apply scores to DataFrame
            articles_df = articles_df.copy()
            articles_df["rerank_score"] = [
                score_map.get(i, 1) for i in range(len(articles_df))
            ]

        except Exception as e:
            # On failure, assign uniform scores (preserves original ranking)
            print(f"WARNING rerank: {type(e).__name__}: {e}")
            self._record_event(
                agent_name="rerank",
                agent_version=f"rerank/{self.fast_model_id}",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                status="error",
                error=f"{type(e).__name__}: {e}",
                start_time=rerank_start,
            )
            articles_df = articles_df.copy()
            articles_df["rerank_score"] = 1

        score_type = articles_df["score_type"].iloc[0] if "score_type" in articles_df.columns else None
        if score_type in {"bm25", "rrf"}:
            score_ascending = False
        else:
            score_ascending = True

        # Sort by rerank score first, then backend-native retrieval ordering.
        sort_cols = ["rerank_score"]
        ascending = [False]
        if "score" in articles_df.columns:
            sort_cols.append("score")
            ascending.append(score_ascending)
        articles_df = articles_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        # Keep top_k
        return articles_df.head(self.top_k).reset_index(drop=True)

    @retry(
        retry=retry_if_exception(lambda e: "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_generation_with_tools(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        schema_cls: type[BaseModel] = TriagePrediction,
        pricing: dict[str, float] | None = None,
        include_case_bank: bool = True,
        include_pmc: bool | None = None,
        require_pmc_once: bool = False,
    ) -> tuple[str | None, list[dict]]:
        """Run multi-turn tool-use loop. Returns (final_text, tool_call_log)."""
        from src.rag.case_bank import (
            CASE_BANK_TOOL_SCHEMA,
            format_tool_result,
            search_cases,
        )
        from src.rag.retrieval import (
            PMC_SEARCH_TOOL_SCHEMA,
            search_pmc_articles as _search_pmc,
        )

        active_model_id = model_id or self.model_id
        active_pricing = pricing or self.pricing

        def _build_tool_defs(include_case_bank_tool: bool, include_pmc_tool: bool) -> list[ToolDefinition]:
            defs: list[ToolDefinition] = []
            if include_case_bank_tool:
                defs.append(
                    ToolDefinition(
                        name=CASE_BANK_TOOL_SCHEMA["name"],
                        description=CASE_BANK_TOOL_SCHEMA["description"],
                        input_schema=CASE_BANK_TOOL_SCHEMA["input_schema"],
                    )
                )
            if include_pmc_tool:
                defs.append(
                    ToolDefinition(
                        name=PMC_SEARCH_TOOL_SCHEMA["name"],
                        description=PMC_SEARCH_TOOL_SCHEMA["description"],
                        input_schema=PMC_SEARCH_TOOL_SCHEMA["input_schema"],
                    ),
                )
            return defs

        # Single-agent tool-use profiles rely on prompt-template-based routing here.
        # Multi-stage orchestration branches pass explicit tool exposure.
        if include_pmc is None:
            if self.prompt_template_name == "tool_use_pmc":
                include_case_bank = False
                include_pmc = True
            elif self.prompt_template_name == "tool_use_dual":
                include_pmc = True
            else:
                include_pmc = self.prompt_template_name == "tool_use_rag" and self.top_k > 0
        pmc_calls_used = 0

        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_call_log: list[dict] = []
        total_prompt = 0
        total_completion = 0
        total_thinking = 0

        for turn in range(MAX_TOOL_TURNS + 1):
            allow_pmc = include_pmc and (
                self.max_pmc_calls_per_row is None
                or pmc_calls_used < self.max_pmc_calls_per_row
            )
            allow_case_bank = include_case_bank and not (
                require_pmc_once and allow_pmc and pmc_calls_used == 0 and turn > 0
            )
            tool_defs = _build_tool_defs(allow_case_bank, allow_pmc)
            resp = llm_generate_with_tools(
                messages,
                model_id=active_model_id,
                tools=tool_defs,
                config=GenerationConfig(
                    temperature=GENERATION_TEMPERATURE,
                    response_mime_type="application/json",
                    response_json_schema=schema_cls.model_json_schema(),
                    thinking_level=self.thinking_level,
                ),
            )

            total_prompt += resp.prompt_tokens
            total_completion += resp.completion_tokens
            total_thinking += resp.thinking_tokens

            if not resp.has_tool_calls:
                if require_pmc_once and pmc_calls_used == 0:
                    messages.append({
                        "role": "assistant",
                        "content": resp.text,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "Before giving the final answer, call `search_pmc_articles` at least once "
                            "to consult PMC evidence for this case. Use that evidence to confirm or "
                            "challenge the current triage judgment, then return the final JSON answer."
                        ),
                    })
                    continue
                # Final answer
                self.generation_prompt_tokens += total_prompt
                self.generation_completion_tokens += total_completion
                self.generation_thinking_tokens += total_thinking
                gen_cost = self._compute_cost_with_pricing(
                    active_pricing,
                    total_prompt,
                    total_completion,
                    total_thinking,
                )
                self.generation_cost_usd += gen_cost
                self.tool_call_count = len(tool_call_log)
                self.tool_call_details = tool_call_log
                return resp.text, tool_call_log

            # Build assistant message with tool calls.
            # Attach raw Content so Google provider can replay it verbatim
            # (preserves thought_signature required by Gemini 3+).
            assistant_msg: dict = {
                "role": "assistant",
                "content": None,
                "_raw_content": getattr(resp, "raw_assistant_content", None),
                "tool_calls": [
                    {
                        "tool_call_id": tc.tool_call_id,
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                    }
                    for tc in resp.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Execute each tool call — dispatch by tool name
            for tc in resp.tool_calls:
                if tc.tool_name == "search_pmc_articles":
                    pmc_limit_reached = (
                        self.max_pmc_calls_per_row is not None
                        and pmc_calls_used >= self.max_pmc_calls_per_row
                    )
                    if pmc_limit_reached:
                        result_text = "PMC search limit reached for this row."
                        n_results = 0
                    else:
                        query = tc.arguments.get("query", "")
                        top_k = min(int(tc.arguments.get("top_k", 5)), 10)
                        result_text = _search_pmc(
                            query,
                            top_k=top_k,
                            context_chars=self.context_chars,
                        )
                        n_results = result_text.count("--- PMID")
                        pmc_calls_used += 1

                    tool_call_log.append({
                        "turn": turn,
                        "tool_call_id": tc.tool_call_id,
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "n_results": n_results,
                        "result_chars": len(result_text),
                        "pmc_limit_reached": pmc_limit_reached,
                    })
                else:
                    # Case bank search (default)
                    cases = search_cases(
                        esi_level=tc.arguments.get("esi_level"),
                        keywords=tc.arguments.get("keywords"),
                        chapter=tc.arguments.get("chapter"),
                    )
                    result_text = format_tool_result(cases)

                    tool_call_log.append({
                        "turn": turn,
                        "tool_call_id": tc.tool_call_id,
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "n_results": len(cases),
                        "result_chars": len(result_text),
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "content": result_text,
                })

        # Exhausted MAX_TOOL_TURNS — fall back to plain generation
        final_resp = llm_generate(
            prompt,
            model_id=active_model_id,
            config=GenerationConfig(
                temperature=GENERATION_TEMPERATURE,
                response_mime_type="application/json",
                response_json_schema=schema_cls.model_json_schema(),
            ),
        )
        total_prompt += final_resp.prompt_tokens
        total_completion += final_resp.completion_tokens
        total_thinking += final_resp.thinking_tokens
        self.generation_prompt_tokens += total_prompt
        self.generation_completion_tokens += total_completion
        self.generation_thinking_tokens += total_thinking
        gen_cost = self._compute_cost_with_pricing(
            active_pricing,
            total_prompt,
            total_completion,
            total_thinking,
        )
        self.generation_cost_usd += gen_cost
        self.tool_call_count = len(tool_call_log)
        self.tool_call_details = tool_call_log
        return final_resp.text, tool_call_log

    def _run_uncertainty_assessment(self, case: dict) -> tuple[int | None, str]:
        """First-pass uncertainty assessment using the fast model.

        Returns (predicted_esi, confidence) where confidence is
        "confident" or "uncertain".  On any failure, returns (None, "uncertain").
        """
        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = format_vitals_for_prompt(case.get("initial_vitals"))

        prompt = UNCERTAINTY_ASSESSMENT_PROMPT.format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
        )

        assess_start = time.monotonic()

        try:
            resp = self._call_fast_generation(
                prompt,
                config=GenerationConfig(
                    temperature=GENERATION_TEMPERATURE,
                    response_mime_type="application/json",
                    response_json_schema=UNCERTAINTY_ASSESSMENT_SCHEMA,
                ),
            )
            raw_text = (resp.text or "").strip()

            # Track assessment tokens/cost
            self.assessment_prompt_tokens += resp.prompt_tokens
            self.assessment_completion_tokens += resp.completion_tokens
            self.assessment_thinking_tokens += resp.thinking_tokens
            assess_cost = self._compute_cost_with_pricing(
                self.fast_pricing,
                resp.prompt_tokens, resp.completion_tokens, resp.thinking_tokens,
            )
            self.assessment_cost_usd += assess_cost

            self._record_event(
                agent_name="uncertainty_assessment",
                agent_version=f"assessment/{self.fast_model_id}",
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=assess_cost,
                status="success",
                error=None,
                start_time=assess_start,
                thinking_tokens=resp.thinking_tokens,
            )

            # Parse JSON response
            parsed = json.loads(raw_text)
            acuity = int(parsed.get("acuity", 0))
            confidence = str(parsed.get("confidence", "uncertain")).lower()

            if acuity < 1 or acuity > 5:
                return None, "uncertain"

            return acuity, confidence

        except Exception as e:
            print(f"WARNING uncertainty_assessment: {type(e).__name__}: {e}")
            self._record_event(
                agent_name="uncertainty_assessment",
                agent_version=f"assessment/{self.fast_model_id}",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                status="error",
                error=f"{type(e).__name__}: {e}",
                start_time=assess_start,
            )
            return None, "uncertain"

    def _run_structured_stage(
        self,
        prompt: str,
        *,
        stage_name: str,
        model_id: str,
        schema_cls: type[BaseModel],
        pricing: dict[str, float] | None,
    ) -> tuple[BaseModel | None, str | None, str | None, str]:
        stage_start = time.monotonic()
        prompt_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0
        raw_text = None
        error = None
        status = "success"

        try:
            self._check_budget("generation")
            resp = self._call_model_generation(
                prompt,
                model_id=model_id,
                config=GenerationConfig(
                    temperature=GENERATION_TEMPERATURE,
                    response_mime_type="application/json",
                    response_json_schema=schema_cls.model_json_schema(),
                ),
            )
            raw_text = resp.text
            prompt_tokens = resp.prompt_tokens
            completion_tokens = resp.completion_tokens
            thinking_tokens = resp.thinking_tokens
            if raw_text is None:
                raise ValueError(f"{stage_name} returned empty text")
            parsed = schema_cls.model_validate_json(raw_text)
        except CostLimitExceeded as e:
            parsed = None
            error = str(e)
            status = "cost_limit"
        except Exception as e:
            parsed = None
            error = f"{type(e).__name__}: {e}"
            status = "error"

        stage_cost = self._compute_cost_with_pricing(
            pricing,
            prompt_tokens,
            completion_tokens,
            thinking_tokens,
        )
        self.generation_prompt_tokens += prompt_tokens
        self.generation_completion_tokens += completion_tokens
        self.generation_thinking_tokens += thinking_tokens
        self.generation_cost_usd += stage_cost

        self._record_event(
            agent_name=stage_name,
            agent_version=f"{stage_name}/{model_id}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=stage_cost,
            status=status,
            error=error,
            start_time=stage_start,
            thinking_tokens=thinking_tokens,
        )
        return parsed, raw_text, error, status

    def _run_tool_structured_stage(
        self,
        prompt: str,
        *,
        stage_name: str,
        model_id: str,
        schema_cls: type[BaseModel],
        pricing: dict[str, float] | None,
        include_case_bank: bool,
        include_pmc: bool,
        require_pmc_once: bool = False,
    ) -> tuple[BaseModel | None, str | None, str | None, str, list[dict]]:
        stage_start = time.monotonic()
        prompt_before = self.generation_prompt_tokens
        completion_before = self.generation_completion_tokens
        thinking_before = self.generation_thinking_tokens
        cost_before = self.generation_cost_usd
        raw_text = None
        error = None
        status = "success"
        tool_call_log: list[dict] = []

        try:
            self._check_budget("generation")
            raw_text, tool_call_log = self._call_generation_with_tools(
                prompt,
                model_id=model_id,
                schema_cls=schema_cls,
                pricing=pricing,
                include_case_bank=include_case_bank,
                include_pmc=include_pmc,
                require_pmc_once=require_pmc_once,
            )
            if raw_text is None:
                raise ValueError(f"{stage_name} tool-use returned empty text")
            parsed = schema_cls.model_validate_json(raw_text)
        except CostLimitExceeded as e:
            parsed = None
            error = str(e)
            status = "cost_limit"
        except Exception as e:
            parsed = None
            error = f"{type(e).__name__}: {e}"
            status = "error"

        self._record_event(
            agent_name=stage_name,
            agent_version=f"{stage_name}/{model_id}",
            prompt_tokens=self.generation_prompt_tokens - prompt_before,
            completion_tokens=self.generation_completion_tokens - completion_before,
            cost_usd=self.generation_cost_usd - cost_before,
            status=status,
            error=error,
            start_time=stage_start,
            thinking_tokens=self.generation_thinking_tokens - thinking_before,
        )
        return parsed, raw_text, error, status, tool_call_log

    @staticmethod
    def _format_nurse_draft(draft: NurseDraft) -> str:
        return (
            f"Provisional ESI: {draft.acuity}\n"
            f"Confidence: {draft.confidence}\n"
            f"Rationale: {draft.rationale}"
        )

    @staticmethod
    def _format_reader_brief(brief: ReaderBrief) -> str:
        return (
            f"Evidence signal: {brief.evidence_signal}\n"
            f"Summary: {brief.summary}"
        )

    @staticmethod
    def _format_articles_for_selection(articles_df: pd.DataFrame, context_chars: int) -> str:
        parts = []
        for idx, (_, row) in enumerate(articles_df.iterrows()):
            text = prepare_article_excerpt(
                row.get("article_text"),
                context_chars,
                context_text=row.get("context_text"),
            )
            parts.append(f"[Article {idx}] (PMID {row.get('pmid', 'N/A')})\n{text}")
        return "\n\n".join(parts) if parts else "No relevant articles found."

    def _run_two_role_pipeline(
        self,
        case: dict,
        *,
        query_result: QueryResult,
        qhash: str,
        uncertainty_gate_result: str | None,
    ) -> dict[str, Any]:
        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = format_vitals_for_prompt(case.get("initial_vitals"))

        base = self._base_result(
            query_result=query_result,
            qhash=qhash,
            n_articles=0,
            uncertainty_gate_result=uncertainty_gate_result,
        )

        nurse_prompt = self._prompts["nurse"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
        )
        nurse_draft, _, nurse_error, nurse_status = self._run_structured_stage(
            nurse_prompt,
            stage_name="nurse",
            model_id=self.fast_model_id,
            schema_cls=NurseDraft,
            pricing=self.fast_pricing,
        )
        if nurse_draft is None:
            base.update({
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "cost_limit_hit": nurse_error if nurse_status == "cost_limit" else None,
                "error": nurse_error,
            })
            return base

        include_pmc = False
        pmc_trigger_reason = "none"
        if self.prompt_template_name == "two_role_case_bank_pmc_conditional":
            include_pmc = nurse_draft.confidence != "high"
            pmc_trigger_reason = "nurse_confidence_not_high" if include_pmc else "none"

        critic_prompt = self._prompts["critic"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
            nurse_draft=self._format_nurse_draft(nurse_draft),
        )
        final_pred, raw_text, critic_error, critic_status, tool_call_log = self._run_tool_structured_stage(
            critic_prompt,
            stage_name="critic",
            model_id=self.model_id,
            schema_cls=TriagePrediction,
            pricing=self.pricing,
            include_case_bank=True,
            include_pmc=include_pmc,
            require_pmc_once=include_pmc,
        )
        tool_summary = self._tool_log_summary(tool_call_log)
        base.update({
            "triage_RAG_raw": raw_text,
            "triage_RAG": final_pred.acuity if final_pred is not None else None,
            "cost_limit_hit": critic_error if critic_status == "cost_limit" else None,
            "tool_call_count": len(tool_call_log),
            "tool_call_details": tool_call_log or None,
            "pmc_trigger_reason": pmc_trigger_reason,
            "final_changed_from_nurse": (
                final_pred.acuity != nurse_draft.acuity if final_pred is not None else None
            ),
            "multi_role_nurse_acuity": nurse_draft.acuity,
            "multi_role_nurse_confidence": nurse_draft.confidence,
            "error": critic_error,
        })
        base.update(tool_summary)
        return base

    def _run_three_role_rerank_pipeline(
        self,
        case: dict,
        *,
        query_result: QueryResult,
        qhash: str,
        articles: pd.DataFrame,
        n_articles: int,
        top1_distance: float | None,
        mean_topk_distance: float | None,
        top1_score: float | None,
        mean_topk_score: float | None,
        score_type: str | None,
        uncertainty_gate_result: str | None,
    ) -> dict[str, Any]:
        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = format_vitals_for_prompt(case.get("initial_vitals"))
        patient_summary = (
            f"History of present illness: {hpi}\n"
            f"Patient info: {patient_info}\n"
            f"Initial vitals: {initial_vitals}"
        )
        base = self._base_result(
            query_result=query_result,
            qhash=qhash,
            n_articles=n_articles,
            top1_distance=top1_distance,
            mean_topk_distance=mean_topk_distance,
            top1_score=top1_score,
            mean_topk_score=mean_topk_score,
            score_type=score_type,
            uncertainty_gate_result=uncertainty_gate_result,
        )

        nurse_prompt = self._prompts["nurse"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
        )
        nurse_draft, _, nurse_error, nurse_status = self._run_structured_stage(
            nurse_prompt,
            stage_name="nurse",
            model_id=self.fast_model_id,
            schema_cls=NurseDraft,
            pricing=self.fast_pricing,
        )
        if nurse_draft is None:
            base.update({
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "cost_limit_hit": nurse_error if nurse_status == "cost_limit" else None,
                "error": nurse_error,
            })
            return base

        reranker_prompt = self._prompts["reranker"].format(
            patient_summary=patient_summary,
            articles_block=self._format_articles_for_selection(articles, self.rerank_chars),
        )
        selection, _, rerank_error, rerank_status = self._run_structured_stage(
            reranker_prompt,
            stage_name="reranker",
            model_id=self.fast_model_id,
            schema_cls=RerankerSelection,
            pricing=self.fast_pricing,
        )
        if selection is None:
            base.update({
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "cost_limit_hit": rerank_error if rerank_status == "cost_limit" else None,
                "multi_role_nurse_acuity": nurse_draft.acuity,
                "multi_role_nurse_confidence": nurse_draft.confidence,
                "error": rerank_error,
            })
            return base

        valid_indices = [i for i in selection.selected_indices if 0 <= i < len(articles)]
        if not valid_indices:
            valid_indices = list(range(min(len(articles), 1)))
        selected_articles = articles.iloc[valid_indices].reset_index(drop=True) if len(articles) else articles
        reranked_evidence = build_context_block(selected_articles, self.context_chars)
        critic_prompt = self._prompts["critic"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
            nurse_draft=self._format_nurse_draft(nurse_draft),
            reranked_evidence=reranked_evidence,
        )
        final_pred, raw_text, critic_error, critic_status, tool_call_log = self._run_tool_structured_stage(
            critic_prompt,
            stage_name="critic",
            model_id=self.model_id,
            schema_cls=TriagePrediction,
            pricing=self.pricing,
            include_case_bank=True,
            include_pmc=False,
        )
        tool_summary = self._tool_log_summary(tool_call_log)
        base.update({
            "triage_RAG_raw": raw_text,
            "triage_RAG": final_pred.acuity if final_pred is not None else None,
            "cost_limit_hit": critic_error if critic_status == "cost_limit" else None,
            "tool_call_count": len(tool_call_log),
            "tool_call_details": tool_call_log or None,
            "pmc_trigger_reason": "reranked_pmc",
            "final_changed_from_nurse": (
                final_pred.acuity != nurse_draft.acuity if final_pred is not None else None
            ),
            "multi_role_nurse_acuity": nurse_draft.acuity,
            "multi_role_nurse_confidence": nurse_draft.confidence,
            "reranker_selected_count": len(valid_indices),
            "reranker_selected_indices": json.dumps(valid_indices),
            "error": critic_error,
        })
        base.update(tool_summary)
        return base

    def _run_multi_role_pipeline(
        self,
        case: dict,
        *,
        query_result: QueryResult,
        qhash: str,
        articles: pd.DataFrame,
        n_articles: int,
        top1_distance: float | None,
        mean_topk_distance: float | None,
        top1_score: float | None,
        mean_topk_score: float | None,
        score_type: str | None,
        uncertainty_gate_result: str | None,
    ) -> dict[str, Any]:
        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = format_vitals_for_prompt(case.get("initial_vitals"))
        patient_summary = (
            f"History of present illness: {hpi}\n"
            f"Patient info: {patient_info}\n"
            f"Initial vitals: {initial_vitals}"
        )

        nurse_prompt = self._prompts["nurse"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
        )
        if self.multi_role_evidence == "pmc_case_bank":
            nurse_draft, _, nurse_error, nurse_status, _ = self._run_tool_structured_stage(
                nurse_prompt,
                stage_name="nurse",
                model_id=self.fast_model_id,
                schema_cls=NurseDraft,
                pricing=self.fast_pricing,
                include_case_bank=True,
                include_pmc=False,
            )
            nurse_cost_limit_hit = nurse_error if nurse_status == "cost_limit" else None
        else:
            nurse_draft, _, nurse_error, nurse_status = self._run_structured_stage(
                nurse_prompt,
                stage_name="nurse",
                model_id=self.fast_model_id,
                schema_cls=NurseDraft,
                pricing=self.fast_pricing,
            )
            nurse_cost_limit_hit = nurse_error if nurse_status == "cost_limit" else None
        if nurse_draft is None:
            return {
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "triage_query_agent": query_result.agent_name,
                "triage_query_text": query_result.query_text,
                "triage_query_hash": qhash,
                "n_articles_retrieved": n_articles,
                "top1_distance": top1_distance,
                "mean_topk_distance": mean_topk_distance,
                "top1_score": top1_score,
                "mean_topk_score": mean_topk_score,
                "score_type": score_type,
                "cost_limit_hit": nurse_cost_limit_hit,
                "uncertainty_gate_result": uncertainty_gate_result,
                "tool_call_count": None,
                "tool_call_details": None,
                "multi_role_nurse_acuity": None,
                "multi_role_nurse_confidence": None,
                "multi_role_reader_signal": None,
                "error": nurse_error,
            }

        pmc_evidence = build_context_block(articles, self.context_chars)
        evidence_sections = [f"PubMed Central evidence:\n{pmc_evidence}"]
        evidence_block = "\n\n".join(evidence_sections)

        reader_prompt = self._prompts["reader"].format(
            patient_summary=patient_summary,
            nurse_draft=self._format_nurse_draft(nurse_draft),
            evidence_block=evidence_block,
        )
        reader_brief, _, reader_error, reader_status = self._run_structured_stage(
            reader_prompt,
            stage_name="reader",
            model_id=self.fast_model_id,
            schema_cls=ReaderBrief,
            pricing=self.fast_pricing,
        )
        if reader_brief is None:
            return {
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "triage_query_agent": query_result.agent_name,
                "triage_query_text": query_result.query_text,
                "triage_query_hash": qhash,
                "n_articles_retrieved": n_articles,
                "top1_distance": top1_distance,
                "mean_topk_distance": mean_topk_distance,
                "top1_score": top1_score,
                "mean_topk_score": mean_topk_score,
                "score_type": score_type,
                "cost_limit_hit": reader_error if reader_status == "cost_limit" else None,
                "uncertainty_gate_result": uncertainty_gate_result,
                "tool_call_count": None,
                "tool_call_details": None,
                "multi_role_nurse_acuity": nurse_draft.acuity,
                "multi_role_nurse_confidence": nurse_draft.confidence,
                "multi_role_reader_signal": None,
                "error": reader_error,
            }

        adjudicator_prompt = self._prompts["adjudicator"].format(
            hpi=hpi,
            patient_info=patient_info,
            initial_vitals=initial_vitals,
            nurse_draft=self._format_nurse_draft(nurse_draft),
            reader_brief=self._format_reader_brief(reader_brief),
        )
        final_pred, raw_text, adjudicator_error, adjudicator_status = self._run_structured_stage(
            adjudicator_prompt,
            stage_name="adjudicator",
            model_id=self.model_id,
            schema_cls=TriagePrediction,
            pricing=self.pricing,
        )

        return {
            "triage_RAG_raw": raw_text,
            "triage_RAG": final_pred.acuity if final_pred is not None else None,
            "triage_query_agent": query_result.agent_name,
            "triage_query_text": query_result.query_text,
            "triage_query_hash": qhash,
            "n_articles_retrieved": n_articles,
            "top1_distance": top1_distance,
            "mean_topk_distance": mean_topk_distance,
            "top1_score": top1_score,
            "mean_topk_score": mean_topk_score,
            "score_type": score_type,
            "cost_limit_hit": adjudicator_error if adjudicator_status == "cost_limit" else None,
            "uncertainty_gate_result": uncertainty_gate_result,
            "tool_call_count": None,
            "tool_call_details": None,
            "multi_role_nurse_acuity": nurse_draft.acuity,
            "multi_role_nurse_confidence": nurse_draft.confidence,
            "multi_role_reader_signal": reader_brief.evidence_signal,
            "error": adjudicator_error,
        }

    def predict_one(self, case: dict) -> dict[str, Any]:
        """Run the full pipeline for one patient case.

        Returns a dict with keys suitable for DataFrame row construction:
            triage_RAG_raw, triage_RAG, triage_query_agent, triage_query_text,
            triage_query_hash, n_articles_retrieved
        """
        uncertainty_gate_result: str | None = None

        # 0. Uncertainty gate: fast first-pass assessment
        if self.uncertainty_gate and self.top_k > 0:
            esi_pred, confidence = self._run_uncertainty_assessment(case)
            uncertainty_gate_result = confidence
            if confidence == "confident" and esi_pred is not None:
                # Skip retrieval — return the fast model's prediction directly
                raw_json = json.dumps({"acuity": esi_pred})
                return {
                    "triage_RAG_raw": raw_json,
                    "triage_RAG": esi_pred,
                    "triage_query_agent": "uncertainty_gate",
                    "triage_query_text": None,
                    "triage_query_hash": None,
                    "n_articles_retrieved": 0,
                    "top1_distance": None,
                    "mean_topk_distance": None,
                    "top1_score": None,
                    "mean_topk_score": None,
                    "score_type": None,
                    "cost_limit_hit": None,
                    "uncertainty_gate_result": uncertainty_gate_result,
                    "tool_call_count": None,
                    "tool_call_details": None,
                }
            # If uncertain, fall through to normal RAG pipeline

        # 1. Build query (check rewrite budget first)
        query_start = time.monotonic()
        rewrite_status = "success"
        rewrite_error = None
        query_result: QueryResult | None = None

        has_rewrite = hasattr(self.query_agent, 'version') and 'rewrite' in getattr(self.query_agent, 'version', '')
        if has_rewrite:
            try:
                self._check_budget("rewrite")
            except CostLimitExceeded as e:
                rewrite_status = "cost_limit"
                rewrite_error = str(e)
                # Record event and return partial result
                self._record_event(
                    agent_name=self.query_agent.name,
                    agent_version=getattr(self.query_agent, 'version', self.query_agent.name),
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    status="cost_limit",
                    error=str(e),
                    start_time=query_start,
                )
                return {
                    "triage_RAG_raw": None,
                    "triage_RAG": None,
                    "triage_query_agent": self.query_agent.name,
                    "triage_query_text": None,
                    "triage_query_hash": None,
                    "n_articles_retrieved": 0,
                    "cost_limit_hit": "rewrite",
                    "uncertainty_gate_result": uncertainty_gate_result,
                }

        try:
            query_result = self.query_agent.build_query(case, {})
        except Exception as e:
            rewrite_status = "error"
            rewrite_error = f"{type(e).__name__}: {e}"

        # Record rewrite event if applicable
        if has_rewrite and query_result is not None:
            rt = query_result.rewrite_tokens or {}
            rp = rt.get("prompt", 0)
            rc = rt.get("completion", 0)
            r_think = rt.get("thinking", 0)
            rewrite_cost = self._compute_cost(rp, rc, r_think)
            self.rewrite_prompt_tokens += rp
            self.rewrite_completion_tokens += rc
            self.rewrite_thinking_tokens += r_think
            self.rewrite_cost_usd += rewrite_cost
            was_fallback = query_result.metadata.get("fallback", False)
            self._record_event(
                agent_name=self.query_agent.name,
                agent_version=getattr(self.query_agent, 'version', self.query_agent.name),
                prompt_tokens=rp,
                completion_tokens=rc,
                cost_usd=rewrite_cost,
                status="fallback" if was_fallback else "success",
                error=query_result.metadata.get("fallback_reason"),
                start_time=query_start,
                thinking_tokens=r_think,
            )
        elif not has_rewrite and query_result is not None:
            # Non-rewrite agents: record a zero-cost event
            self._record_event(
                agent_name=self.query_agent.name,
                agent_version=getattr(self.query_agent, 'version', self.query_agent.name),
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                status="success",
                error=None,
                start_time=query_start,
            )

        if query_result is None:
            return {
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "triage_query_agent": self.query_agent.name,
                "triage_query_text": None,
                "triage_query_hash": None,
                "n_articles_retrieved": 0,
                "cost_limit_hit": None,
                "uncertainty_gate_result": uncertainty_gate_result,
            }

        # 2. Retrieval (cache-first; skipped when top_k=0 i.e. LLM-only mode)
        qhash = query_result.query_hash
        n_articles = 0
        top1_distance = None
        mean_topk_distance = None
        top1_score = None
        mean_topk_score = None
        score_type = None
        articles = pd.DataFrame()

        if self.top_k > 0:
            # Over-retrieve when reranking is enabled
            retrieve_k = self.top_k * 4 if self.rerank else self.top_k

            if qhash in self.retrieval_cache:
                articles = self.retrieval_cache[qhash]
            else:
                articles = search_pubmed_articles(query_result.query_text, top_k=retrieve_k)

            # Rerank if enabled: score all retrieved articles, keep top_k
            if self.rerank and len(articles) > 0:
                hpi_summary = str(case.get("HPI", ""))[:300]
                patient_summary = f"{hpi_summary} | {str(case.get('patient_info', ''))[:100]}"
                articles = self._rerank_articles(articles, patient_summary)

            n_articles = len(articles)

            if n_articles > 0 and "score" in articles.columns:
                top1_score = round(float(articles["score"].iloc[0]), 6)
                mean_topk_score = round(float(articles["score"].mean()), 6)
                score_type = articles["score_type"].iloc[0] if "score_type" in articles.columns else None

            # Legacy cosine-distance diagnostics (only meaningful for dense backends)
            if n_articles > 0 and "distance" in articles.columns:
                top1_distance = round(float(articles["distance"].iloc[0]), 6)
                mean_topk_distance = round(float(articles["distance"].mean()), 6)
            elif score_type == "cosine_distance" and top1_score is not None:
                top1_distance = top1_score
                mean_topk_distance = mean_topk_score

        if self.prompt_template_name == "multi_role":
            return self._run_multi_role_pipeline(
                case,
                query_result=query_result,
                qhash=qhash,
                articles=articles,
                n_articles=n_articles,
                top1_distance=top1_distance,
                mean_topk_distance=mean_topk_distance,
                top1_score=top1_score,
                mean_topk_score=mean_topk_score,
                score_type=score_type,
                uncertainty_gate_result=uncertainty_gate_result,
            )
        if self.prompt_template_name in TWO_ROLE_PROMPT_TEMPLATES:
            return self._run_two_role_pipeline(
                case,
                query_result=query_result,
                qhash=qhash,
                uncertainty_gate_result=uncertainty_gate_result,
            )
        if self.prompt_template_name == "three_role_rerank_critic":
            return self._run_three_role_rerank_pipeline(
                case,
                query_result=query_result,
                qhash=qhash,
                articles=articles,
                n_articles=n_articles,
                top1_distance=top1_distance,
                mean_topk_distance=mean_topk_distance,
                top1_score=top1_score,
                mean_topk_score=mean_topk_score,
                score_type=score_type,
                uncertainty_gate_result=uncertainty_gate_result,
            )

        # 3. Build prompt
        hpi = str(case.get("HPI", ""))
        patient_info = str(case.get("patient_info", ""))
        initial_vitals = format_vitals_for_prompt(case.get("initial_vitals"))

        if self.top_k > 0:
            context_block = build_context_block(articles, self.context_chars)
            if self.handbook_text:
                prompt = self._prompts["handbook_rag"].format(
                    esi_handbook=self.handbook_text,
                    context=context_block,
                    hpi=hpi,
                    patient_info=patient_info,
                    initial_vitals=initial_vitals,
                )
            else:
                prompt = self._prompts["rag"].format(
                    context=context_block,
                    hpi=hpi,
                    patient_info=patient_info,
                    initial_vitals=initial_vitals,
                )
        else:
            if self.handbook_text:
                prompt = self._prompts["handbook_only"].format(
                    esi_handbook=self.handbook_text,
                    hpi=hpi,
                    patient_info=patient_info,
                    initial_vitals=initial_vitals,
                )
            else:
                prompt = self._prompts["llm"].format(
                    hpi=hpi,
                    patient_info=patient_info,
                    initial_vitals=initial_vitals,
                )

        # 5. Check generation budget + call model
        gen_start = time.monotonic()
        try:
            self._check_budget("generation")
        except CostLimitExceeded as e:
            self._record_event(
                agent_name="generation",
                agent_version=f"generation/{self.model_id}",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                status="cost_limit",
                error=str(e),
                start_time=gen_start,
            )
            return {
                "triage_RAG_raw": None,
                "triage_RAG": None,
                "triage_query_agent": query_result.agent_name,
                "triage_query_text": query_result.query_text,
                "triage_query_hash": qhash,
                "n_articles_retrieved": n_articles,
                "top1_distance": top1_distance,
                "mean_topk_distance": mean_topk_distance,
                "top1_score": top1_score,
                "mean_topk_score": mean_topk_score,
                "score_type": score_type,
                "cost_limit_hit": "generation",
                "uncertainty_gate_result": uncertainty_gate_result,
            }

        raw_text = None
        gen_prompt_tokens = 0
        gen_completion_tokens = 0
        gen_thinking_tokens = 0
        gen_status = "success"
        gen_error = None
        final_error = None
        tool_call_log: list[dict] = []
        fallback_reason = None

        # Snapshot accumulators before generation (tool-use mutates them internally)
        _snap_prompt = self.generation_prompt_tokens
        _snap_compl = self.generation_completion_tokens
        _snap_think = self.generation_thinking_tokens
        _snap_cost = self.generation_cost_usd

        try:
            if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES:
                # Tool-use path: multi-turn loop with case-bank tool
                self.tool_call_count = 0
                self.tool_call_details = []
                raw_text, tool_call_log = self._call_generation_with_tools(prompt)
                # _call_generation_with_tools already accumulated tokens/cost;
                # compute the delta for event recording
                gen_prompt_tokens = self.generation_prompt_tokens - _snap_prompt
                gen_completion_tokens = self.generation_completion_tokens - _snap_compl
                gen_thinking_tokens = self.generation_thinking_tokens - _snap_think
                if raw_text is None:
                    gen_error = "tool-use loop returned None text"
                    gen_status = "error"
            else:
                resp = self._call_generation(prompt)
                raw_text = resp.text
                gen_prompt_tokens = resp.prompt_tokens
                gen_completion_tokens = resp.completion_tokens
                gen_thinking_tokens = resp.thinking_tokens
                if raw_text is None:
                    gen_error = f"resp.text is None (finish_reason={resp.finish_reason})"
                    gen_status = "error"
        except Exception as e:
            gen_status = "error"
            gen_error = f"{type(e).__name__}: {e}"
            print(f"WARNING predict_one: {gen_error}")

        if self.prompt_template_name not in TOOL_USE_PROMPT_TEMPLATES:
            # Non-tool-use: accumulate tokens here (tool-use does it internally)
            gen_cost = self._compute_cost(gen_prompt_tokens, gen_completion_tokens, gen_thinking_tokens)
            self.generation_prompt_tokens += gen_prompt_tokens
            self.generation_completion_tokens += gen_completion_tokens
            self.generation_thinking_tokens += gen_thinking_tokens
            self.generation_cost_usd += gen_cost

        self._record_event(
            agent_name="generation",
            agent_version=f"generation/{self.model_id}",
            prompt_tokens=gen_prompt_tokens,
            completion_tokens=gen_completion_tokens,
            cost_usd=self._compute_cost(gen_prompt_tokens, gen_completion_tokens, gen_thinking_tokens),
            status=gen_status,
            error=gen_error,
            start_time=gen_start,
            thinking_tokens=gen_thinking_tokens,
        )

        # 6. Parse
        parsed = parse_triage(raw_text)
        if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES:
            if raw_text is None:
                fallback_reason = "tool_response_none"
            elif parsed is None:
                fallback_reason = "tool_parse_failed"

            if fallback_reason is not None:
                fallback_prompt = self._build_default_prediction_prompt(
                    hpi=hpi,
                    patient_info=patient_info,
                    initial_vitals=initial_vitals,
                    articles=articles,
                )
                fallback_pred, fallback_raw, fallback_error, fallback_status = self._run_structured_stage(
                    fallback_prompt,
                    stage_name="generation_fallback",
                    model_id=self.model_id,
                    schema_cls=TriagePrediction,
                    pricing=self.pricing,
                )
                if fallback_pred is not None:
                    parsed = fallback_pred.acuity
                    raw_text = fallback_raw
                    final_error = None
                else:
                    raw_text = fallback_raw
                    final_error = (
                        f"{gen_error or fallback_reason}; fallback={fallback_error}"
                        if fallback_error
                        else (gen_error or fallback_reason)
                    )
                    gen_error = final_error
                    gen_status = fallback_status
            else:
                final_error = gen_error
        else:
            final_error = gen_error

        # ── Boundary review: re-examine ESI-2/3 predictions ────────────
        boundary_review_changed = False
        if (
            self.boundary_review
            and parsed is not None
            and parsed in (2, 3)
        ):
            review_prompt = ESI_BOUNDARY_REVIEW_PROMPT.format(
                predicted_esi=parsed,
                hpi=hpi,
                patient_info=patient_info,
                initial_vitals=initial_vitals,
            )
            review_start = time.monotonic()
            try:
                review_resp = self._call_generation(review_prompt)
                review_parsed = parse_triage(review_resp.text)
                review_cost = self._compute_cost(
                    review_resp.prompt_tokens,
                    review_resp.completion_tokens,
                    review_resp.thinking_tokens,
                )
                self.generation_prompt_tokens += review_resp.prompt_tokens
                self.generation_completion_tokens += review_resp.completion_tokens
                self.generation_thinking_tokens += review_resp.thinking_tokens
                self.generation_cost_usd += review_cost
                self._record_event(
                    agent_name="boundary_review",
                    agent_version=f"boundary_review/{self.model_id}",
                    prompt_tokens=review_resp.prompt_tokens,
                    completion_tokens=review_resp.completion_tokens,
                    cost_usd=review_cost,
                    status="success",
                    error=None,
                    start_time=review_start,
                    thinking_tokens=review_resp.thinking_tokens,
                )
                if review_parsed is not None and review_parsed != parsed:
                    boundary_review_changed = True
                    raw_text = review_resp.text
                    parsed = review_parsed
            except Exception as e:
                print(f"WARNING boundary_review: {type(e).__name__}: {e}")
                self._record_event(
                    agent_name="boundary_review",
                    agent_version=f"boundary_review/{self.model_id}",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    start_time=review_start,
                )

        # ── Vitals guardrail: uptriage ESI-3+ when vitals in danger zone ──
        vitals_guardrail_fired = False
        vitals_flags: list[str] = []
        if (
            self.vitals_guardrail
            and parsed is not None
            and parsed >= 3
        ):
            vitals_prompt = VITALS_DANGER_ZONE_PROMPT.format(
                patient_info=patient_info,
                initial_vitals=initial_vitals,
            )
            vitals_start = time.monotonic()
            try:
                vitals_resp = self._call_fast_generation(
                    vitals_prompt,
                    config=GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_json_schema=VitalsDangerZone.model_json_schema(),
                    ),
                )
                vitals_result = VitalsDangerZone.model_validate_json(
                    vitals_resp.text
                )
                vitals_cost = self._compute_cost_with_pricing(
                    self.fast_pricing,
                    vitals_resp.prompt_tokens,
                    vitals_resp.completion_tokens,
                    vitals_resp.thinking_tokens,
                )
                self.generation_prompt_tokens += vitals_resp.prompt_tokens
                self.generation_completion_tokens += vitals_resp.completion_tokens
                self.generation_thinking_tokens += vitals_resp.thinking_tokens
                self.generation_cost_usd += vitals_cost
                self._record_event(
                    agent_name="vitals_guardrail",
                    agent_version=f"vitals_guardrail/{self.fast_model_id}",
                    prompt_tokens=vitals_resp.prompt_tokens,
                    completion_tokens=vitals_resp.completion_tokens,
                    cost_usd=vitals_cost,
                    status="success",
                    error=None,
                    start_time=vitals_start,
                    thinking_tokens=vitals_resp.thinking_tokens,
                )
                if vitals_result.danger_zone:
                    vitals_guardrail_fired = True
                    vitals_flags = vitals_result.flags
                    parsed = 2
            except Exception as e:
                print(f"WARNING vitals_guardrail: {type(e).__name__}: {e}")
                self._record_event(
                    agent_name="vitals_guardrail",
                    agent_version=f"vitals_guardrail/{self.fast_model_id}",
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    start_time=vitals_start,
                )

        tool_summary = self._tool_log_summary(tool_call_log)

        return {
            "triage_RAG_raw": raw_text,
            "triage_RAG": parsed,
            "triage_query_agent": query_result.agent_name,
            "triage_query_text": query_result.query_text,
            "triage_query_hash": qhash,
            "n_articles_retrieved": n_articles,
            "top1_distance": top1_distance,
            "mean_topk_distance": mean_topk_distance,
            "top1_score": top1_score,
            "mean_topk_score": mean_topk_score,
            "score_type": score_type,
            "cost_limit_hit": gen_error if gen_status == "cost_limit" else None,
            "uncertainty_gate_result": uncertainty_gate_result,
            "tool_call_count": self.tool_call_count if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES else None,
            "case_bank_call_count": tool_summary["case_bank_call_count"] if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES else None,
            "pmc_call_count": tool_summary["pmc_call_count"] if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES else None,
            "tool_call_sequence": tool_summary["tool_call_sequence"] if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES else None,
            "tool_call_details": tool_call_log or None,
            "critic_used_case_bank": tool_summary["critic_used_case_bank"] if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES else None,
            "critic_used_pmc": tool_summary["critic_used_pmc"] if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES else None,
            "pmc_trigger_reason": (
                self._single_agent_pmc_trigger_reason(self.prompt_template_name, tool_summary)
                if self.prompt_template_name in TOOL_USE_PROMPT_TEMPLATES
                else None
            ),
            "final_changed_from_nurse": None,
            "reranker_selected_count": None,
            "reranker_selected_indices": None,
            "fallback_reason": fallback_reason,
            "boundary_review_changed": boundary_review_changed if self.boundary_review else None,
            "vitals_guardrail_fired": vitals_guardrail_fired if self.vitals_guardrail else None,
            "vitals_flags": json.dumps(vitals_flags) if vitals_flags else None,
            "error": final_error,
        }
