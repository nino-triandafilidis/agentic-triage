"""Typed query-agent contracts and implementations for triage RAG.

Agents construct retrieval queries from patient case data.  Each agent
implements the QueryAgent protocol, returning a QueryResult that the
orchestrator (agentic_pipeline.py) feeds to retrieval + generation.

Phase 1 agents:
- ConcatQueryAgent   — concatenate HPI + patient_info + vitals (baseline)
- HPIOnlyQueryAgent  — HPI only (ablation)
- RewriteQueryAgent  — LLM-rewritten query with concat fallback
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
import uuid
from typing import Any, Callable, Protocol


# ── Typed contracts ──────────────────────────────────────────────────────

@dataclasses.dataclass
class RewriteResult:
    """Result from an LLM rewrite call."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_id: str
    thinking_tokens: int = 0


@dataclasses.dataclass
class QueryResult:
    """Output of a QueryAgent.build_query() call."""
    query_text: str
    agent_name: str
    metadata: dict[str, Any]
    knowledge_source: str = "pmc"
    handbook_mode: str = "none"
    rewrite_tokens: dict[str, int] | None = None

    @property
    def query_hash(self) -> str:
        return hashlib.sha256(self.query_text.encode()).hexdigest()


@dataclasses.dataclass
class ExecutionEvent:
    """Minimal trace object for one pipeline step."""
    run_id: str
    step_id: str
    parent_step_id: str | None
    agent_name: str
    agent_version: str
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    cost_usd: float
    status: str  # "success" | "fallback" | "error" | "cost_limit"
    error: str | None
    timestamp_utc: str
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class QueryAgent(Protocol):
    """Protocol for query construction agents."""
    name: str

    def build_query(self, case: dict, context: dict) -> QueryResult: ...


# ── Helpers ──────────────────────────────────────────────────────────────

def _new_step_id() -> str:
    return uuid.uuid4().hex[:12]


def _concat_fields(case: dict) -> str:
    hpi = str(case.get("HPI", ""))
    patient_info = str(case.get("patient_info", ""))
    initial_vitals = str(case.get("initial_vitals", ""))
    return f"{hpi} {patient_info} {initial_vitals}".strip()


# ── Agent implementations ────────────────────────────────────────────────

class NoopQueryAgent:
    """No-op agent for LLM-only mode. Returns empty query (retrieval skipped)."""

    name = "noop"
    version = "noop/1.0"

    def build_query(self, case: dict, context: dict) -> QueryResult:
        return QueryResult(
            query_text="",
            agent_name=self.name,
            metadata={"strategy": "noop"},
        )


class ConcatQueryAgent:
    """Baseline: concatenate HPI + patient_info + initial_vitals."""

    name = "concat"
    version = "concat/1.0"

    def build_query(self, case: dict, context: dict) -> QueryResult:
        query_text = _concat_fields(case)
        return QueryResult(
            query_text=query_text,
            agent_name=self.name,
            metadata={"strategy": "concat"},
        )


class HPIOnlyQueryAgent:
    """Ablation: use only the HPI field as query."""

    name = "hpi_only"
    version = "hpi_only/1.0"

    def build_query(self, case: dict, context: dict) -> QueryResult:
        query_text = str(case.get("HPI", "")).strip()
        return QueryResult(
            query_text=query_text,
            agent_name=self.name,
            metadata={"strategy": "hpi_only"},
        )


class RewriteQueryAgent:
    """LLM-rewritten query. Falls back to concat on failure."""

    name = "rewrite_v1"
    version = "rewrite_v1/1.0"

    def __init__(self, rewrite_fn: Callable[[str, str], RewriteResult]):
        """
        Args:
            rewrite_fn: Callable(raw_query, instruction) -> RewriteResult.
                The caller provides the actual LLM call; this agent owns the
                instruction text and fallback logic.
        """
        self._rewrite_fn = rewrite_fn

    _REWRITE_INSTRUCTION = (
        "You are a medical information retrieval specialist. "
        "Given a patient's clinical presentation below, rewrite it into a concise "
        "PubMed search query that would retrieve the most relevant emergency medicine "
        "literature for triage decision support. Focus on key symptoms, conditions, "
        "and clinical findings. Output ONLY the rewritten query, nothing else.\n\n"
        "Clinical presentation:\n{raw_query}"
    )

    def build_query(self, case: dict, context: dict) -> QueryResult:
        raw_query = _concat_fields(case)
        instruction = self._REWRITE_INSTRUCTION.format(raw_query=raw_query)

        try:
            result = self._rewrite_fn(raw_query, instruction)
            rewritten = result.text.strip()
            if not rewritten:
                raise ValueError("Empty rewrite result")
            return QueryResult(
                query_text=rewritten,
                agent_name=self.name,
                metadata={
                    "strategy": "rewrite",
                    "original_query": raw_query,
                    "rewrite_model": result.model_id,
                    "fallback": False,
                },
                rewrite_tokens={
                    "prompt": result.prompt_tokens,
                    "completion": result.completion_tokens,
                    "thinking": result.thinking_tokens,
                },
            )
        except Exception as e:
            # Fallback to concat
            return QueryResult(
                query_text=raw_query,
                agent_name=self.name,
                metadata={
                    "strategy": "rewrite",
                    "original_query": raw_query,
                    "fallback": True,
                    "fallback_reason": f"{type(e).__name__}: {e}",
                },
            )


# ── Factory ──────────────────────────────────────────────────────────────

_AGENT_REGISTRY: dict[str, type] = {
    "noop": NoopQueryAgent,
    "concat": ConcatQueryAgent,
    "hpi_only": HPIOnlyQueryAgent,
    "rewrite": RewriteQueryAgent,
}


def get_agent(
    name: str,
    *,
    rewrite_fn: Callable[[str, str], RewriteResult] | None = None,
) -> ConcatQueryAgent | HPIOnlyQueryAgent | RewriteQueryAgent:
    """Resolve an agent by CLI name.

    Args:
        name: One of "concat", "hpi_only", "rewrite".
        rewrite_fn: Required when name="rewrite".

    Raises:
        ValueError: Unknown agent name or missing rewrite_fn.
    """
    if name not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {sorted(_AGENT_REGISTRY)}")
    cls = _AGENT_REGISTRY[name]
    if cls is RewriteQueryAgent:
        if rewrite_fn is None:
            raise ValueError("rewrite_fn is required for 'rewrite' agent")
        return cls(rewrite_fn=rewrite_fn)
    return cls()
