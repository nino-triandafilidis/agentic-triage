from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.query_strategy_sweep import _validate_args
from src.rag.agentic_pipeline import CostLimitExceeded, TriageAgenticPipeline
from src.rag.query_agents import QueryResult
from src.schemas import NurseDraft, ReaderBrief


class _PredictAgent:
    name = "predict"
    version = "predict/1.0"

    def build_query(self, case, ctx):
        return QueryResult(
            query_text="chest pain stable",
            agent_name=self.name,
            metadata={"strategy": "predict"},
        )


def _make_pipeline(**overrides):
    query_result = QueryResult("chest pain stable", "predict", {})
    retrieval_cache = {
        query_result.query_hash: pd.DataFrame(
            [
                {
                    "pmid": "10",
                    "score": 0.1,
                    "score_type": "cosine_distance",
                    "article_text": "Relevant PMC article text.",
                }
            ]
        )
    }
    kwargs = {
        "query_agent": _PredictAgent(),
        "top_k": 1,
        "context_chars": 50,
        "model_id": "main-model",
        "fast_model_id": "fast-model",
        "pricing": {"input": 10.0, "output": 20.0},
        "fast_pricing": {"input": 2.0, "output": 3.0},
        "retrieval_cache": retrieval_cache,
        "prompt_template": "multi_role",
        "multi_role_evidence": "pmc",
    }
    kwargs.update(overrides)
    return TriageAgenticPipeline(**kwargs)


def _patch_role_calls(monkeypatch, pipeline, *, calls, token_count=1):
    responses = iter([
        '{"acuity": 2, "confidence": "medium", "rationale": "initial impression"}',
        '{"evidence_signal": "supports_same", "summary": "Evidence aligns with the draft."}',
        '{"acuity": 2}',
    ])

    def _fake_call_model_generation(prompt, *, model_id, config):
        calls.append((model_id, prompt, config.response_json_schema))
        return SimpleNamespace(
            text=next(responses),
            prompt_tokens=token_count,
            completion_tokens=token_count,
            thinking_tokens=0,
        )

    monkeypatch.setattr(pipeline, "_call_model_generation", _fake_call_model_generation)


class _FakeToolResponse:
    def __init__(self, text):
        self.text = text
        self.prompt_tokens = 1
        self.completion_tokens = 1
        self.thinking_tokens = 0
        self.has_tool_calls = False
        self.tool_calls = []


@pytest.mark.parametrize(
    ("mode", "top_k", "message"),
    [
        ("llm", 0, "requires --mode rag"),
        ("rag", 0, "requires --top-k > 0"),
    ],
)
def test_validate_args_rejects_invalid_multi_role_configs(mode, top_k, message):
    args = SimpleNamespace(
        prompt_template="multi_role",
        mode=mode,
        top_k=top_k,
        distance_gate=None,
        uncertainty_gate=False,
        handbook_prefix=False,
    )

    with pytest.raises(SystemExit, match=message):
        _validate_args(args)


def test_multi_role_pmc_routes_fast_fast_main_and_excludes_case_bank(monkeypatch):
    pipeline = _make_pipeline(multi_role_evidence="pmc")
    calls = []
    _patch_role_calls(monkeypatch, pipeline, calls=calls)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert [model_id for model_id, _, _ in calls] == [
        "fast-model",
        "fast-model",
        "main-model",
    ]
    assert "ESI case bank evidence" not in calls[1][1]
    assert result["triage_RAG"] == 2
    assert result["multi_role_nurse_acuity"] == 2
    assert result["multi_role_nurse_confidence"] == "medium"
    assert result["multi_role_reader_signal"] == "supports_same"


def test_multi_role_pmc_case_bank_gives_case_bank_tool_to_nurse_only(monkeypatch):
    pipeline = _make_pipeline(multi_role_evidence="pmc_case_bank")
    calls = []
    tool_calls = []
    responses = iter([
        '{"evidence_signal": "supports_same", "summary": "Evidence aligns with the draft."}',
        '{"acuity": 2}',
    ])

    def _fake_call_model_generation(prompt, *, model_id, config):
        calls.append((model_id, prompt, config.response_json_schema))
        return SimpleNamespace(
            text=next(responses),
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    def _fake_generate_with_tools(messages, *, model_id, tools, config):
        tool_calls.append((model_id, [tool.name for tool in tools], config.response_json_schema))
        return _FakeToolResponse(
            '{"acuity": 2, "confidence": "medium", "rationale": "case-bank assisted"}'
        )

    monkeypatch.setattr(pipeline, "_call_model_generation", _fake_call_model_generation)
    monkeypatch.setattr(
        "src.rag.agentic_pipeline.llm_generate_with_tools",
        _fake_generate_with_tools,
    )

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert len(tool_calls) == 1
    assert tool_calls[0][0] == "fast-model"
    assert tool_calls[0][1] == ["search_esi_case_bank"]
    assert [model_id for model_id, _, _ in calls] == ["fast-model", "main-model"]
    assert "ESI case bank evidence" not in calls[0][1]
    assert result["multi_role_nurse_confidence"] == "medium"


def test_multi_role_no_articles_still_runs_reader_and_adjudicator(monkeypatch):
    query_result = QueryResult("chest pain stable", "predict", {})
    retrieval_cache = {query_result.query_hash: pd.DataFrame()}
    pipeline = _make_pipeline(retrieval_cache=retrieval_cache)
    calls = []
    _patch_role_calls(monkeypatch, pipeline, calls=calls)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert result["triage_RAG"] == 2
    assert "No relevant articles found." in calls[1][1]


def test_multi_role_uses_fast_pricing_for_nurse_and_reader(monkeypatch):
    pipeline = _make_pipeline()
    calls = []
    _patch_role_calls(monkeypatch, pipeline, calls=calls, token_count=1)

    pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert pipeline.generation_cost_usd == pytest.approx(40.0)


def test_multi_role_cost_limit_hit_propagates_from_nurse_tool_path(monkeypatch):
    pipeline = _make_pipeline(multi_role_evidence="pmc_case_bank")

    def _raise_cost_limit(*args, **kwargs):
        raise CostLimitExceeded("total", 7.01, 7.0)

    monkeypatch.setattr(pipeline, "_call_generation_with_tools", _raise_cost_limit)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert result["triage_RAG"] is None
    assert result["cost_limit_hit"] == str(CostLimitExceeded("total", 7.01, 7.0))
    assert result["error"] == result["cost_limit_hit"]


def test_structured_stage_literals_reject_invalid_values():
    with pytest.raises(Exception):
        NurseDraft.model_validate_json(
            '{"acuity": 2, "confidence": "certain", "rationale": "bad enum"}'
        )

    with pytest.raises(Exception):
        ReaderBrief.model_validate_json(
            '{"evidence_signal": "supports_weird", "summary": "bad enum"}'
        )
