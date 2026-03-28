from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.query_strategy_sweep import _validate_args
from src.rag.agentic_pipeline import TriageAgenticPipeline
from src.rag.query_agents import QueryResult


class _PredictAgent:
    name = "predict"
    version = "predict/1.0"

    def build_query(self, case, ctx):
        return QueryResult(
            query_text="acute chest pain",
            agent_name=self.name,
            metadata={"strategy": "predict"},
        )


class _FakeToolCall:
    def __init__(self, tool_call_id, tool_name, arguments):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.arguments = arguments


class _FakeToolResponse:
    def __init__(self, *, text=None, tool_calls=None):
        self.text = text
        self.prompt_tokens = 1
        self.completion_tokens = 1
        self.thinking_tokens = 0
        self.tool_calls = tool_calls or []
        self.has_tool_calls = bool(self.tool_calls)
        self.raw_assistant_content = None


def _make_pipeline(**overrides):
    query_result = QueryResult("acute chest pain", "predict", {})
    retrieval_cache = {
        query_result.query_hash: pd.DataFrame(
            [
                {"pmid": "10", "score": 0.11, "score_type": "cosine_distance", "article_text": "Article one text."},
                {"pmid": "20", "score": 0.12, "score_type": "cosine_distance", "article_text": "Article two text."},
                {"pmid": "30", "score": 0.13, "score_type": "cosine_distance", "article_text": "Article three text."},
            ]
        )
    }
    kwargs = {
        "query_agent": _PredictAgent(),
        "top_k": 0,
        "context_chars": 80,
        "rerank_chars": 30,
        "model_id": "main-model",
        "fast_model_id": "fast-model",
        "pricing": {"input": 1.0, "output": 2.0},
        "fast_pricing": {"input": 0.5, "output": 1.0},
        "retrieval_cache": retrieval_cache,
        "prompt_template": "two_role_case_bank",
    }
    kwargs.update(overrides)
    return TriageAgenticPipeline(**kwargs)


@pytest.mark.parametrize(
    ("prompt_template", "mode", "top_k", "message"),
    [
        ("tool_use_pmc", "rag", 5, "requires --mode llm"),
        ("two_role_case_bank", "rag", 5, "requires --mode llm"),
        ("two_role_case_bank_pmc_conditional", "rag", 5, "requires --mode llm"),
        ("three_role_rerank_critic", "llm", 0, "requires --mode rag"),
        ("three_role_rerank_critic", "rag", 0, "requires --top-k > 0"),
    ],
)
def test_validate_args_rejects_new_profile_mismatches(prompt_template, mode, top_k, message):
    args = SimpleNamespace(
        prompt_template=prompt_template,
        mode=mode,
        top_k=top_k,
        distance_gate=None,
        uncertainty_gate=False,
        handbook_prefix=False,
    )

    with pytest.raises(SystemExit, match=message):
        _validate_args(args)


def test_three_role_rerank_constructor_rejects_top_k_zero():
    with pytest.raises(ValueError, match="requires top_k > 0"):
        _make_pipeline(prompt_template="three_role_rerank_critic", top_k=0)


def test_tool_use_pmc_exposes_only_pmc(monkeypatch):
    captured = {}

    def _capture_tools(messages, *, model_id, tools, config):
        captured["tool_names"] = [tool.name for tool in tools]
        return _FakeToolResponse(text='{"acuity": 2}')

    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    pipeline = _make_pipeline(prompt_template="tool_use_pmc")
    text, _ = pipeline._call_generation_with_tools("prompt")

    assert text == '{"acuity": 2}'
    assert captured["tool_names"] == ["search_pmc_articles"]


def test_tool_use_dual_exposes_case_bank_and_pmc(monkeypatch):
    captured = {}

    def _capture_tools(messages, *, model_id, tools, config):
        captured["tool_names"] = [tool.name for tool in tools]
        return _FakeToolResponse(text='{"acuity": 2}')

    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    pipeline = _make_pipeline(prompt_template="tool_use_dual")
    text, _ = pipeline._call_generation_with_tools("prompt")

    assert text == '{"acuity": 2}'
    assert captured["tool_names"] == ["search_esi_case_bank", "search_pmc_articles"]


def test_tool_use_pmc_falls_back_when_tool_loop_returns_none(monkeypatch):
    pipeline = _make_pipeline(prompt_template="tool_use_pmc")
    model_calls = []

    def _fallback_generation(prompt, *, model_id, config):
        model_calls.append(model_id)
        return SimpleNamespace(
            text='{"acuity": 4}',
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    monkeypatch.setattr(pipeline, "_call_model_generation", _fallback_generation)
    monkeypatch.setattr(
        "src.rag.agentic_pipeline.llm_generate_with_tools",
        lambda messages, *, model_id, tools, config: _FakeToolResponse(text=None),
    )

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert model_calls == ["main-model"]
    assert result["triage_RAG"] == 4
    assert result["tool_call_count"] == 0
    assert result["case_bank_call_count"] == 0
    assert result["pmc_call_count"] == 0
    assert result["tool_call_sequence"] is None
    assert result["critic_used_case_bank"] is False
    assert result["critic_used_pmc"] is False
    assert result["pmc_trigger_reason"] == "none"
    assert result["fallback_reason"] == "tool_response_none"
    assert result["error"] is None
    assert any(ev.agent_name == "generation_fallback" and ev.status == "success" for ev in pipeline.events)


def test_tool_use_dual_tracks_tool_sequence_and_falls_back_on_invalid_json(monkeypatch):
    pipeline = _make_pipeline(prompt_template="tool_use_dual")
    model_calls = []
    responses = iter([
        _FakeToolResponse(
            tool_calls=[_FakeToolCall("cb-1", "search_esi_case_bank", {"keywords": "pain"})]
        ),
        _FakeToolResponse(
            tool_calls=[_FakeToolCall("pmc-1", "search_pmc_articles", {"query": "chest pain", "top_k": 3})]
        ),
        _FakeToolResponse(text='{"bad": "json"}'),
    ])

    def _fallback_generation(prompt, *, model_id, config):
        model_calls.append(model_id)
        return SimpleNamespace(
            text='{"acuity": 5}',
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    def _capture_tools(messages, *, model_id, tools, config):
        return next(responses)

    monkeypatch.setattr(pipeline, "_call_model_generation", _fallback_generation)
    monkeypatch.setattr("src.rag.case_bank.search_cases", lambda **kwargs: [])
    monkeypatch.setattr("src.rag.retrieval.search_pmc_articles", lambda query, top_k=5: "--- PMID 123\nPMC")
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert model_calls == ["main-model"]
    assert result["triage_RAG"] == 5
    assert result["tool_call_count"] == 2
    assert result["case_bank_call_count"] == 1
    assert result["pmc_call_count"] == 1
    assert result["tool_call_sequence"] == '["search_esi_case_bank", "search_pmc_articles"]'
    assert result["critic_used_case_bank"] is True
    assert result["critic_used_pmc"] is True
    assert result["pmc_trigger_reason"] == "always_available"
    assert result["fallback_reason"] == "tool_parse_failed"
    assert result["error"] is None
    assert any(ev.agent_name == "generation_fallback" and ev.status == "success" for ev in pipeline.events)


def test_two_role_case_bank_routes_fast_then_main_and_records_case_bank_usage(monkeypatch):
    pipeline = _make_pipeline(prompt_template="two_role_case_bank")
    model_calls = []
    tool_models = []

    def _fake_call_model_generation(prompt, *, model_id, config):
        model_calls.append(model_id)
        return SimpleNamespace(
            text='{"acuity": 3, "confidence": "medium", "rationale": "initial"}',
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    responses = iter([
        _FakeToolResponse(
            tool_calls=[_FakeToolCall("cb-1", "search_esi_case_bank", {"keywords": "chest pain"})]
        ),
        _FakeToolResponse(text='{"acuity": 2}'),
    ])

    def _capture_tools(messages, *, model_id, tools, config):
        tool_models.append((model_id, [tool.name for tool in tools]))
        return next(responses)

    monkeypatch.setattr(pipeline, "_call_model_generation", _fake_call_model_generation)
    monkeypatch.setattr("src.rag.case_bank.search_cases", lambda **kwargs: [])
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert model_calls == ["fast-model"]
    assert tool_models == [("main-model", ["search_esi_case_bank"]), ("main-model", ["search_esi_case_bank"])]
    assert result["triage_RAG"] == 2
    assert result["critic_used_case_bank"] is True
    assert result["critic_used_pmc"] is False
    assert result["case_bank_call_count"] == 1
    assert result["pmc_call_count"] == 0
    assert result["tool_call_sequence"] == '["search_esi_case_bank"]'
    assert result["pmc_trigger_reason"] == "none"
    assert result["final_changed_from_nurse"] is True
    assert result["multi_role_nurse_confidence"] == "medium"


def test_two_role_case_bank_pmc_conditional_skips_pmc_when_nurse_confidence_is_high(monkeypatch):
    pipeline = _make_pipeline(prompt_template="two_role_case_bank_pmc_conditional")

    def _fake_call_model_generation(prompt, *, model_id, config):
        return SimpleNamespace(
            text='{"acuity": 3, "confidence": "high", "rationale": "initial"}',
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    captured = {}

    def _capture_tools(messages, *, model_id, tools, config):
        captured["tool_names"] = [tool.name for tool in tools]
        return _FakeToolResponse(text='{"acuity": 3}')

    monkeypatch.setattr(pipeline, "_call_model_generation", _fake_call_model_generation)
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert captured["tool_names"] == ["search_esi_case_bank"]
    assert result["pmc_trigger_reason"] == "none"
    assert result["critic_used_pmc"] is False
    assert result["case_bank_call_count"] == 0
    assert result["pmc_call_count"] == 0


def test_two_role_case_bank_pmc_conditional_requires_one_pmc_call_when_nurse_not_high(monkeypatch):
    pipeline = _make_pipeline(prompt_template="two_role_case_bank_pmc_conditional")
    tool_snapshots = []
    responses = iter([
        _FakeToolResponse(text='{"acuity": 3}'),
        _FakeToolResponse(
            tool_calls=[_FakeToolCall("pmc-1", "search_pmc_articles", {"query": "chest pain", "top_k": 3})]
        ),
        _FakeToolResponse(text='{"acuity": 2}'),
    ])

    def _fake_call_model_generation(prompt, *, model_id, config):
        return SimpleNamespace(
            text='{"acuity": 3, "confidence": "medium", "rationale": "initial"}',
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    def _capture_tools(messages, *, model_id, tools, config):
        tool_snapshots.append({
            "tool_names": [tool.name for tool in tools],
            "message_count": len(messages),
            "last_user": messages[-1]["content"],
        })
        return next(responses)

    monkeypatch.setattr(pipeline, "_call_model_generation", _fake_call_model_generation)
    monkeypatch.setattr("src.rag.retrieval.search_pmc_articles", lambda query, top_k=5: "--- PMID 123\nPMC")
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert [snap["tool_names"] for snap in tool_snapshots] == [
        ["search_esi_case_bank", "search_pmc_articles"],
        ["search_pmc_articles"],
        ["search_esi_case_bank", "search_pmc_articles"],
    ]
    assert "call `search_pmc_articles` at least once" in tool_snapshots[1]["last_user"]
    assert result["triage_RAG"] == 2
    assert result["pmc_trigger_reason"] == "nurse_confidence_not_high"
    assert result["critic_used_pmc"] is True
    assert result["case_bank_call_count"] == 0
    assert result["pmc_call_count"] == 1
    assert result["tool_call_sequence"] == '["search_pmc_articles"]'


def test_three_role_rerank_critic_uses_fast_fast_main_and_telemetry(monkeypatch):
    pipeline = _make_pipeline(
        prompt_template="three_role_rerank_critic",
        top_k=3,
    )
    captured_prompts = []
    model_calls = []

    responses = iter([
        '{"acuity": 2, "confidence": "low", "rationale": "initial impression"}',
        '{"selected_indices": [1, 2], "selection_rationale": "Most relevant evidence"}',
    ])

    def _fake_call_model_generation(prompt, *, model_id, config):
        model_calls.append(model_id)
        captured_prompts.append(prompt)
        return SimpleNamespace(
            text=next(responses),
            prompt_tokens=1,
            completion_tokens=1,
            thinking_tokens=0,
        )

    tool_models = []

    def _capture_tools(messages, *, model_id, tools, config):
        tool_models.append((model_id, [tool.name for tool in tools], messages[0]["content"]))
        return _FakeToolResponse(text='{"acuity": 2}')

    monkeypatch.setattr(pipeline, "_call_model_generation", _fake_call_model_generation)
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    result = pipeline.predict_one(
        {"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"}
    )

    assert model_calls == ["fast-model", "fast-model"]
    assert tool_models[0][0] == "main-model"
    assert tool_models[0][1] == ["search_esi_case_bank"]
    critic_prompt = tool_models[0][2]
    assert "PMID 20" in critic_prompt
    assert "PMID 30" in critic_prompt
    assert "PMID 10" not in critic_prompt
    assert result["triage_RAG"] == 2
    assert result["reranker_selected_count"] == 2
    assert result["reranker_selected_indices"] == "[1, 2]"
    assert result["critic_used_case_bank"] is False
    assert result["critic_used_pmc"] is False
    assert result["pmc_trigger_reason"] == "reranked_pmc"
