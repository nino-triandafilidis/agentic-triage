import signal
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.query_strategy_sweep import (
    APITimeoutError,
    _empty_result,
    _installed_sigalrm_handler,
    _is_timeout_error,
    _reset_provider_client,
    _sigalrm_handler,
)
from src.llm.providers.anthropic import AnthropicProvider
from src.rag.agentic_pipeline import TriageAgenticPipeline
from src.rag.query_agents import QueryResult


class _DummyAgent:
    name = "dummy"
    version = "dummy/1.0"

    def build_query(self, case, ctx):
        raise AssertionError("build_query should not be called in this test")


class _FakeToolResponse:
    def __init__(self, text='{"acuity": 2}', *, has_tool_calls=False):
        self.text = text
        self.prompt_tokens = 1
        self.completion_tokens = 1
        self.thinking_tokens = 0
        self.has_tool_calls = has_tool_calls
        self.tool_calls = []


class _PredictAgent:
    name = "predict"
    version = "predict/1.0"

    def build_query(self, case, ctx):
        return QueryResult(
            query_text="stable query",
            agent_name=self.name,
            metadata={"strategy": "predict"},
        )


def test_empty_result_uses_rag_keys():
    assert _empty_result("boom") == {
        "triage_RAG": None,
        "triage_RAG_raw": None,
        "error": "boom",
    }


def test_reset_provider_client_uses_registry(monkeypatch):
    provider = SimpleNamespace(_client=object())

    def _resolve(model_id: str):
        assert model_id == "gemini-2.5-flash"
        return provider

    monkeypatch.setattr("src.llm.registry.resolve_provider", _resolve)
    _reset_provider_client("gemini-2.5-flash")
    assert provider._client is None


def test_installed_sigalrm_handler_restores_on_exception(monkeypatch):
    calls = []

    def _fake_signal(sig, handler):
        calls.append((sig, handler))
        return "previous-handler"

    monkeypatch.setattr("experiments.query_strategy_sweep.signal.signal", _fake_signal)

    with pytest.raises(RuntimeError, match="boom"):
        with _installed_sigalrm_handler(_sigalrm_handler):
            raise RuntimeError("boom")

    assert calls == [
        (signal.SIGALRM, _sigalrm_handler),
        (signal.SIGALRM, "previous-handler"),
    ]


def test_timeout_classifier_requires_specific_readtimeout():
    class ReadTimeoutError(Exception):
        pass

    class AlreadyReadError(Exception):
        pass

    assert _is_timeout_error(APITimeoutError("wall clock"))
    assert _is_timeout_error(ReadTimeoutError("timed out"))
    assert not _is_timeout_error(AlreadyReadError("not a timeout"))


def test_anthropic_provider_sets_http_timeout(monkeypatch):
    captured = {}

    class _FakeAnthropicClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeTimeout:
        def __init__(self, total, *, connect):
            self.total = total
            self.connect = connect

    fake_anthropic = SimpleNamespace(Anthropic=_FakeAnthropicClient)
    fake_httpx = SimpleNamespace(Timeout=_FakeTimeout)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    provider = AnthropicProvider(api_key="test-key")
    provider._ensure_client()

    assert captured["api_key"] == "test-key"
    assert captured["timeout"].total == 300.0
    assert captured["timeout"].connect == 30.0


def test_tool_use_rag_hides_pmc_tool_when_top_k_zero(monkeypatch):
    captured = {}

    def _capture_tools(messages, *, model_id, tools, config):
        captured["tool_names"] = [tool.name for tool in tools]
        return _FakeToolResponse()

    monkeypatch.setattr("src.rag.case_bank.get_case_bank", lambda: [])
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)

    pipeline = TriageAgenticPipeline(
        query_agent=_DummyAgent(),
        top_k=0,
        context_chars=8000,
        model_id="gemini-2.5-flash",
        pricing={"input": 0.0, "output": 0.0},
        prompt_template="tool_use_rag",
    )

    text, tool_log = pipeline._call_generation_with_tools("prompt")

    assert text == '{"acuity": 2}'
    assert tool_log == []
    assert captured["tool_names"] == ["search_esi_case_bank"]


def test_tool_use_rag_hides_pmc_after_limit(monkeypatch):
    seen_tool_names = []

    class _FakeToolCall:
        def __init__(self, tool_call_id, tool_name, arguments):
            self.tool_call_id = tool_call_id
            self.tool_name = tool_name
            self.arguments = arguments

    class _FakeToolLoopResponse:
        def __init__(self, *, tool_calls=None, text=None):
            self.prompt_tokens = 1
            self.completion_tokens = 1
            self.thinking_tokens = 0
            self.tool_calls = tool_calls or []
            self.has_tool_calls = bool(self.tool_calls)
            self.text = text
            self.raw_assistant_content = None

    responses = iter([
        _FakeToolLoopResponse(
            tool_calls=[_FakeToolCall("pmc-1", "search_pmc_articles", {"query": "chest pain", "top_k": 2})]
        ),
        _FakeToolLoopResponse(text='{"acuity": 2}'),
    ])

    def _capture_tools(messages, *, model_id, tools, config):
        seen_tool_names.append([tool.name for tool in tools])
        return next(responses)

    monkeypatch.setattr("src.rag.case_bank.get_case_bank", lambda: [])
    monkeypatch.setattr("src.rag.agentic_pipeline.llm_generate_with_tools", _capture_tools)
    monkeypatch.setattr(
        "src.rag.retrieval.search_pmc_articles",
        lambda query, top_k=5: "--- PMID 12345\nResult text",
    )

    pipeline = TriageAgenticPipeline(
        query_agent=_DummyAgent(),
        top_k=5,
        context_chars=8000,
        model_id="gemini-2.5-flash",
        pricing={"input": 0.0, "output": 0.0},
        prompt_template="tool_use_rag",
        max_pmc_calls_per_row=1,
    )

    text, tool_log = pipeline._call_generation_with_tools("prompt")

    assert text == '{"acuity": 2}'
    assert tool_log[0]["tool_name"] == "search_pmc_articles"
    assert tool_log[0]["pmc_limit_reached"] is False
    assert seen_tool_names[0] == ["search_esi_case_bank", "search_pmc_articles"]
    assert seen_tool_names[1] == ["search_esi_case_bank"]


def test_rerank_uses_zero_based_indices_and_bm25_tiebreak(monkeypatch):
    prompt_holder = {}

    def _fake_fast_generation(prompt):
        prompt_holder["prompt"] = prompt
        return SimpleNamespace(
            text='[{"article_index": 0, "score": 5}, {"article_index": 1, "score": 5}]',
            prompt_tokens=3,
            completion_tokens=4,
            thinking_tokens=0,
        )

    pipeline = TriageAgenticPipeline(
        query_agent=_DummyAgent(),
        top_k=1,
        context_chars=12,
        rerank_chars=5,
        model_id="gemini-2.5-flash",
        pricing={"input": 0.0, "output": 0.0},
        rerank=True,
    )
    monkeypatch.setattr(pipeline, "_call_fast_generation", _fake_fast_generation)

    articles = pd.DataFrame(
        [
            {"pmid": "10", "score": 1.0, "score_type": "bm25", "article_text": "ABCDEFGHIJKLTAIL"},
            {"pmid": "20", "score": 10.0, "score_type": "bm25", "article_text": "MNOPQRSTUVWXTAIL"},
        ],
        index=[10, 20],
    )

    reranked = pipeline._rerank_articles(articles, "patient summary")

    assert "[Article 0]" in prompt_holder["prompt"]
    assert "[Article 1]" in prompt_holder["prompt"]
    assert "[Article 10]" not in prompt_holder["prompt"]
    assert "[Article 20]" not in prompt_holder["prompt"]
    assert "FGHIJ" not in prompt_holder["prompt"]
    assert "QRSTU" not in prompt_holder["prompt"]
    assert str(reranked.iloc[0]["pmid"]) == "20"


def test_predict_one_sets_uncertainty_gate_result_for_uncertain_and_disabled(monkeypatch):
    retrieval_cache = {
        QueryResult("stable query", "predict", {}).query_hash: pd.DataFrame(
            [{"pmid": "10", "score": 0.1, "score_type": "cosine_distance", "article_text": "context"}]
        )
    }

    def _fake_generation(prompt):
        return SimpleNamespace(
            text='{"acuity": 3}',
            prompt_tokens=2,
            completion_tokens=1,
            thinking_tokens=0,
            finish_reason="stop",
        )

    uncertain_pipeline = TriageAgenticPipeline(
        query_agent=_PredictAgent(),
        top_k=1,
        context_chars=20,
        model_id="gemini-2.5-flash",
        pricing={"input": 0.0, "output": 0.0},
        retrieval_cache=retrieval_cache,
        uncertainty_gate=True,
    )
    monkeypatch.setattr(uncertain_pipeline, "_run_uncertainty_assessment", lambda case: (3, "uncertain"))
    monkeypatch.setattr(uncertain_pipeline, "_call_generation", _fake_generation)

    disabled_pipeline = TriageAgenticPipeline(
        query_agent=_PredictAgent(),
        top_k=0,
        context_chars=20,
        model_id="gemini-2.5-flash",
        pricing={"input": 0.0, "output": 0.0},
        uncertainty_gate=False,
    )
    monkeypatch.setattr(disabled_pipeline, "_call_generation", _fake_generation)

    uncertain_result = uncertain_pipeline.predict_one({"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"})
    disabled_result = disabled_pipeline.predict_one({"HPI": "pain", "patient_info": "adult", "initial_vitals": "stable"})

    assert uncertain_result["uncertainty_gate_result"] == "uncertain"
    assert disabled_result["uncertainty_gate_result"] is None
