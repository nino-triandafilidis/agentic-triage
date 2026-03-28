"""Tests for the provider-agnostic LLM interface.

Uses mock providers so no real API keys or network calls are needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm import (
    EmbeddingResponse,
    GenerationConfig,
    LLMResponse,
    generate,
    embed,
    register_prefix,
    register_provider,
    reset,
)
from src.llm.base import LLMProvider
from src.llm.registry import resolve_provider, _providers, _PREFIX_RULES


# ── Fixtures ─────────────────────────────────────────────────────────────


class FakeProvider(LLMProvider):
    """In-memory stub for testing the dispatch layer."""

    def __init__(self, text: str = "fake-response", embed_dim: int = 3):
        self._text = text
        self._embed_dim = embed_dim
        self.generate_calls: list[dict] = []
        self.embed_calls: list[dict] = []

    def generate(self, prompt, *, model_id, config=None):
        self.generate_calls.append(
            {"prompt": prompt, "model_id": model_id, "config": config}
        )
        return LLMResponse(
            text=self._text,
            prompt_tokens=10,
            completion_tokens=5,
            thinking_tokens=0,
            model_id=model_id,
        )

    def embed(self, text, *, model_id):
        self.embed_calls.append({"text": text, "model_id": model_id})
        return EmbeddingResponse(
            values=[0.1] * self._embed_dim,
            model_id=model_id,
        )

    def is_available(self):
        return True


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global provider cache between tests."""
    reset()
    # Clear prefix rules so _ensure_rules re-populates them from defaults.
    from src.llm import registry
    registry._PREFIX_RULES.clear()
    yield
    reset()
    registry._PREFIX_RULES.clear()


# ── Registry tests ───────────────────────────────────────────────────────


class TestRegistry:
    def test_resolve_gemini_returns_google_key(self):
        provider = resolve_provider("gemini-2.5-flash")
        assert provider is not None
        from src.llm.providers.google import GoogleProvider
        assert isinstance(provider, GoogleProvider)

    def test_resolve_claude_returns_anthropic_key(self):
        provider = resolve_provider("claude-3-opus-20240229")
        from src.llm.providers.anthropic import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)

    def test_resolve_moonshot_returns_kimi_key(self):
        provider = resolve_provider("moonshot-v1-8k")
        from src.llm.providers.kimi import KimiProvider
        assert isinstance(provider, KimiProvider)

    def test_resolve_kimi_prefix_returns_kimi_key(self):
        provider = resolve_provider("kimi-v1")
        from src.llm.providers.kimi import KimiProvider
        assert isinstance(provider, KimiProvider)

    def test_resolve_text_embedding_returns_google(self):
        provider = resolve_provider("text-embedding-005")
        from src.llm.providers.google import GoogleProvider
        assert isinstance(provider, GoogleProvider)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="No provider registered"):
            resolve_provider("llama-3-70b")

    def test_register_prefix_takes_priority(self):
        fake = FakeProvider()
        register_prefix("llama-", "ollama", lambda: fake)
        assert resolve_provider("llama-3-70b") is fake

    def test_register_provider_direct(self):
        fake = FakeProvider()
        register_provider("test", fake)
        register_prefix("test-", "test", lambda: fake)
        assert resolve_provider("test-model") is fake

    def test_provider_cached_across_calls(self):
        p1 = resolve_provider("gemini-2.5-flash")
        p2 = resolve_provider("gemini-2.0-flash")
        assert p1 is p2


# ── Public API tests ─────────────────────────────────────────────────────


class TestGenerateEmbed:
    def test_generate_dispatches_to_provider(self):
        fake = FakeProvider(text="hello")
        register_provider("test_gen", fake)
        register_prefix("fakemodel-", "test_gen", lambda: fake)

        resp = generate("prompt text", model_id="fakemodel-1")
        assert resp.text == "hello"
        assert resp.prompt_tokens == 10
        assert resp.model_id == "fakemodel-1"
        assert len(fake.generate_calls) == 1
        assert fake.generate_calls[0]["prompt"] == "prompt text"

    def test_generate_passes_config(self):
        fake = FakeProvider()
        register_provider("cfg", fake)
        register_prefix("cfg-", "cfg", lambda: fake)

        cfg = GenerationConfig(temperature=0.5, max_output_tokens=100)
        generate("hi", model_id="cfg-model", config=cfg)
        assert fake.generate_calls[0]["config"] is cfg

    def test_embed_dispatches_to_provider(self):
        fake = FakeProvider(embed_dim=768)
        register_provider("emb", fake)
        register_prefix("emb-", "emb", lambda: fake)

        resp = embed("some text", model_id="emb-v1")
        assert len(resp.values) == 768
        assert resp.model_id == "emb-v1"
        assert len(fake.embed_calls) == 1

    def test_embed_not_supported_raises(self):
        class NoEmbedProvider(LLMProvider):
            def generate(self, prompt, *, model_id, config=None):
                return LLMResponse(text="x")
            def is_available(self):
                return True

        p = NoEmbedProvider()
        register_provider("noemb", p)
        register_prefix("noemb-", "noemb", lambda: p)
        with pytest.raises(NotImplementedError, match="does not support embeddings"):
            embed("text", model_id="noemb-v1")


# ── Type tests ───────────────────────────────────────────────────────────


class TestTypes:
    def test_llm_response_defaults(self):
        r = LLMResponse(text="hi")
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.thinking_tokens == 0
        assert r.model_id == ""
        assert r.finish_reason is None
        assert r.raw_response is None

    def test_generation_config_defaults(self):
        c = GenerationConfig()
        assert c.temperature == 0.0
        assert c.max_output_tokens is None
        assert c.system_instruction is None
        assert c.response_mime_type is None
        assert c.response_json_schema is None

    def test_generation_config_frozen(self):
        c = GenerationConfig(temperature=0.5)
        with pytest.raises(AttributeError):
            c.temperature = 1.0

    def test_embedding_response(self):
        r = EmbeddingResponse(values=[0.1, 0.2], model_id="m")
        assert len(r.values) == 2
        assert r.model_id == "m"


# ── Google provider unit tests (mocked SDK) ──────────────────────────────


class TestGoogleProviderMocked:
    def _make_provider(self):
        from src.llm.providers.google import GoogleProvider
        p = GoogleProvider(project="test-project")
        return p

    def test_generate_calls_genai(self):
        p = self._make_provider()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "test output"
        mock_resp.usage_metadata.prompt_token_count = 20
        mock_resp.usage_metadata.candidates_token_count = 10
        mock_resp.usage_metadata.thoughts_token_count = 0
        mock_resp.candidates = [MagicMock(finish_reason="STOP")]
        mock_client.models.generate_content.return_value = mock_resp
        p._client = mock_client

        resp = p.generate("hello", model_id="gemini-2.5-flash")
        assert resp.text == "test output"
        assert resp.prompt_tokens == 20
        assert resp.completion_tokens == 10
        mock_client.models.generate_content.assert_called_once()

    def test_generate_passes_json_schema(self):
        p = self._make_provider()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"acuity": 3}'
        mock_resp.usage_metadata.prompt_token_count = 5
        mock_resp.usage_metadata.candidates_token_count = 3
        mock_resp.usage_metadata.thoughts_token_count = 0
        mock_resp.candidates = [MagicMock(finish_reason="STOP")]
        mock_client.models.generate_content.return_value = mock_resp
        p._client = mock_client

        cfg = GenerationConfig(
            response_mime_type="application/json",
            response_json_schema={"type": "object"},
        )
        resp = p.generate("hi", model_id="gemini-2.5-flash", config=cfg)
        assert resp.text == '{"acuity": 3}'

    def test_embed_calls_genai(self):
        p = self._make_provider()
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_resp = MagicMock()
        mock_resp.embeddings = [mock_embedding]
        mock_client.models.embed_content.return_value = mock_resp
        p._client = mock_client

        resp = p.embed("text", model_id="text-embedding-005")
        assert resp.values == [0.1, 0.2, 0.3]
        assert resp.model_id == "text-embedding-005"

    def test_is_available(self):
        p = self._make_provider()
        assert p.is_available() is True


# ── Anthropic provider unit tests (mocked SDK) ──────────────────────────


class TestAnthropicProviderMocked:
    def _make_provider(self):
        from src.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key="test-key")

    def test_generate_calls_sdk(self):
        p = self._make_provider()
        mock_client = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Claude says hi"
        mock_resp = MagicMock()
        mock_resp.content = [text_block]
        mock_resp.usage.input_tokens = 15
        mock_resp.usage.output_tokens = 8
        mock_resp.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_resp
        p._client = mock_client

        resp = p.generate("hello", model_id="claude-3-opus-20240229")
        assert resp.text == "Claude says hi"
        assert resp.prompt_tokens == 15
        assert resp.completion_tokens == 8
        assert resp.model_id == "claude-3-opus-20240229"

    def test_generate_with_system_instruction(self):
        p = self._make_provider()
        mock_client = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "response"
        mock_resp = MagicMock()
        mock_resp.content = [text_block]
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 3
        mock_resp.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_resp
        p._client = mock_client

        cfg = GenerationConfig(system_instruction="You are a nurse.")
        p.generate("hi", model_id="claude-3-opus-20240229", config=cfg)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a nurse."

    def test_json_extraction(self):
        from src.llm.providers.anthropic import _extract_json
        assert _extract_json('{"a": 1}') == '{"a": 1}'
        assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert _extract_json('Here is the result:\n{"a": 1}') == '{"a": 1}'

    def test_is_available_with_key(self):
        p = self._make_provider()
        assert p.is_available() is True

    def test_is_available_without_key(self):
        from src.llm.providers.anthropic import AnthropicProvider
        with patch.dict("os.environ", {}, clear=True):
            p = AnthropicProvider(api_key=None)
            # Will be False because no key and env not set
            # (assuming anthropic is installed)
            try:
                import anthropic  # noqa: F401
                assert p.is_available() is False
            except ImportError:
                assert p.is_available() is False


# ── Kimi provider unit tests (mocked SDK) ───────────────────────────────


class TestKimiProviderMocked:
    def _make_provider(self):
        from src.llm.providers.kimi import KimiProvider
        return KimiProvider(api_key="test-moonshot-key")

    def test_generate_calls_openai_sdk(self):
        p = self._make_provider()
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Kimi response"
        choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 12
        mock_resp.usage.completion_tokens = 6
        mock_client.chat.completions.create.return_value = mock_resp
        p._client = mock_client

        resp = p.generate("hi", model_id="moonshot-v1-8k")
        assert resp.text == "Kimi response"
        assert resp.prompt_tokens == 12
        assert resp.completion_tokens == 6
        assert resp.model_id == "moonshot-v1-8k"

    def test_generate_with_json_format(self):
        p = self._make_provider()
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"acuity": 3}'
        choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        mock_client.chat.completions.create.return_value = mock_resp
        p._client = mock_client

        cfg = GenerationConfig(response_json_schema={"type": "object"})
        p.generate("hi", model_id="moonshot-v1-8k", config=cfg)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_embed_not_supported(self):
        p = self._make_provider()
        with pytest.raises(NotImplementedError):
            p.embed("text", model_id="moonshot-v1-8k")


# ── Integration: end-to-end dispatch with fake providers ─────────────────


class TestEndToEndDispatch:
    def test_multi_provider_routing(self):
        google_fake = FakeProvider(text="google-resp")
        claude_fake = FakeProvider(text="claude-resp")
        kimi_fake = FakeProvider(text="kimi-resp")

        register_provider("google", google_fake)
        register_provider("anthropic", claude_fake)
        register_provider("kimi", kimi_fake)

        r1 = generate("p", model_id="gemini-2.5-flash")
        r2 = generate("p", model_id="claude-3-opus-20240229")
        r3 = generate("p", model_id="moonshot-v1-8k")

        assert r1.text == "google-resp"
        assert r2.text == "claude-resp"
        assert r3.text == "kimi-resp"
        assert len(google_fake.generate_calls) == 1
        assert len(claude_fake.generate_calls) == 1
        assert len(kimi_fake.generate_calls) == 1

    def test_embed_routing(self):
        google_fake = FakeProvider(embed_dim=768)
        register_provider("google", google_fake)

        resp = embed("query text", model_id="text-embedding-005")
        assert len(resp.values) == 768
        assert google_fake.embed_calls[0]["text"] == "query text"
