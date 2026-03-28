"""Model-to-provider routing and provider lifecycle management."""

from __future__ import annotations

import logging
from typing import Callable

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Prefix → (provider_key, factory)
# Order matters: first match wins.
_PREFIX_RULES: list[tuple[str, str, Callable[[], LLMProvider]]] = []
_providers: dict[str, LLMProvider] = {}


def _default_rules() -> list[tuple[str, str, Callable[[], LLMProvider]]]:
    """Lazily build the default prefix rules (avoids import-time SDK deps)."""

    def _google_factory() -> LLMProvider:
        from src.llm.providers.google import GoogleProvider
        return GoogleProvider()

    def _anthropic_factory() -> LLMProvider:
        from src.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider()

    def _kimi_factory() -> LLMProvider:
        from src.llm.providers.kimi import KimiProvider
        return KimiProvider()

    return [
        ("gemini-", "google", _google_factory),
        ("text-embedding-", "google", _google_factory),
        ("claude-", "anthropic", _anthropic_factory),
        ("moonshot-", "kimi", _kimi_factory),
        ("kimi-", "kimi", _kimi_factory),
    ]


def _ensure_rules() -> None:
    global _PREFIX_RULES
    if not _PREFIX_RULES:
        _PREFIX_RULES = _default_rules()


def register_prefix(
    prefix: str,
    provider_key: str,
    factory: Callable[[], LLMProvider],
) -> None:
    """Register a new model-prefix → provider mapping.

    Inserted at the front so user registrations take priority.
    """
    _ensure_rules()
    _PREFIX_RULES.insert(0, (prefix, provider_key, factory))


def register_provider(key: str, provider: LLMProvider) -> None:
    """Directly register a pre-built provider instance (useful for testing)."""
    _providers[key] = provider


def resolve_provider(model_id: str) -> LLMProvider:
    """Return the provider instance responsible for *model_id*.

    Creates the provider lazily on first use and caches it by key.
    """
    _ensure_rules()

    for prefix, key, factory in _PREFIX_RULES:
        if model_id.startswith(prefix):
            if key not in _providers:
                _providers[key] = factory()
                logger.info("Initialised LLM provider %r for model %r", key, model_id)
            return _providers[key]

    available = sorted({key for _, key, _ in _PREFIX_RULES})
    raise ValueError(
        f"No provider registered for model {model_id!r}. "
        f"Known providers: {available}. "
        f"Register one with llm.register_prefix()."
    )


def get_provider(key: str) -> LLMProvider | None:
    """Return a cached provider by key, or None."""
    return _providers.get(key)


def reset() -> None:
    """Clear all cached providers (useful for testing)."""
    _providers.clear()
