"""Abstract base for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

from src.llm.types import (
    EmbeddingResponse,
    GenerationConfig,
    LLMResponse,
    ToolDefinition,
    ToolUseResponse,
)


class LLMProvider(ABC):
    """Base class that every provider adapter must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate text from *prompt*."""

    def embed(
        self,
        text: str,
        *,
        model_id: str,
    ) -> EmbeddingResponse:
        """Return an embedding vector for *text*.

        Not every provider supports embeddings; the default raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings."
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        tools: list[ToolDefinition],
        config: GenerationConfig | None = None,
    ) -> ToolUseResponse:
        """Generate with tool-use support (multi-turn).

        Not every provider supports tool use; the default raises.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool use."
        )

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider's credentials / SDK are present."""
