"""Shared types for the provider-agnostic LLM interface."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    """Provider-agnostic generation parameters."""

    temperature: float = 0.0
    max_output_tokens: int | None = None
    system_instruction: str | None = None
    response_mime_type: str | None = None
    response_json_schema: dict[str, Any] | None = None
    thinking_level: str | None = None  # "LOW", "MEDIUM", "HIGH", or None (model default)


@dataclasses.dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""

    text: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    model_id: str = ""
    finish_reason: str | None = None
    raw_response: Any = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass
class EmbeddingResponse:
    """Standardised embedding response."""

    values: list[float]
    model_id: str = ""
    raw_response: Any = dataclasses.field(default=None, repr=False)


# ── Tool-use types ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    """Provider-agnostic tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclasses.dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclasses.dataclass
class ToolResult:
    """Result of executing a tool call."""

    tool_call_id: str
    content: str


@dataclasses.dataclass
class ToolUseResponse:
    """Response from generate_with_tools.

    When the model wants to call tools: tool_calls is non-empty, text is None.
    When the model produces a final answer: tool_calls is empty, text is set.
    """

    text: str | None
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    model_id: str = ""
    finish_reason: str | None = None
    raw_response: Any = dataclasses.field(default=None, repr=False)
    raw_assistant_content: Any = dataclasses.field(default=None, repr=False)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
