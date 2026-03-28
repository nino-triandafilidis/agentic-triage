"""Anthropic (Claude) provider."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.llm.base import LLMProvider
from src.llm.types import (
    EmbeddingResponse,
    GenerationConfig,
    LLMResponse,
    ToolCall,
    ToolDefinition,
    ToolUseResponse,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Wraps the ``anthropic`` SDK for Claude models.

    Auth via the ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(self, *, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic
        import httpx

        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        cfg = config or GenerationConfig()

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": cfg.max_output_tokens or 4096,
            "temperature": cfg.temperature,
        }
        if cfg.system_instruction:
            kwargs["system"] = cfg.system_instruction

        if cfg.response_json_schema is not None:
            schema = _normalize_schema_for_anthropic(cfg.response_json_schema)
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }

        resp = client.messages.create(**kwargs)

        text_parts: list[str] = []
        thinking_tokens = 0
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_tokens += getattr(block, "thinking_tokens", 0)

        raw_text = "".join(text_parts) or None

        prompt_tokens = resp.usage.input_tokens if resp.usage else 0
        completion_tokens = resp.usage.output_tokens if resp.usage else 0

        return LLMResponse(
            text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            model_id=model_id,
            finish_reason=resp.stop_reason,
            raw_response=resp,
        )

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        tools: list[ToolDefinition],
        config: GenerationConfig | None = None,
    ) -> ToolUseResponse:
        client = self._ensure_client()
        cfg = config or GenerationConfig()

        anthropic_messages = _convert_messages_for_anthropic(messages)
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            "max_tokens": cfg.max_output_tokens or 4096,
            "temperature": cfg.temperature,
        }
        if cfg.system_instruction:
            kwargs["system"] = cfg.system_instruction

        if cfg.response_json_schema is not None:
            schema = _normalize_schema_for_anthropic(cfg.response_json_schema)
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }

        resp = client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_tokens = 0
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        arguments=block.input,
                    )
                )
            elif block.type == "thinking":
                thinking_tokens += getattr(block, "thinking_tokens", 0)

        prompt_tokens = resp.usage.input_tokens if resp.usage else 0
        completion_tokens = resp.usage.output_tokens if resp.usage else 0
        raw_text = "".join(text_parts) or None

        return ToolUseResponse(
            text=raw_text if not tool_calls else None,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            model_id=model_id,
            finish_reason=resp.stop_reason,
            raw_response=resp,
        )

    def is_available(self) -> bool:
        try:
            import anthropic as _  # noqa: F401
            return bool(self._api_key or os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            return False


_UNSUPPORTED_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                      "minLength", "maxLength", "pattern", "minItems", "maxItems"}


def _normalize_schema_for_anthropic(schema: dict) -> dict:
    """Adapt a JSON schema for Anthropic structured outputs.

    - Adds ``additionalProperties: false`` on all object types (required).
    - Strips validation keywords unsupported by Anthropic (minimum, maximum, etc.).
    """
    schema = {k: v for k, v in schema.items() if k not in _UNSUPPORTED_KEYS}
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for key in ("properties", "$defs", "definitions"):
        if key in schema:
            container = schema[key]
            if isinstance(container, dict):
                schema[key] = {
                    k: _normalize_schema_for_anthropic(v) if isinstance(v, dict) else v
                    for k, v in container.items()
                }
    return schema


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from model output.

    Claude may wrap JSON in markdown fences or add preamble text;
    this helper strips that so callers get raw JSON.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    for start_marker in ("```json", "```"):
        if start_marker in stripped:
            after = stripped.split(start_marker, 1)[1]
            json_str = after.split("```", 1)[0].strip()
            if json_str:
                return json_str

    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = stripped[brace_start : brace_end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return stripped


def _convert_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert generic messages to Anthropic's format.

    Our format:
      {"role": "user",      "content": "text"}
      {"role": "assistant", "content": "text", "tool_calls": [...]}
      {"role": "tool",      "tool_call_id": "...", "content": "result"}

    Anthropic format:
      {"role": "user",      "content": "text"}
      {"role": "assistant", "content": [{"type": "tool_use", ...}]}
      {"role": "user",      "content": [{"type": "tool_result", ...}]}
    """
    result: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user" and isinstance(msg["content"], str):
            result.append({"role": "user", "content": msg["content"]})

        elif msg["role"] == "assistant":
            content_blocks: list[dict] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["tool_call_id"],
                        "name": tc["tool_name"],
                        "input": tc["arguments"],
                    }
                )
            result.append({"role": "assistant", "content": content_blocks})

        elif msg["role"] == "tool":
            # Anthropic sends tool results as user messages; collect consecutive
            tool_results: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tr = messages[i]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tr["tool_call_id"],
                        "content": tr["content"],
                    }
                )
                i += 1
            result.append({"role": "user", "content": tool_results})
            continue  # skip the i += 1 below

        i += 1
    return result
