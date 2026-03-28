"""Kimi / Moonshot AI provider.

Moonshot exposes an OpenAI-compatible chat-completions API,
so this provider uses the ``openai`` SDK with a custom base URL.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.llm.base import LLMProvider
from src.llm.types import EmbeddingResponse, GenerationConfig, LLMResponse

logger = logging.getLogger(__name__)

KIMI_BASE_URL = "https://api.moonshot.cn/v1"


class KimiProvider(LLMProvider):
    """Wraps the ``openai`` SDK pointed at the Moonshot API.

    Auth via the ``MOONSHOT_API_KEY`` environment variable.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = KIMI_BASE_URL,
    ):
        self._api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        self._base_url = base_url
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        self._client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
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

        messages: list[dict[str, str]] = []
        if cfg.system_instruction:
            messages.append({"role": "system", "content": cfg.system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": cfg.temperature,
        }
        if cfg.max_output_tokens is not None:
            kwargs["max_tokens"] = cfg.max_output_tokens

        if cfg.response_json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None

        raw_text = choice.message.content if choice else None
        finish_reason = choice.finish_reason if choice else None

        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        return LLMResponse(
            text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=0,
            model_id=model_id,
            finish_reason=finish_reason,
            raw_response=resp,
        )

    def is_available(self) -> bool:
        try:
            import openai as _  # noqa: F401
            return bool(self._api_key or os.environ.get("MOONSHOT_API_KEY"))
        except ImportError:
            return False
