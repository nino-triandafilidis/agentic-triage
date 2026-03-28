"""Google Vertex AI (Gemini) provider."""

from __future__ import annotations

import logging
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


def _safety_settings_off() -> list:
    """Return safety settings that disable all content filtering.

    Medical ED content (trauma, assault, bleeding) triggers Gemini's safety
    filters, causing silent empty-candidate responses.  Disabling filters
    eliminates these parse failures in a clinical decision-support context.
    """
    from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

    categories = [
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]
    return [
        SafetySetting(category=c, threshold=HarmBlockThreshold.OFF)
        for c in categories
    ]


class GoogleProvider(LLMProvider):
    """Wraps the ``google-genai`` SDK for Vertex AI Gemini models.

    Auth uses GCP Application Default Credentials (ADC), consistent
    with the existing ``src.config`` setup.
    """

    def __init__(self, *, project: str | None = None, location: str = "global"):
        self._project = project
        self._location = location
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        from google import genai

        project = self._project
        if project is None:
            import src.config as config
            config._ensure_resolved()
            project = config._project_id
            credentials = config._credentials
        else:
            from google.auth import default
            credentials, _ = default()

        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=self._location,
            credentials=credentials,
        )
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        model_id: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        from google.genai.types import GenerateContentConfig

        client = self._ensure_client()
        cfg = config or GenerationConfig()

        genai_cfg_kwargs: dict[str, Any] = {
            "temperature": cfg.temperature,
            "safety_settings": _safety_settings_off(),
        }
        if cfg.max_output_tokens is not None:
            genai_cfg_kwargs["max_output_tokens"] = cfg.max_output_tokens
        if cfg.system_instruction is not None:
            genai_cfg_kwargs["system_instruction"] = cfg.system_instruction
        if cfg.response_mime_type is not None:
            genai_cfg_kwargs["response_mime_type"] = cfg.response_mime_type
        if cfg.response_json_schema is not None:
            genai_cfg_kwargs["response_json_schema"] = cfg.response_json_schema
        if cfg.thinking_level:
            from google.genai.types import ThinkingConfig
            genai_cfg_kwargs["thinking_config"] = ThinkingConfig(thinking_level=cfg.thinking_level)

        resp = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=GenerateContentConfig(**genai_cfg_kwargs),
        )

        prompt_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0
        if resp.usage_metadata:
            prompt_tokens = resp.usage_metadata.prompt_token_count or 0
            completion_tokens = resp.usage_metadata.candidates_token_count or 0
            thinking_tokens = getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0

        finish_reason = None
        if resp.candidates:
            finish_reason = str(resp.candidates[0].finish_reason)

        return LLMResponse(
            text=resp.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            model_id=model_id,
            finish_reason=finish_reason,
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
        from google.genai.types import (
            Content,
            FunctionDeclaration,
            FunctionResponse,
            GenerateContentConfig,
            Part,
            Tool,
        )

        client = self._ensure_client()
        cfg = config or GenerationConfig()

        contents = _convert_messages_for_google(messages)

        function_declarations = [
            FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.input_schema,
            )
            for t in tools
        ]
        google_tools = [Tool(function_declarations=function_declarations)]

        genai_cfg_kwargs: dict[str, Any] = {
            "temperature": cfg.temperature,
            "tools": google_tools,
            "safety_settings": _safety_settings_off(),
        }
        if cfg.max_output_tokens is not None:
            genai_cfg_kwargs["max_output_tokens"] = cfg.max_output_tokens
        if cfg.system_instruction is not None:
            genai_cfg_kwargs["system_instruction"] = cfg.system_instruction
        if cfg.response_mime_type is not None:
            genai_cfg_kwargs["response_mime_type"] = cfg.response_mime_type
        if cfg.response_json_schema is not None:
            genai_cfg_kwargs["response_json_schema"] = cfg.response_json_schema
        if cfg.thinking_level:
            from google.genai.types import ThinkingConfig
            genai_cfg_kwargs["thinking_config"] = ThinkingConfig(thinking_level=cfg.thinking_level)

        resp = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=GenerateContentConfig(**genai_cfg_kwargs),
        )

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        if resp.candidates:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            tool_call_id=fc.name,  # Google uses name as ID
                            tool_name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )
                elif part.text:
                    text_parts.append(part.text)

        prompt_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0
        if resp.usage_metadata:
            prompt_tokens = resp.usage_metadata.prompt_token_count or 0
            completion_tokens = resp.usage_metadata.candidates_token_count or 0
            thinking_tokens = (
                getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0
            )

        finish_reason = None
        if resp.candidates:
            finish_reason = str(resp.candidates[0].finish_reason)

        raw_text = "".join(text_parts) or None

        # Preserve the raw Content object so multi-turn tool-use can replay
        # it verbatim — required by Gemini 3+ which includes thought_signature
        # fields that must be echoed back.
        raw_content = resp.candidates[0].content if resp.candidates else None

        return ToolUseResponse(
            text=raw_text if not tool_calls else None,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            model_id=model_id,
            finish_reason=finish_reason,
            raw_response=resp,
            raw_assistant_content=raw_content,
        )

    def embed(self, text: str, *, model_id: str) -> EmbeddingResponse:
        client = self._ensure_client()
        resp = client.models.embed_content(model=model_id, contents=text)
        return EmbeddingResponse(
            values=resp.embeddings[0].values,
            model_id=model_id,
            raw_response=resp,
        )

    def is_available(self) -> bool:
        try:
            from google import genai as _  # noqa: F401
            return True
        except ImportError:
            return False


def _convert_messages_for_google(messages: list[dict]) -> list:
    """Convert generic messages to Google Content format.

    Google requires that the number of FunctionResponse parts in a Content
    message equals the number of FunctionCall parts in the preceding model
    message.  When the model issues N function calls in one turn, the
    caller appends N consecutive ``role="tool"`` messages.  We must group
    those into a **single** ``Content(role="user", parts=[...])`` with all
    N ``FunctionResponse`` parts — otherwise the API returns
    INVALID_ARGUMENT.
    """
    from google.genai.types import Content, FunctionCall, FunctionResponse, Part

    contents: list = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user":
            contents.append(
                Content(role="user", parts=[Part(text=msg["content"])])
            )

        elif msg["role"] == "assistant":
            # Prefer raw Content object (preserves thought_signature for Gemini 3+)
            raw = msg.get("_raw_content")
            if raw is not None:
                contents.append(raw)
            else:
                parts: list = []
                if msg.get("content"):
                    parts.append(Part(text=msg["content"]))
                for tc in msg.get("tool_calls", []):
                    parts.append(
                        Part(
                            function_call=FunctionCall(
                                name=tc["tool_name"],
                                args=tc["arguments"],
                            )
                        )
                    )
                contents.append(Content(role="model", parts=parts))

        elif msg["role"] == "tool":
            # Collect all consecutive tool-result messages into one Content
            # so the FunctionResponse count matches the FunctionCall count.
            tool_parts: list = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tr = messages[i]
                tool_parts.append(
                    Part(
                        function_response=FunctionResponse(
                            name=tr.get("tool_name", "search_esi_case_bank"),
                            response={"result": tr["content"]},
                        )
                    )
                )
                i += 1
            contents.append(Content(role="user", parts=tool_parts))
            continue  # skip the i += 1 below

        i += 1
    return contents
