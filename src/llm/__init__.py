"""Provider-agnostic LLM interface.

Usage::

    from src.llm import generate, embed
    from src.llm.types import GenerationConfig

    resp = generate("Explain triage.", model_id="gemini-2.5-flash")
    print(resp.text)

    emb = embed("emergency medicine", model_id="text-embedding-005")
    print(len(emb.values))

Providers are resolved automatically from the model-id prefix:
  - ``gemini-*`` / ``text-embedding-*``  → Google Vertex AI
  - ``claude-*``                         → Anthropic
  - ``moonshot-*`` / ``kimi-*``          → Kimi (Moonshot AI)

Custom providers can be added via :func:`register_prefix`.
"""

from src.llm.types import (
    EmbeddingResponse,
    GenerationConfig,
    LLMResponse,
    ToolDefinition,
    ToolUseResponse,
)
from src.llm.registry import (
    register_prefix,
    register_provider,
    resolve_provider,
    reset,
)


def generate(
    prompt: str,
    *,
    model_id: str,
    config: GenerationConfig | None = None,
) -> LLMResponse:
    """Generate text from *prompt* using the provider for *model_id*."""
    provider = resolve_provider(model_id)
    return provider.generate(prompt, model_id=model_id, config=config)


def embed(
    text: str,
    *,
    model_id: str,
) -> EmbeddingResponse:
    """Return an embedding vector for *text* using the provider for *model_id*."""
    provider = resolve_provider(model_id)
    return provider.embed(text, model_id=model_id)


def generate_with_tools(
    messages: list[dict],
    *,
    model_id: str,
    tools: list[ToolDefinition],
    config: GenerationConfig | None = None,
) -> ToolUseResponse:
    """Generate with tool-use support using the provider for *model_id*."""
    provider = resolve_provider(model_id)
    return provider.generate_with_tools(
        messages, model_id=model_id, tools=tools, config=config,
    )


__all__ = [
    "generate",
    "generate_with_tools",
    "embed",
    "GenerationConfig",
    "LLMResponse",
    "EmbeddingResponse",
    "ToolDefinition",
    "ToolUseResponse",
    "register_prefix",
    "register_provider",
    "resolve_provider",
    "reset",
]
