"""LLM providers: one interface, several HTTP backends, no SDK dependencies."""

from __future__ import annotations

from crucible.llm.base import Completion, LLMProvider, Message, estimate_cost
from crucible.llm.providers import (
    AnthropicProvider,
    EchoProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ScriptedProvider,
)
from crucible.llm.registry import (
    available_providers,
    build_provider,
    register_provider,
)

__all__ = [
    "AnthropicProvider",
    "Completion",
    "EchoProvider",
    "LLMProvider",
    "Message",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ScriptedProvider",
    "available_providers",
    "build_provider",
    "estimate_cost",
    "register_provider",
]
