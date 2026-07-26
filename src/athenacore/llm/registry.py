"""Provider resolution.

Turns a spec string plus :class:`~athenacore.config.Settings` into a live
provider, so the model backend is a one-line config change everywhere::

    ATHENA_MODEL=ollama:llama3.1          # local, free, default
    ATHENA_MODEL=openai:gpt-4o-mini       # or any OpenAI-compatible endpoint
    ATHENA_MODEL=groq:llama-3.3-70b       # same schema, different base URL
    ATHENA_MODEL=anthropic:claude-sonnet-4-5
    ATHENA_MODEL=echo:test                # offline, deterministic
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from athenacore.config import Settings, split_model_spec
from athenacore.errors import ConfigurationError
from athenacore.llm.base import LLMProvider
from athenacore.llm.providers import (
    AnthropicProvider,
    EchoProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ScriptedProvider,
)

ProviderFactory = Callable[[str, Settings], LLMProvider]

# Read once so alias factories can tell "user set a base URL" from "still default".
_DEFAULT_OPENAI_BASE_URL = Settings().openai_base_url

# Hosted services that speak the OpenAI schema. Registering them as aliases means
# `ATHENA_MODEL=groq:llama-3.3-70b` works without touching a base URL by hand.
OPENAI_COMPATIBLE_ALIASES: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
    "llamacpp": "http://localhost:8080/v1",
}


def _common(settings: Settings) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "timeout_s": settings.request_timeout_s,
        "max_retries": settings.max_retries,
        "backoff_s": settings.retry_backoff_s,
    }


def _build_ollama(model: str, settings: Settings) -> LLMProvider:
    return OllamaProvider(model, host=settings.ollama_host, **_common(settings))


def _build_openai(model: str, settings: Settings) -> LLMProvider:
    return OpenAICompatibleProvider(
        model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        **_common(settings),
    )


def _build_anthropic(model: str, settings: Settings) -> LLMProvider:
    return AnthropicProvider(
        model,
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        **_common(settings),
    )


def _build_echo(model: str, settings: Settings) -> LLMProvider:
    return EchoProvider(model, **_common(settings))


def _build_scripted(model: str, settings: Settings) -> LLMProvider:
    # Responses are supplied programmatically; the spec only names the provider.
    return ScriptedProvider([], model=model, **_common(settings))


_REGISTRY: dict[str, ProviderFactory] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "claude": _build_anthropic,
    "echo": _build_echo,
    "mock": _build_echo,
    "scripted": _build_scripted,
}

for _alias, _url in OPENAI_COMPATIBLE_ALIASES.items():

    def _make(alias_url: str = _url) -> ProviderFactory:
        def factory(model: str, settings: Settings) -> LLMProvider:
            # An explicitly configured base URL always wins over the alias default.
            base = (
                settings.openai_base_url
                if settings.openai_base_url != _DEFAULT_OPENAI_BASE_URL
                else alias_url
            )
            return OpenAICompatibleProvider(
                model, api_key=settings.openai_api_key, base_url=base, **_common(settings)
            )

        return factory

    _REGISTRY[_alias] = _make()


def register_provider(name: str, factory: ProviderFactory, *, overwrite: bool = False) -> None:
    """Add a provider so ``ATHENA_MODEL=<name>:<model>`` resolves to it."""
    key = name.strip().lower()
    if key in _REGISTRY and not overwrite:
        raise ConfigurationError(f"provider {key!r} is already registered")
    _REGISTRY[key] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def build_provider(spec: str | None = None, settings: Settings | None = None) -> LLMProvider:
    """Resolve ``"provider:model"`` into a ready provider instance."""
    settings = settings or Settings()
    provider_name, model_name = split_model_spec(spec or settings.model)
    factory = _REGISTRY.get(provider_name)
    if factory is None:
        raise ConfigurationError(
            f"unknown provider {provider_name!r}",
            hint=f"Available: {', '.join(available_providers())}",
        )
    return factory(model_name, settings)
