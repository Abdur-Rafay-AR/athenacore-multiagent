"""Concrete providers: Ollama, OpenAI-compatible, Anthropic, and two fakes.

The OpenAI-compatible provider is the workhorse — the same class talks to OpenAI,
Groq, Together, DeepSeek, OpenRouter, vLLM, llama.cpp and LM Studio, since they
all expose ``/chat/completions``. Point ``ATHENA_OPENAI_BASE_URL`` at any of them.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Sequence
from typing import Any

from athenacore.errors import ProviderError
from athenacore.llm.base import (
    Completion,
    LLMProvider,
    Message,
    iter_sse_data,
)
from athenacore.memory.models import Usage, estimate_tokens


class OllamaProvider(LLMProvider):
    """Local models through the Ollama daemon (the zero-cost default)."""

    name = "ollama"
    supports_streaming = True

    def __init__(self, model: str, *, host: str = "http://localhost:11434", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")

    def connection_hint(self) -> str:
        return (
            f"Is Ollama running? Try `ollama serve`, then `ollama pull {self.model}`. "
            f"Override the endpoint with ATHENA_OLLAMA_HOST (currently {self.host})."
        )

    def _payload(self, messages: Sequence[Message], stream: bool, **options: Any) -> dict[str, Any]:
        opts = self.options(**options)
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
            "options": {
                "temperature": opts["temperature"],
                "num_predict": opts["max_output_tokens"],
            },
        }

    def _complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        data = self._post_json(
            f"{self.host}/api/chat", self._payload(messages, stream=False, **options)
        )
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text and data.get("error"):
            raise ProviderError(str(data["error"]), provider=self.name, hint=self.connection_hint())
        return Completion(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
                calls=1,
                latency_ms=int((data.get("total_duration") or 0) / 1_000_000),
            ),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    def _stream(self, messages: Sequence[Message], **options: Any) -> Iterator[str]:
        # Ollama streams newline-delimited JSON rather than SSE.
        for raw in self._post_stream(
            f"{self.host}/api/chat", self._payload(messages, stream=True, **options)
        ):
            try:
                chunk = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("message") or {}).get("content")
            if delta:
                yield delta
            if chunk.get("done"):
                break

    def list_models(self) -> list[str]:
        """Locally pulled models, used to populate the UI's model picker."""
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))


class OpenAICompatibleProvider(LLMProvider):
    """Any endpoint speaking the OpenAI ``/chat/completions`` schema."""

    name = "openai"
    supports_streaming = True

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def connection_hint(self) -> str:
        return f"Check ATHENA_OPENAI_BASE_URL (currently {self.base_url}) and ATHENA_OPENAI_API_KEY."

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: Sequence[Message], stream: bool, **options: Any) -> dict[str, Any]:
        opts = self.options(**options)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": opts["temperature"],
            "max_tokens": opts["max_output_tokens"],
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        if not self.api_key and "api.openai.com" in self.base_url:
            raise ProviderError(
                "no API key configured for OpenAI",
                provider=self.name,
                hint="Set ATHENA_OPENAI_API_KEY, or use a local model: ATHENA_MODEL=ollama:llama3.1",
            )
        data = self._post_json(
            f"{self.base_url}/chat/completions",
            self._payload(messages, stream=False, **options),
            headers=self._headers(),
        )
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("response contained no choices", provider=self.name, retryable=True)
        choice = choices[0]
        usage_raw = data.get("usage") or {}
        return Completion(
            text=(choice.get("message") or {}).get("content", "").strip(),
            model=data.get("model", self.model),
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                calls=1,
            ),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def _stream(self, messages: Sequence[Message], **options: Any) -> Iterator[str]:
        stream = self._post_stream(
            f"{self.base_url}/chat/completions",
            self._payload(messages, stream=True, **options),
            headers=self._headers(),
        )
        for payload in iter_sse_data(stream):
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    yield delta


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API.

    Two schema differences from OpenAI are handled here: the system prompt is a
    top-level field rather than a message, and ``max_tokens`` is required.
    """

    name = "anthropic"
    supports_streaming = True
    api_version = "2023-06-01"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def connection_hint(self) -> str:
        return "Set ATHENA_ANTHROPIC_API_KEY, or switch to a local model."

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
        }

    def _payload(self, messages: Sequence[Message], stream: bool, **options: Any) -> dict[str, Any]:
        opts = self.options(**options)
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m.to_dict() for m in messages if m.role != "system"]
        if not turns:
            turns = [{"role": "user", "content": system or "Continue."}]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": opts["max_output_tokens"],
            "temperature": opts["temperature"],
            "stream": stream,
        }
        if system:
            payload["system"] = system
        return payload

    def _complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        if not self.api_key:
            raise ProviderError(
                "no API key configured for Anthropic",
                provider=self.name,
                hint="Set ATHENA_ANTHROPIC_API_KEY.",
            )
        data = self._post_json(
            f"{self.base_url}/messages",
            self._payload(messages, stream=False, **options),
            headers=self._headers(),
        )
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        usage_raw = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", self.model),
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(usage_raw.get("input_tokens") or 0),
                completion_tokens=int(usage_raw.get("output_tokens") or 0),
                calls=1,
            ),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    def _stream(self, messages: Sequence[Message], **options: Any) -> Iterator[str]:
        stream = self._post_stream(
            f"{self.base_url}/messages",
            self._payload(messages, stream=True, **options),
            headers=self._headers(),
        )
        for payload in iter_sse_data(stream):
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = (event.get("delta") or {}).get("text")
                if delta:
                    yield delta


class EchoProvider(LLMProvider):
    """Deterministic offline provider.

    This is what makes the project testable and demoable with no model, no daemon
    and no network. It does not pretend to be intelligent — it reflects structure
    back — but it exercises every code path: prompts, streaming, tool protocol
    parsing, usage accounting, memory writes and compaction.

    Set ``ATHENA_MODEL=echo:test`` to run the whole system offline.
    """

    name = "echo"
    supports_streaming = True

    def __init__(self, model: str = "test", *, latency_ms: int = 0, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.latency_ms = latency_ms
        self.calls: list[list[Message]] = []

    def _complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        self.calls.append(list(messages))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        text = self._render(messages)
        return Completion(
            text=text,
            model=self.model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
                completion_tokens=estimate_tokens(text),
                calls=1,
            ),
            finish_reason="stop",
        )

    def _stream(self, messages: Sequence[Message], **options: Any) -> Iterator[str]:
        for word in self._complete(messages, **options).text.split(" "):
            yield word + " "

    def _render(self, messages: Sequence[Message]) -> str:
        """Produce a plausible-shaped answer keyed off the instruction.

        The output echoes salient nouns from the prompt so that memory recall,
        de-duplication and convergence detection all have real signal to work on
        in tests rather than constant strings.
        """
        system = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        role = _first_line(system) or "agent"
        subject = _keywords(user, limit=6)
        lines = [f"[{self.name}] {role} responding to: {subject or 'the topic'}."]
        lowered = f"{system}\n{user}".lower()
        if "counterargument" in lowered or "devil" in lowered or "critique" in lowered:
            lines += [
                f"- Risk: the claim about {subject or 'the topic'} rests on unstated assumptions.",
                "- Missing: no base rate or comparison case is offered.",
            ]
        elif "summar" in lowered or "condens" in lowered or "compact" in lowered:
            lines += [f"- {subject or 'The topic'} is the central thread.", "- Open questions remain."]
        elif "insight" in lowered or "takeaway" in lowered:
            lines += [f"- Leverage point: {subject or 'the topic'}.", "- Second-order effect noted."]
        else:
            lines += [f"- Finding on {subject or 'the topic'}.", "- Confidence: moderate."]
        return "\n".join(lines)


class ScriptedProvider(LLMProvider):
    """Returns pre-written responses in order. For tests asserting exact text."""

    name = "scripted"

    def __init__(self, responses: Sequence[str], *, model: str = "scripted", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.responses = list(responses)
        self.index = 0
        self.calls: list[list[Message]] = []

    def _complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        self.calls.append(list(messages))
        if not self.responses:
            raise ProviderError("scripted provider has no responses left", provider=self.name)
        text = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return Completion(text=text, model=self.model, provider=self.name, finish_reason="stop")


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{3,}")
_NOISE = frozenset(
    """this that with from have been will your their about which would could should
    prior memory entries agent agents topic content respond response answer question
    following based please rules write plain none only more most also than then when
    where what does using used into over under while very just they them there here
    system user assistant tokens""".split()
)


def _keywords(text: str, *, limit: int = 6) -> str:
    seen: list[str] = []
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        if lowered in _NOISE or lowered in seen:
            continue
        seen.append(lowered)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""
