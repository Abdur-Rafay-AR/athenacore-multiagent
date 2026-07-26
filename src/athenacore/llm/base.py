"""Provider abstraction.

One interface, four backends, no SDK dependencies: everything speaks HTTP through
``urllib``. The base class owns the parts that are identical for every provider:
retry with jittered exponential backoff, timeout handling, usage accounting,
streaming-to-blocking fallback - so a new provider is roughly 40 lines.
"""

from __future__ import annotations

import abc
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from athenacore.errors import ProviderError, ProviderTimeout
from athenacore.logging_setup import get_logger
from athenacore.memory.models import Usage, estimate_tokens

log = get_logger(__name__)

# Retryable HTTP statuses: rate limits and transient server faults.
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


@dataclass(slots=True)
class Message:
    """One chat message. ``role`` is ``system``, ``user`` or ``assistant``."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls("assistant", content)


@dataclass(slots=True)
class Completion:
    """A finished generation plus everything worth measuring about it."""

    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text

    @property
    def truncated(self) -> bool:
        """Whether the model stopped because it hit the output cap.

        Worth surfacing: a truncated critique reads like a confident one.
        """
        return self.finish_reason in {"length", "max_tokens", "max_output_tokens"}


class LLMProvider(abc.ABC):
    """Base class for chat-completion backends."""

    name: str = "provider"
    supports_streaming: bool = False

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.4,
        max_output_tokens: int = 1024,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.5,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.backoff_s = backoff_s
        self.total_usage = Usage()

    # -- to implement --------------------------------------------------------

    @abc.abstractmethod
    def _complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        """Perform one request. Raise :class:`ProviderError` on failure."""

    def _stream(self, messages: Sequence[Message], **options: Any) -> Iterator[str]:
        """Yield text deltas. Providers that can stream should override this."""
        raise NotImplementedError

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe used by ``athenacore doctor``."""
        try:
            self.complete([Message.user("ping")], max_output_tokens=8)
            return True, "reachable"
        except ProviderError as exc:
            return False, str(exc)

    # -- public API ----------------------------------------------------------

    def complete(self, messages: Sequence[Message], **options: Any) -> Completion:
        """Generate once, retrying transient failures.

        Backoff is exponential with full jitter. Jitter matters more than it looks
        for this workload: a graph fans several agents out at the same instant, and
        un-jittered retries would keep them synchronised into the same rate limit.
        """
        attempt = 0
        last: ProviderError | None = None
        while attempt <= self.max_retries:
            started = time.monotonic()
            try:
                completion = self._complete(messages, **options)
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == self.max_retries:
                    raise
                delay = self.backoff_s * (2**attempt) * (0.5 + random.random() / 2)
                log.warning(
                    "provider call failed, retrying",
                    extra={
                        "provider": self.name,
                        "attempt": attempt + 1,
                        "of": self.max_retries,
                        "delay_s": round(delay, 2),
                        "error": exc.message,
                    },
                )
                time.sleep(delay)
                attempt += 1
                continue

            elapsed_ms = int((time.monotonic() - started) * 1000)
            if not completion.usage.latency_ms:
                completion.usage.latency_ms = elapsed_ms
            if not completion.usage.calls:
                completion.usage.calls = 1
            self._fill_usage(completion, messages)
            self.total_usage = self.total_usage + completion.usage
            log.debug(
                "completion",
                extra={
                    "provider": self.name,
                    "model": completion.model,
                    "tokens": completion.usage.total_tokens,
                    "ms": completion.usage.latency_ms,
                },
            )
            return completion

        raise last or ProviderError("provider exhausted retries", provider=self.name)

    def stream(self, messages: Sequence[Message], **options: Any) -> Iterator[str]:
        """Yield text deltas, falling back to one chunk for blocking providers.

        Callers can therefore always write streaming code, and non-streaming
        backends simply produce a single delta.
        """
        if not self.supports_streaming:
            yield self.complete(messages, **options).text
            return
        try:
            yield from self._stream(messages, **options)
        except NotImplementedError:  # pragma: no cover - defensive
            yield self.complete(messages, **options).text

    def _fill_usage(self, completion: Completion, messages: Sequence[Message]) -> None:
        """Estimate any token counts the provider did not report."""
        if not completion.usage.prompt_tokens:
            completion.usage.prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        if not completion.usage.completion_tokens:
            completion.usage.completion_tokens = estimate_tokens(completion.text)
        if not completion.usage.cost_usd:
            completion.usage.cost_usd = estimate_cost(
                self.model, completion.usage.prompt_tokens, completion.usage.completion_tokens
            )

    # -- HTTP helpers --------------------------------------------------------

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _read_error(exc)
            raise ProviderError(
                f"{self.name} returned HTTP {exc.code}: {detail}",
                provider=self.name,
                status=exc.code,
                retryable=exc.code in _RETRY_STATUS,
                hint=_status_hint(exc.code),
            ) from exc
        except TimeoutError as exc:
            raise ProviderTimeout(
                f"{self.name} timed out after {self.timeout_s:.0f}s", provider=self.name
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"cannot reach {self.name} at {url}: {exc.reason}",
                provider=self.name,
                retryable=True,
                hint=self.connection_hint(),
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{self.name} returned a non-JSON body", provider=self.name, retryable=True
            ) from exc

    def _post_stream(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> Iterator[bytes]:
        """POST and yield raw response lines (for SSE / JSON-lines streams)."""
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for line in response:
                    if line.strip():
                        yield line
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"{self.name} returned HTTP {exc.code}: {_read_error(exc)}",
                provider=self.name,
                status=exc.code,
                retryable=exc.code in _RETRY_STATUS,
                hint=_status_hint(exc.code),
            ) from exc
        except TimeoutError as exc:
            raise ProviderTimeout(
                f"{self.name} stream timed out after {self.timeout_s:.0f}s", provider=self.name
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"cannot reach {self.name} at {url}: {exc.reason}",
                provider=self.name,
                retryable=True,
                hint=self.connection_hint(),
            ) from exc

    def connection_hint(self) -> str:
        return "Check the endpoint URL and that the service is running."

    # -- misc ----------------------------------------------------------------

    def options(self, **overrides: Any) -> dict[str, Any]:
        """Merge per-call overrides over the provider defaults."""
        merged = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} model={self.model!r}>"


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # pragma: no cover - stream already consumed
        return exc.reason or "unknown error"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:400]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if data.get("message"):
            return str(data["message"])
    return raw[:400]


def _status_hint(status: int) -> str | None:
    if status in {401, 403}:
        return "Check the API key for this provider (see .env.example)."
    if status == 404:
        return "The model name is probably wrong, or not pulled locally."
    if status == 429:
        return "Rate limited. Lower ATHENA_MAX_PARALLEL_AGENTS or wait."
    if status >= 500:
        return "Upstream fault; retries are automatic."
    return None


# Published per-million-token prices for common models, used for run cost
# estimates. Unknown models simply report zero rather than guessing.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o3-mini": (1.10, 4.40),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (1.00, 5.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort USD cost. Locally hosted models are free, so this returns 0.0
    for anything not in the price table."""
    key = model.lower()
    prices = _PRICES_PER_MTOK.get(key)
    if prices is None:
        prices = next((v for k, v in _PRICES_PER_MTOK.items() if key.startswith(k)), None)
    if prices is None:
        return 0.0
    prompt_rate, completion_rate = prices
    return round(
        (prompt_tokens / 1_000_000) * prompt_rate
        + (completion_tokens / 1_000_000) * completion_rate,
        6,
    )


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Extract ``data:`` payloads from a Server-Sent Events byte stream."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                yield payload
