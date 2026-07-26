"""Exception hierarchy.

Every error raised on purpose by AthenaCore derives from :class:`AthenaCoreError`,
so embedders can catch one type at their boundary. Errors carry a ``hint`` where
there is a plausible next action for the operator, because most failures in this
system are environmental (model not pulled, daemon not running, key missing).
"""

from __future__ import annotations


class AthenaCoreError(Exception):
    """Base class for all AthenaCore errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\n  hint: {self.hint}"
        return self.message


class ConfigurationError(AthenaCoreError):
    """Settings are missing, contradictory, or unparseable."""


class ProviderError(AthenaCoreError):
    """An LLM provider could not produce a completion."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
        retryable: bool = False,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.provider = provider
        self.status = status
        self.retryable = retryable


class ProviderTimeout(ProviderError):
    """The provider did not answer inside the configured timeout."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, retryable=True)


class MemoryError_(AthenaCoreError):
    """Persistence layer failure. Named with a trailing underscore to avoid
    shadowing the builtin ``MemoryError``."""


class TopicNotFound(MemoryError_):
    """The requested topic does not exist in the store."""


class ToolError(AthenaCoreError):
    """A tool refused or failed to execute."""

    def __init__(self, message: str, *, tool: str | None = None) -> None:
        super().__init__(message)
        self.tool = tool


class GraphError(AthenaCoreError):
    """The agent graph is malformed (cycle, unknown dependency, empty)."""


class RunCancelled(AthenaCoreError):
    """A run was cancelled through its cancellation token."""
