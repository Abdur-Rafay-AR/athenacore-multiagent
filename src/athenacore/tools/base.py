"""Tool protocol.

Native function calling is unevenly supported across local models, and the whole
point of this project is that it runs on a laptop with Ollama. So tools use a
plain-text protocol that any instruction-following model can produce:

    TOOL: calculator {"expression": "1.4e6 * 0.47"}

The agent loop parses those lines, executes the tools, appends the results as an
observation, and asks the model to continue. Simple, debuggable, and it degrades
gracefully — a model that ignores tools entirely still produces a valid answer.

Tools declare a JSON-schema-shaped spec, so the same registry can be handed to a
native function-calling backend later without changing any tool code.
"""

from __future__ import annotations

import abc
import json
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from athenacore.errors import ToolError
from athenacore.logging_setup import get_logger

log = get_logger(__name__)

TOOL_CALL_RE = re.compile(
    r"^\s*TOOL\s*:\s*(?P<name>[a-z_][a-z0-9_]*)\s*(?P<args>\{.*\})?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(slots=True)
class ToolCall:
    """A parsed request to run a tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass(slots=True)
class ToolResult:
    """The outcome of running a tool. Failures are values, not exceptions.

    Returning errors as results is deliberate: a bad tool call should let the
    model see what went wrong and try again, not abort the whole run.
    """

    call: ToolCall
    output: str
    ok: bool = True
    error: str | None = None
    duration_ms: int = 0

    def as_observation(self) -> str:
        status = "OK" if self.ok else "ERROR"
        body = self.output if self.ok else (self.error or "unknown error")
        return f"OBSERVATION ({self.call.name}, {status}):\n{body}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.call.name,
            "arguments": self.call.arguments,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class Tool(abc.ABC):
    """A capability an agent can invoke."""

    name: str = "tool"
    description: str = ""
    parameters: dict[str, Any] = {}
    """JSON-Schema ``properties`` block describing the arguments."""

    required: tuple[str, ...] = ()
    safe: bool = True
    """``False`` for tools with side effects or cost; the registry can gate these."""

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute and return output as text. Raise :class:`ToolError` to fail."""

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function schema, for native tool-calling backends."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.required),
                },
            },
        }

    def usage_line(self) -> str:
        """One line of prompt documentation for the text protocol."""
        args = ", ".join(
            f'"{key}": <{meta.get("type", "any")}>' for key, meta in self.parameters.items()
        )
        return f"TOOL: {self.name} {{{args}}}  — {self.description}"

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check required keys and drop unknown ones.

        Unknown keys are dropped rather than rejected because models routinely
        add a plausible extra field, and failing the call over it wastes a turn.
        """
        missing = [key for key in self.required if key not in arguments]
        if missing:
            raise ToolError(f"missing required argument(s): {', '.join(missing)}", tool=self.name)
        if self.parameters:
            return {k: v for k, v in arguments.items() if k in self.parameters}
        return arguments


class ToolRegistry:
    """The tools available to a run."""

    def __init__(self, tools: Iterable[Tool] = (), *, allow_unsafe: bool = False) -> None:
        self._tools: dict[str, Tool] = {}
        self.allow_unsafe = allow_unsafe
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def prompt_section(self) -> str:
        """The block injected into a system prompt to teach the text protocol."""
        if not self._tools:
            return ""
        lines = [
            "### TOOLS",
            "You may call a tool by writing a line exactly in this form, and nothing else on that line:",
            "",
            'TOOL: <name> {"arg": "value"}',
            "",
            "Available tools:",
        ]
        lines += [f"- {tool.usage_line()}" for tool in self._tools.values()]
        lines += [
            "",
            "Rules: call at most one tool per line. After a call, stop and wait — the",
            "result arrives as an OBSERVATION and you may then continue or answer.",
            "If no tool is needed, just answer directly.",
        ]
        return "\n".join(lines)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one call, converting every failure into a :class:`ToolResult`."""
        started = time.monotonic()
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call=call,
                output="",
                ok=False,
                error=f"unknown tool {call.name!r}; available: {', '.join(self.names()) or 'none'}",
            )
        if not tool.safe and not self.allow_unsafe:
            return ToolResult(
                call=call, output="", ok=False, error=f"tool {call.name!r} is disabled by policy"
            )
        try:
            arguments = tool.validate(dict(call.arguments))
            output = tool.run(**arguments)
            ok, error = True, None
        except ToolError as exc:
            output, ok, error = "", False, exc.message
        except Exception as exc:  # a buggy tool must not kill the run
            log.warning("tool raised", extra={"tool": call.name, "error": str(exc)})
            output, ok, error = "", False, f"{type(exc).__name__}: {exc}"
        duration = int((time.monotonic() - started) * 1000)
        return ToolResult(
            call=call,
            output=str(output)[:8000],
            ok=ok,
            error=error,
            duration_ms=duration,
        )

    def execute_all(self, calls: Sequence[ToolCall], *, limit: int = 4) -> list[ToolResult]:
        return [self.execute(call) for call in calls[:limit]]


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract ``TOOL:`` lines from model output.

    Tolerant by design: single quotes, trailing commas and a missing argument
    object are all accepted, because rejecting a nearly-right call costs a full
    round trip. Anything unsalvageable is skipped rather than raising.
    """
    calls: list[ToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group("name").lower()
        raw_args = (match.group("args") or "").strip()
        arguments: dict[str, Any] = {}
        if raw_args:
            arguments = _loose_json(raw_args)
            if arguments is None:
                continue
        calls.append(ToolCall(name=name, arguments=arguments, raw=match.group(0).strip()))
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove ``TOOL:`` lines so they never leak into stored memory."""
    return TOOL_CALL_RE.sub("", text).strip()


def _loose_json(raw: str) -> dict[str, Any] | None:
    for candidate in (raw, raw.replace("'", '"'), re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
