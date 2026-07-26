"""Built-in tools.

Each is deliberately narrow and side-effect free by default. The calculator
evaluates arithmetic through a whitelisted AST walk rather than ``eval``, which is
the only responsible way to let model output near an expression evaluator.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from athenacore.errors import ToolError
from athenacore.logging_setup import get_logger
from athenacore.memory.models import EntryKind
from athenacore.memory.store import MemoryStore
from athenacore.tools.base import Tool

log = get_logger(__name__)


# -- calculator --------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *args: sum(args),
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

# Guard against expressions that are cheap to type and expensive to evaluate,
# e.g. 9**9**9.
_MAX_POW_EXPONENT = 1000


class CalculatorTool(Tool):
    """Arithmetic the model should not be trusted to do in its head."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression (+ - * / // % **, sqrt, log, min, max, round, pi, e)."
    )
    parameters = {"expression": {"type": "string", "description": "e.g. '1.4e6 * 0.47 / 12'"}}
    required = ("expression",)

    def run(self, **kwargs: Any) -> str:
        expression = str(kwargs["expression"]).strip()
        if len(expression) > 500:
            raise ToolError("expression too long", tool=self.name)
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"cannot parse {expression!r}: {exc.msg}", tool=self.name) from exc
        try:
            value = self._eval(tree.body)
        except ToolError:
            raise
        except ZeroDivisionError as exc:
            raise ToolError("division by zero", tool=self.name) from exc
        except (ValueError, OverflowError, TypeError) as exc:
            raise ToolError(f"cannot evaluate: {exc}", tool=self.name) from exc
        rendered = f"{value:.10g}" if isinstance(value, float) else str(value)
        return f"{expression} = {rendered}"

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ToolError("only numeric literals are allowed", tool=self.name)
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ToolError(f"operator {type(node.op).__name__} is not allowed", tool=self.name)
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
                raise ToolError("exponent too large", tool=self.name)
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ToolError("unary operator not allowed", tool=self.name)
            return op(self._eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ToolError("only whitelisted functions may be called", tool=self.name)
            if node.keywords:
                raise ToolError("keyword arguments are not supported", tool=self.name)
            return _FUNCTIONS[node.func.id](*[self._eval(a) for a in node.args])
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ToolError(f"unknown name {node.id!r}", tool=self.name)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e) for e in node.elts)
        raise ToolError(f"expression element {type(node).__name__} is not allowed", tool=self.name)


# -- memory search -----------------------------------------------------------


class MemorySearchTool(Tool):
    """Lets an agent query the shared memory beyond what was auto-recalled.

    This is the tool that makes the memory *collaborative* rather than merely
    persistent: an agent can go look for what someone else established earlier,
    including in other topics.
    """

    name = "memory_search"
    description = (
        "Search the team's persistent memory for prior findings. Returns matching entries."
    )
    parameters = {
        "query": {"type": "string", "description": "what to look for"},
        "topic": {"type": "string", "description": "optional topic to restrict to"},
        "limit": {"type": "integer", "description": "max results (default 5)"},
    }
    required = ("query",)

    def __init__(self, store: MemoryStore, *, default_topic: str | None = None) -> None:
        self.store = store
        self.default_topic = default_topic

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs["query"]).strip()
        topic = kwargs.get("topic") or self.default_topic
        limit = max(1, min(10, int(kwargs.get("limit") or 5)))
        hits = self.store.keyword_search(query, topic=topic, limit=limit)
        if not hits:
            return f"No memory entries matched {query!r}."
        lines = [f"{len(hits)} match(es) for {query!r}:"]
        for entry, score in hits:
            stamp = entry.created_at.strftime("%Y-%m-%d")
            lines.append(
                f"- [{entry.agent} · {entry.kind.value} · {stamp} · rel {score:.2f}] {entry.preview(400)}"
            )
        return "\n".join(lines)


class MemoryWriteTool(Tool):
    """Lets an agent deliberately record a durable fact or decision.

    Marked unsafe because it mutates shared state: enable it only for agents you
    want to have write authority.
    """

    name = "remember"
    description = "Record an important fact or decision in long-term memory so future runs see it."
    parameters = {
        "content": {"type": "string", "description": "the fact to remember"},
        "kind": {"type": "string", "description": "note | decision | insight"},
        "salience": {"type": "number", "description": "importance 0-1 (default 0.8)"},
    }
    required = ("content",)
    safe = False

    def __init__(self, store: MemoryStore, *, topic: str, agent: str = "agent") -> None:
        self.store = store
        self.topic = topic
        self.agent = agent

    def run(self, **kwargs: Any) -> str:
        from athenacore.memory.models import Entry

        raw_kind = str(kwargs.get("kind") or "note").lower()
        try:
            kind = EntryKind(raw_kind)
        except ValueError:
            kind = EntryKind.NOTE
        entry = self.store.add_entry(
            Entry(
                topic=self.topic,
                agent=self.agent,
                kind=kind,
                content=str(kwargs["content"]),
                salience=float(kwargs.get("salience") or 0.8),
                tags=("agent-authored",),
            )
        )
        return f"Recorded as {entry.id} ({kind.value})."


# -- time --------------------------------------------------------------------


class ClockTool(Tool):
    """Models have no clock, and 'current' is a load-bearing word in research."""

    name = "now"
    description = "Get the current UTC date and time."
    parameters = {}

    def run(self, **kwargs: Any) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%d %H:%M UTC (%A)")


# -- web search --------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


class WebSearchTool(Tool):
    """Web search via DuckDuckGo.

    Off by default (``ATHENA_WEB_SEARCH_ENABLED=true`` to enable) because network
    egress should be opt-in. Uses the ``duckduckgo-search`` package when
    installed and falls back to the dependency-free Instant Answer endpoint,
    which is weaker but needs nothing extra.
    """

    name = "web_search"
    description = "Search the web for current information. Returns titles, snippets and URLs."
    parameters = {
        "query": {"type": "string", "description": "search query"},
        "limit": {"type": "integer", "description": "max results (default 5)"},
    }
    required = ("query",)
    safe = False

    def __init__(self, *, timeout_s: float = 15.0) -> None:
        self.timeout_s = timeout_s

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs["query"]).strip()
        limit = max(1, min(10, int(kwargs.get("limit") or 5)))
        if not query:
            raise ToolError("query cannot be empty", tool=self.name)
        results = self._search_package(query, limit) or self._search_instant_answer(query, limit)
        if not results:
            return f"No web results for {query!r}."
        lines = [f"Web results for {query!r}:"]
        for i, item in enumerate(results, start=1):
            lines.append(f"{i}. {item['title']}\n   {item['snippet']}\n   {item['url']}")
        return "\n".join(lines)

    def _search_package(self, query: str, limit: int) -> list[dict[str, str]]:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError:
            return []
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=limit))
        except Exception as exc:
            log.warning("web search failed", extra={"error": str(exc)})
            return []
        return [
            {
                "title": item.get("title", "")[:200],
                "snippet": _clean(item.get("body", ""))[:400],
                "url": item.get("href", ""),
            }
            for item in raw
        ]

    def _search_instant_answer(self, query: str, limit: int) -> list[dict[str, str]]:
        url = (
            "https://api.duckduckgo.com/?q="
            + urllib.parse.quote(query)
            + "&format=json&no_html=1&no_redirect=1"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "athenacore/0.2"})
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ToolError(f"web search unavailable: {exc}", tool=self.name) from exc

        results: list[dict[str, str]] = []
        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading") or query,
                    "snippet": _clean(data["AbstractText"])[:400],
                    "url": data.get("AbstractURL", ""),
                }
            )
        for item in data.get("RelatedTopics") or []:
            if len(results) >= limit:
                break
            if item.get("Text") and item.get("FirstURL"):
                results.append(
                    {
                        "title": item["Text"].split(" - ")[0][:200],
                        "snippet": _clean(item["Text"])[:400],
                        "url": item["FirstURL"],
                    }
                )
        return results


def _clean(text: str) -> str:
    return " ".join(_TAG_RE.sub("", text).split())


def default_registry(
    store: MemoryStore | None = None,
    *,
    topic: str | None = None,
    web_search: bool = False,
    writable: bool = False,
    extra: Sequence[Tool] = (),
):
    """Assemble the standard tool set for a run.

    Kept as a function rather than a module constant so each run gets tools bound
    to its own store and topic.
    """
    from athenacore.tools.base import ToolRegistry

    tools: list[Tool] = [CalculatorTool(), ClockTool()]
    if store is not None:
        tools.append(MemorySearchTool(store, default_topic=topic))
        if writable and topic:
            tools.append(MemoryWriteTool(store, topic=topic))
    if web_search:
        tools.append(WebSearchTool())
    tools.extend(extra)
    return ToolRegistry(tools, allow_unsafe=web_search or writable)
