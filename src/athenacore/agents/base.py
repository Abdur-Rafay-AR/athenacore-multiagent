"""The agent base class.

Every agent runs the same cycle, defined once here:

1. **Recall** - pull relevant memory for this topic using the agent's own recall
   policy (which kinds it cares about, how much budget it gets).
2. **Compose** - build a system prompt (role + tools + output contract) and a
   user prompt (recalled memory + upstream agent outputs + the task).
3. **Generate** - call the provider, then run the tool loop until the model stops
   asking for tools or hits the per-turn cap.
4. **Persist** - write the result to shared memory as a typed entry, tagged with
   the run, model, latency and any citations it made.

Subclasses normally override only :attr:`role`, :attr:`system_prompt`,
:attr:`entry_kind` and :attr:`recall_policy`. That constraint is the point: it
keeps agents to a few dozen lines each and makes the interesting behaviour:
memory, tools, orchestration - shared rather than copy-pasted.
"""

from __future__ import annotations

import abc
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from athenacore.errors import ProviderError
from athenacore.llm.base import LLMProvider, Message
from athenacore.logging_setup import get_logger
from athenacore.memory.models import Entry, EntryKind, Usage
from athenacore.memory.retrieval import MemoryRetriever, RecallResult
from athenacore.memory.store import MemoryStore
from athenacore.tools.base import ToolRegistry, ToolResult, parse_tool_calls, strip_tool_calls

log = get_logger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(slots=True)
class RecallPolicy:
    """How much and what sort of memory an agent gets.

    Different roles genuinely want different context: a critic should see the
    claims to attack, a summarizer should see everything, a planner mostly wants
    conclusions. Encoding that per-agent measurably improves output quality over
    dumping the whole log into every prompt.
    """

    kinds: tuple[EntryKind, ...] | None = None
    max_entries: int = 10
    token_budget: int = 3000
    use_query: bool = True
    """Whether the task text is used as the retrieval query, or recall is purely
    recency/salience driven."""

    include_own_previous: bool = True
    """When False, the agent does not see its own prior output - used to stop
    critics from anchoring on their earlier critiques."""


@dataclass(slots=True)
class AgentContext:
    """Everything an agent needs for one turn."""

    topic: str
    task: str = ""
    run_id: str | None = None
    upstream: dict[str, str] = field(default_factory=dict)
    """Outputs of the nodes this one depends on, keyed by node name."""

    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def upstream_block(self) -> str:
        if not self.upstream:
            return ""
        parts = [
            f"--- FROM {name.upper()} ---\n{text.strip()}"
            for name, text in self.upstream.items()
            if text and text.strip()
        ]
        return "\n\n".join(parts)


@dataclass(slots=True)
class AgentResult:
    """What an agent produced, plus its full provenance."""

    agent: str
    content: str
    kind: EntryKind
    entry: Entry | None = None
    usage: Usage = field(default_factory=Usage)
    recall: RecallResult | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "kind": self.kind.value,
            "content": self.content,
            "entry_id": self.entry.id if self.entry else None,
            "usage": self.usage.to_dict(),
            "tools": [t.to_dict() for t in self.tool_results],
            "citations": self.citations,
            "recalled": len(self.recall) if self.recall else 0,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class Agent(abc.ABC):
    """Base class for all agents."""

    name: str = "agent"
    role: str = "Generalist"
    description: str = ""
    entry_kind: EntryKind = EntryKind.NOTE
    default_salience: float = 0.5
    temperature: float | None = None
    """Per-agent temperature override; ``None`` inherits the provider default."""

    recall_policy: RecallPolicy = RecallPolicy()
    uses_tools: bool = True
    requires_task: bool = False
    """When True, the agent refuses to run without a task/query."""

    def __init__(
        self,
        provider: LLMProvider,
        store: MemoryStore,
        *,
        retriever: MemoryRetriever | None = None,
        tools: ToolRegistry | None = None,
        max_tool_calls: int = 4,
        name: str | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.retriever = retriever or MemoryRetriever(store)
        self.tools = tools if tools is not None else ToolRegistry()
        self.max_tool_calls = max_tool_calls
        if name:
            self.name = name

    # -- prompt surface (subclasses override) --------------------------------

    @property
    @abc.abstractmethod
    def system_prompt(self) -> str:
        """The role instruction. First line should name the role - the offline
        Echo provider and the logs both key off it."""

    def build_task_prompt(self, ctx: AgentContext, recall: RecallResult) -> str:
        """Assemble the user-turn prompt. Override for role-specific framing."""
        sections: list[str] = []
        memory_block = recall.as_prompt_context()
        if memory_block:
            sections.append(memory_block)
        upstream = ctx.upstream_block()
        if upstream:
            sections.append(f"### THIS ROUND\n\n{upstream}")
        sections.append(f"### YOUR TASK\n\n{self.task_instruction(ctx)}")
        return "\n\n".join(sections)

    def task_instruction(self, ctx: AgentContext) -> str:
        """The concrete ask. Default: the caller's task, or a topic-level nudge."""
        if ctx.task.strip():
            return ctx.task.strip()
        return f"Advance the team's understanding of: {ctx.topic}"

    # -- execution -----------------------------------------------------------

    def run(self, ctx: AgentContext) -> AgentResult:
        """Execute one turn. Never raises: failures come back as a result with
        ``error`` set, so one bad node cannot abort a whole graph."""
        started = time.monotonic()
        if self.requires_task and not ctx.task.strip():
            return AgentResult(
                agent=self.name,
                content="",
                kind=self.entry_kind,
                error=f"{self.name} requires a task/query but none was given",
            )

        recall = self._recall(ctx)
        messages = [
            Message.system(self._full_system_prompt()),
            Message.user(self.build_task_prompt(ctx, recall)),
        ]

        try:
            text, usage, tool_results = self._generate(messages)
        except ProviderError as exc:
            log.error("agent failed", extra={"agent": self.name, "error": exc.message})
            return AgentResult(
                agent=self.name,
                content="",
                kind=self.entry_kind,
                error=str(exc),
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        text = self.postprocess(text, ctx)
        duration_ms = int((time.monotonic() - started) * 1000)
        citations = self._resolve_citations(text, recall)

        entry = None
        if text:
            entry = self.store.add_entry(
                Entry(
                    topic=ctx.topic,
                    agent=self.name,
                    kind=self.entry_kind,
                    content=text,
                    run_id=ctx.run_id,
                    salience=self.salience_for(text, ctx),
                    model=f"{self.provider.name}:{self.provider.model}",
                    latency_ms=duration_ms,
                    tokens=usage.completion_tokens or 0,
                    metadata={
                        "role": self.role,
                        "round": ctx.round_index,
                        "cited_entry_ids": citations,
                        "recalled_entry_ids": [s.entry.id for s in recall.entries],
                        "tools_used": [t.call.name for t in tool_results],
                    },
                )
            )
        return AgentResult(
            agent=self.name,
            content=text,
            kind=self.entry_kind,
            entry=entry,
            usage=usage,
            recall=recall,
            tool_results=tool_results,
            citations=citations,
            duration_ms=duration_ms,
        )

    def _recall(self, ctx: AgentContext) -> RecallResult:
        policy = self.recall_policy
        query = ctx.task if (policy.use_query and ctx.task.strip()) else ctx.topic
        exclude: list[str] = []
        if not policy.include_own_previous:
            from athenacore.memory.store import EntryFilter

            exclude = [
                e.id
                for e in self.store.query_entries(
                    EntryFilter(topic=ctx.topic, agents=[self.name], limit=50)
                )
            ]
        return self.retriever.recall(
            ctx.topic,
            query,
            kinds=policy.kinds,
            exclude_ids=exclude,
            max_entries=policy.max_entries,
            token_budget=policy.token_budget,
        )

    def _full_system_prompt(self) -> str:
        parts = [self.system_prompt.strip()]
        if self.uses_tools and self.tools:
            parts.append(self.tools.prompt_section())
        parts.append(self.output_contract())
        return "\n\n".join(p for p in parts if p)

    def output_contract(self) -> str:
        """Shared formatting rules. Explicit contracts matter more for small
        local models than for frontier ones, so they are stated bluntly."""
        return (
            "### OUTPUT RULES\n"
            "- Write the substance only. No preamble, no sign-off, no restating the task.\n"
            "- Cite prior memory by its bracketed number, e.g. [2], when you rely on it.\n"
            "- If the memory does not support a claim, say so rather than inventing detail.\n"
            "- Be specific: prefer names, numbers and dates over general statements."
        )

    def _generate(self, messages: list[Message]) -> tuple[str, Usage, list[ToolResult]]:
        """Call the model, servicing tool requests until it stops asking.

        The loop is bounded twice over - by ``max_tool_calls`` and by the fact
        that each iteration must produce at least one new call - so a model stuck
        in a tool loop costs a fixed, small number of turns.
        """
        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        completion = self.provider.complete(messages, **options)
        usage = completion.usage
        text = completion.text
        results: list[ToolResult] = []

        if not (self.uses_tools and self.tools):
            return text.strip(), usage, results

        budget = self.max_tool_calls
        while budget > 0:
            calls = parse_tool_calls(text)
            if not calls:
                break
            batch = calls[:budget]
            batch_results = self.tools.execute_all(batch, limit=len(batch))
            results.extend(batch_results)
            budget -= len(batch_results)

            observations = "\n\n".join(r.as_observation() for r in batch_results)
            messages = [
                *messages,
                Message.assistant(text),
                Message.user(
                    f"{observations}\n\nContinue. Use these observations to complete your task. "
                    "Do not call the same tool with the same arguments again."
                ),
            ]
            completion = self.provider.complete(messages, **options)
            usage = usage + completion.usage
            text = completion.text

        return strip_tool_calls(text), usage, results

    def postprocess(self, text: str, ctx: AgentContext) -> str:
        """Last chance to clean output before it is persisted.

        Strips the "Sure, here is..." preamble that instruction-tuned models add
        despite being told not to.
        """
        cleaned = text.strip()
        cleaned = re.sub(
            r"^(sure|certainly|of course|here(?:'s| is)|absolutely)[^\n]{0,80}[:\n]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def salience_for(self, text: str, ctx: AgentContext) -> float:
        """Importance of this contribution. Overridable per role."""
        return self.default_salience

    def _resolve_citations(self, text: str, recall: RecallResult) -> list[str]:
        """Map ``[n]`` markers back to the entry ids they refer to."""
        if not recall:
            return []
        mapping = recall.citation_map()
        seen: list[str] = []
        for match in CITATION_RE.finditer(text):
            entry_id = mapping.get(int(match.group(1)))
            if entry_id and entry_id not in seen:
                seen.append(entry_id)
        return seen

    # -- introspection -------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description or self.__doc__ or "",
            "kind": self.entry_kind.value,
            "uses_tools": self.uses_tools and bool(self.tools),
            "recall": {
                "kinds": [k.value for k in (self.recall_policy.kinds or ())],
                "max_entries": self.recall_policy.max_entries,
                "token_budget": self.recall_policy.token_budget,
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} name={self.name!r}>"


def format_transcript(results: Sequence[AgentResult]) -> str:
    """Render several agent results as a readable transcript (CLI and exports)."""
    blocks = []
    for result in results:
        header = f"## {result.agent} ({result.kind.value})"
        body = result.content if result.ok else f"[failed] {result.error}"
        blocks.append(f"{header}\n\n{body}")
    return "\n\n".join(blocks)
