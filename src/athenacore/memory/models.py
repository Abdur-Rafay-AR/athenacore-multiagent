"""The persisted domain model.

Three record types carry everything the system remembers:

``Topic``
    A long-lived thread of collaboration. Memory is scoped per topic.
``Entry``
    One immutable contribution to a topic: an agent answer, a critique, a
    summary, a user note. Entries are append-only; "editing" means writing a new
    entry that supersedes an older one.
``Run``
    One execution of an agent graph against a topic, with its usage totals.

Append-only entries are what make the memory auditable: you can always replay
how a topic's understanding evolved, which is the point of the project.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    """Timezone-aware current time. Never use naive datetimes in the store."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Short, sortable-enough, human-greppable identifier (``run_9f3c1a2b``)."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class EntryKind(str, Enum):
    """What sort of contribution an entry is.

    The kind drives three behaviours: which entries an agent prefers to recall,
    how the UI renders them, and whether compaction may fold them away
    (:attr:`SUMMARY` and :attr:`DECISION` are protected).
    """

    NOTE = "note"           # human-authored context
    QUESTION = "question"   # the prompt that started a run
    RESEARCH = "research"   # sourced findings
    SUMMARY = "summary"     # condensed prior state (protected from compaction)
    CRITIQUE = "critique"   # counterarguments, risks, holes
    INSIGHT = "insight"     # strategic takeaway
    SYNTHESIS = "synthesis" # reconciliation of conflicting views
    PLAN = "plan"           # proposed next steps
    DECISION = "decision"   # a conclusion worth protecting (protected)
    TOOL = "tool"           # raw tool output
    ERROR = "error"         # a failure worth remembering

    @property
    def protected(self) -> bool:
        """Whether compaction must preserve this entry verbatim."""
        return self in {EntryKind.SUMMARY, EntryKind.DECISION, EntryKind.NOTE}

    @property
    def icon(self) -> str:
        return _KIND_ICONS.get(self, "•")


_KIND_ICONS = {
    EntryKind.NOTE: "📝",
    EntryKind.QUESTION: "❓",
    EntryKind.RESEARCH: "🔎",
    EntryKind.SUMMARY: "🧾",
    EntryKind.CRITIQUE: "⚔️",
    EntryKind.INSIGHT: "💡",
    EntryKind.SYNTHESIS: "🧩",
    EntryKind.PLAN: "🗺️",
    EntryKind.DECISION: "✅",
    EntryKind.TOOL: "🔧",
    EntryKind.ERROR: "🚨",
}


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"    # some nodes failed but the run produced output
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self not in {RunStatus.PENDING, RunStatus.RUNNING}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for budgeting when no tokenizer is available.

    Roughly 4 characters per token for English prose, with a word-count floor so
    that whitespace-heavy text (lists, code) is not under-counted. It is only
    ever used for budget decisions, never billing, so being within ~15% is fine.
    """
    if not text:
        return 0
    return max(len(text) // 4, len(text.split()), 1)


@dataclass(slots=True)
class Entry:
    """An immutable contribution to a topic's memory."""

    topic: str
    agent: str
    content: str
    kind: EntryKind = EntryKind.NOTE
    id: str = field(default_factory=lambda: new_id("ent"))
    run_id: str | None = None
    parent_id: str | None = None
    """The entry this one responds to, if any. Enables threaded views."""

    created_at: datetime = field(default_factory=utcnow)
    salience: float = 0.5
    """Operator- or agent-assigned importance in ``[0, 1]``; feeds recall ranking."""

    archived: bool = False
    """Set when compaction supersedes this entry. Archived entries are retained
    for audit but excluded from recall by default."""

    superseded_by: str | None = None
    tokens: int = 0
    model: str | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = EntryKind(self.kind)
        self.content = self.content.strip()
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)
        self.salience = min(1.0, max(0.0, float(self.salience)))

    # -- views ---------------------------------------------------------------

    def preview(self, width: int = 160) -> str:
        flat = " ".join(self.content.split())
        return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"

    def as_prompt_block(self, *, index: int | None = None) -> str:
        """Render this entry the way agents see it inside a prompt."""
        label = f"[{index}] " if index is not None else ""
        stamp = self.created_at.strftime("%Y-%m-%d %H:%M")
        return f"{label}{self.agent} · {self.kind.value} · {stamp}\n{self.content}"

    def with_(self, **changes: Any) -> Entry:
        return replace(self, **changes)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "agent": self.agent,
            "kind": self.kind.value,
            "content": self.content,
            "run_id": self.run_id,
            "parent_id": self.parent_id,
            "created_at": to_iso(self.created_at),
            "salience": self.salience,
            "archived": self.archived,
            "superseded_by": self.superseded_by,
            "tokens": self.tokens,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entry:
        raw_meta = data.get("metadata") or {}
        if isinstance(raw_meta, str):
            raw_meta = json.loads(raw_meta or "{}")
        raw_tags = data.get("tags") or ()
        if isinstance(raw_tags, str):
            raw_tags = tuple(t for t in raw_tags.split(",") if t)
        created = data.get("created_at")
        return cls(
            id=data.get("id") or new_id("ent"),
            topic=data["topic"],
            agent=data["agent"],
            content=data["content"],
            kind=EntryKind(data.get("kind", EntryKind.NOTE.value)),
            run_id=data.get("run_id"),
            parent_id=data.get("parent_id"),
            created_at=from_iso(created) if isinstance(created, str) else (created or utcnow()),
            salience=float(data.get("salience", 0.5)),
            archived=bool(data.get("archived", False)),
            superseded_by=data.get("superseded_by"),
            tokens=int(data.get("tokens") or 0),
            model=data.get("model"),
            latency_ms=data.get("latency_ms"),
            metadata=dict(raw_meta),
            tags=tuple(raw_tags),
        )


@dataclass(slots=True)
class Topic:
    """A named, long-lived collaboration thread."""

    name: str
    id: str = field(default_factory=lambda: new_id("top"))
    description: str = ""
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    entry_count: int = 0
    live_tokens: int = 0
    """Tokens across non-archived entries; drives the compaction trigger."""

    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": to_iso(self.created_at),
            "updated_at": to_iso(self.updated_at),
            "entry_count": self.entry_count,
            "live_tokens": self.live_tokens,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Usage:
    """Token/latency/cost accounting, summable across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
            latency_ms=self.latency_ms + other.latency_ms,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            calls=int(data.get("calls", 0)),
            latency_ms=int(data.get("latency_ms", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
        )


@dataclass(slots=True)
class Run:
    """One execution of a graph against a topic."""

    topic: str
    preset: str
    id: str = field(default_factory=lambda: new_id("run"))
    query: str = ""
    status: RunStatus = RunStatus.PENDING
    model: str = ""
    started_at: datetime = field(default_factory=utcnow)
    ended_at: datetime | None = None
    usage: Usage = field(default_factory=Usage)
    node_states: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        end = self.ended_at or utcnow()
        return int((end - self.started_at).total_seconds() * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "preset": self.preset,
            "query": self.query,
            "status": self.status.value,
            "model": self.model,
            "started_at": to_iso(self.started_at),
            "ended_at": to_iso(self.ended_at) if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "usage": self.usage.to_dict(),
            "node_states": self.node_states,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        return cls(
            id=data["id"],
            topic=data["topic"],
            preset=data.get("preset", ""),
            query=data.get("query", ""),
            status=RunStatus(data.get("status", RunStatus.PENDING.value)),
            model=data.get("model", ""),
            started_at=from_iso(data["started_at"]),
            ended_at=from_iso(data["ended_at"]) if data.get("ended_at") else None,
            usage=Usage.from_dict(data.get("usage") or {}),
            node_states=dict(data.get("node_states") or {}),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )
