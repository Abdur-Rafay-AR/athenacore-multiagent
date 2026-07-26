"""The memory store interface.

Agents never touch SQL. They talk to this interface, which means the SQLite
implementation can be swapped for Postgres, Redis or an in-memory fake without
touching a single agent. :class:`InMemoryMemoryStore` at the bottom is the fake
used by the test suite.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from athenacore.errors import TopicNotFound
from athenacore.memory.models import Entry, EntryKind, Run, Topic, utcnow


@dataclass(slots=True)
class EntryFilter:
    """Declarative query over entries. All fields AND together."""

    topic: str | None = None
    kinds: Sequence[EntryKind] | None = None
    agents: Sequence[str] | None = None
    run_id: str | None = None
    include_archived: bool = False
    since: datetime | None = None
    limit: int = 200
    offset: int = 0
    newest_first: bool = True

    def matches(self, entry: Entry) -> bool:
        """Python-side evaluation, used by the in-memory store and as the
        reference semantics the SQL builder must reproduce."""
        if self.topic is not None and entry.topic != self.topic:
            return False
        if self.kinds and entry.kind not in set(self.kinds):
            return False
        if self.agents and entry.agent not in set(self.agents):
            return False
        if self.run_id is not None and entry.run_id != self.run_id:
            return False
        if not self.include_archived and entry.archived:
            return False
        return not (self.since is not None and entry.created_at < self.since)


@dataclass(slots=True)
class ScoredEntry:
    """An entry plus the ranking signals that selected it.

    The breakdown is kept rather than collapsed into one number so the UI can
    explain *why* a memory surfaced — which is the difference between a system an
    operator trusts and a black box.
    """

    entry: Entry
    score: float
    keyword: float = 0.0
    semantic: float = 0.0
    recency: float = 0.0
    salience: float = 0.0

    def explain(self) -> str:
        return (
            f"score={self.score:.3f} (kw={self.keyword:.2f} "
            f"sem={self.semantic:.2f} rec={self.recency:.2f} sal={self.salience:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": round(self.score, 4),
            "signals": {
                "keyword": round(self.keyword, 4),
                "semantic": round(self.semantic, 4),
                "recency": round(self.recency, 4),
                "salience": round(self.salience, 4),
            },
        }


class MemoryStore(abc.ABC):
    """Persistence contract for topics, entries and runs."""

    # -- topics --------------------------------------------------------------

    @abc.abstractmethod
    def ensure_topic(self, name: str, *, description: str = "", tags: Sequence[str] = ()) -> Topic:
        """Return the topic, creating it if absent. Idempotent."""

    @abc.abstractmethod
    def get_topic(self, name: str) -> Topic | None: ...

    @abc.abstractmethod
    def list_topics(self, *, limit: int = 100, search: str | None = None) -> list[Topic]: ...

    @abc.abstractmethod
    def delete_topic(self, name: str) -> int:
        """Delete a topic and its entries. Returns the number of entries removed."""

    def require_topic(self, name: str) -> Topic:
        topic = self.get_topic(name)
        if topic is None:
            raise TopicNotFound(f"topic {name!r} does not exist")
        return topic

    # -- entries -------------------------------------------------------------

    @abc.abstractmethod
    def add_entry(self, entry: Entry) -> Entry:
        """Persist one entry (also creating its topic if needed) and return it."""

    def add_entries(self, entries: Iterable[Entry]) -> list[Entry]:
        return [self.add_entry(e) for e in entries]

    @abc.abstractmethod
    def get_entry(self, entry_id: str) -> Entry | None: ...

    @abc.abstractmethod
    def query_entries(self, spec: EntryFilter) -> list[Entry]: ...

    @abc.abstractmethod
    def keyword_search(
        self, query: str, *, topic: str | None = None, limit: int = 50
    ) -> list[tuple[Entry, float]]:
        """Full-text search. Returns ``(entry, relevance)`` with relevance in
        ``[0, 1]``, highest first. Implementations without a text index may fall
        back to substring matching."""

    @abc.abstractmethod
    def archive_entries(self, entry_ids: Sequence[str], *, superseded_by: str | None = None) -> int:
        """Mark entries as archived so recall skips them. Returns rows touched."""

    @abc.abstractmethod
    def set_salience(self, entry_id: str, salience: float) -> None: ...

    def timeline(
        self, topic: str, *, limit: int = 200, include_archived: bool = True
    ) -> list[Entry]:
        """Chronological history of a topic, oldest first — the audit view."""
        return self.query_entries(
            EntryFilter(
                topic=topic,
                include_archived=include_archived,
                limit=limit,
                newest_first=False,
            )
        )

    # -- embeddings ----------------------------------------------------------

    @abc.abstractmethod
    def set_embedding(self, entry_id: str, vector: Sequence[float]) -> None: ...

    @abc.abstractmethod
    def get_embeddings(self, entry_ids: Sequence[str]) -> dict[str, list[float]]: ...

    @abc.abstractmethod
    def entries_missing_embeddings(
        self, *, topic: str | None = None, limit: int = 500
    ) -> list[Entry]: ...

    # -- runs ----------------------------------------------------------------

    @abc.abstractmethod
    def save_run(self, run: Run) -> Run:
        """Insert or update a run by id."""

    @abc.abstractmethod
    def get_run(self, run_id: str) -> Run | None: ...

    @abc.abstractmethod
    def list_runs(self, *, topic: str | None = None, limit: int = 50) -> list[Run]: ...

    # -- aggregate -----------------------------------------------------------

    @abc.abstractmethod
    def stats(self, *, topic: str | None = None) -> dict[str, Any]:
        """Counts and usage totals for dashboards."""

    def close(self) -> None:  # optional hook: not every store holds resources
        """Release resources. Safe to call more than once."""
        return None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InMemoryMemoryStore(MemoryStore):
    """Dict-backed store. Useful for tests, demos and ephemeral sessions.

    Semantics match :class:`~athenacore.memory.sqlite_store.SqliteMemoryStore`
    closely enough that the shared conformance test suite runs against both.
    """

    def __init__(self) -> None:
        self._topics: dict[str, Topic] = {}
        self._entries: dict[str, Entry] = {}
        self._runs: dict[str, Run] = {}
        self._vectors: dict[str, list[float]] = {}

    # topics
    def ensure_topic(self, name: str, *, description: str = "", tags: Sequence[str] = ()) -> Topic:
        existing = self._topics.get(name)
        if existing:
            if description and not existing.description:
                existing.description = description
            if tags:
                existing.tags = tuple(dict.fromkeys((*existing.tags, *tags)))
            return existing
        topic = Topic(name=name, description=description, tags=tuple(tags))
        self._topics[name] = topic
        return topic

    def get_topic(self, name: str) -> Topic | None:
        return self._topics.get(name)

    def list_topics(self, *, limit: int = 100, search: str | None = None) -> list[Topic]:
        items = list(self._topics.values())
        if search:
            needle = search.lower()
            items = [
                t for t in items if needle in t.name.lower() or needle in t.description.lower()
            ]
        items.sort(key=lambda t: t.updated_at, reverse=True)
        return items[:limit]

    def delete_topic(self, name: str) -> int:
        removed = [e for e in self._entries.values() if e.topic == name]
        for entry in removed:
            self._entries.pop(entry.id, None)
            self._vectors.pop(entry.id, None)
        self._topics.pop(name, None)
        for run_id in [r.id for r in self._runs.values() if r.topic == name]:
            self._runs.pop(run_id, None)
        return len(removed)

    # entries
    def add_entry(self, entry: Entry) -> Entry:
        topic = self.ensure_topic(entry.topic)
        self._entries[entry.id] = entry
        topic.entry_count += 1
        topic.live_tokens += entry.tokens
        topic.updated_at = utcnow()
        return entry

    def get_entry(self, entry_id: str) -> Entry | None:
        return self._entries.get(entry_id)

    def query_entries(self, spec: EntryFilter) -> list[Entry]:
        rows = [e for e in self._entries.values() if spec.matches(e)]
        rows.sort(key=lambda e: e.created_at, reverse=spec.newest_first)
        return rows[spec.offset : spec.offset + spec.limit]

    def keyword_search(
        self, query: str, *, topic: str | None = None, limit: int = 50
    ) -> list[tuple[Entry, float]]:
        terms = [t for t in query.lower().split() if len(t) > 1]
        if not terms:
            return []
        scored: list[tuple[Entry, float]] = []
        for entry in self._entries.values():
            if topic is not None and entry.topic != topic:
                continue
            haystack = entry.content.lower()
            hits = sum(haystack.count(term) for term in terms)
            if hits:
                covered = sum(1 for term in terms if term in haystack) / len(terms)
                scored.append((entry, covered * (1.0 - 1.0 / (1.0 + hits))))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def archive_entries(self, entry_ids: Sequence[str], *, superseded_by: str | None = None) -> int:
        touched = 0
        for entry_id in entry_ids:
            entry = self._entries.get(entry_id)
            if entry is None or entry.archived:
                continue
            entry.archived = True
            entry.superseded_by = superseded_by
            topic = self._topics.get(entry.topic)
            if topic:
                topic.live_tokens = max(0, topic.live_tokens - entry.tokens)
            touched += 1
        return touched

    def set_salience(self, entry_id: str, salience: float) -> None:
        entry = self._entries.get(entry_id)
        if entry is not None:
            entry.salience = min(1.0, max(0.0, salience))

    # embeddings
    def set_embedding(self, entry_id: str, vector: Sequence[float]) -> None:
        self._vectors[entry_id] = list(vector)

    def get_embeddings(self, entry_ids: Sequence[str]) -> dict[str, list[float]]:
        return {eid: self._vectors[eid] for eid in entry_ids if eid in self._vectors}

    def entries_missing_embeddings(
        self, *, topic: str | None = None, limit: int = 500
    ) -> list[Entry]:
        rows = [
            e
            for e in self._entries.values()
            if e.id not in self._vectors and (topic is None or e.topic == topic)
        ]
        rows.sort(key=lambda e: e.created_at)
        return rows[:limit]

    # runs
    def save_run(self, run: Run) -> Run:
        self._runs[run.id] = run
        return run

    def get_run(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self, *, topic: str | None = None, limit: int = 50) -> list[Run]:
        rows = [r for r in self._runs.values() if topic is None or r.topic == topic]
        rows.sort(key=lambda r: r.started_at, reverse=True)
        return rows[:limit]

    def stats(self, *, topic: str | None = None) -> dict[str, Any]:
        entries = [e for e in self._entries.values() if topic is None or e.topic == topic]
        runs = [r for r in self._runs.values() if topic is None or r.topic == topic]
        by_kind: dict[str, int] = {}
        by_agent: dict[str, int] = {}
        for entry in entries:
            by_kind[entry.kind.value] = by_kind.get(entry.kind.value, 0) + 1
            by_agent[entry.agent] = by_agent.get(entry.agent, 0) + 1
        return {
            "topics": len(self._topics) if topic is None else 1,
            "entries": len(entries),
            "archived": sum(1 for e in entries if e.archived),
            "tokens": sum(e.tokens for e in entries),
            "runs": len(runs),
            "by_kind": by_kind,
            "by_agent": by_agent,
            "usage": {
                "prompt_tokens": sum(r.usage.prompt_tokens for r in runs),
                "completion_tokens": sum(r.usage.completion_tokens for r in runs),
                "calls": sum(r.usage.calls for r in runs),
                "cost_usd": round(sum(r.usage.cost_usd for r in runs), 6),
            },
        }
