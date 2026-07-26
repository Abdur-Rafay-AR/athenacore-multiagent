"""SQLite-backed memory store.

Why SQLite rather than a vector database: a topic's memory is small (thousands of
entries), and SQLite gives durable transactions, real full-text search via FTS5,
and a single-file artifact an operator can copy, diff or inspect with any
tool - with zero services to run. Embeddings live in the same file as float32
blobs and are scored in Python, which stays comfortably fast well past the point
where a hobby project needs anything else.

Notable choices:

* **WAL mode + busy timeout** so the API server, CLI and UI can share one file.
* **FTS5 external-content index** kept in sync by triggers, so text search never
  drifts from the entries table. If the interpreter was built without FTS5 the
  store degrades to ``LIKE`` matching instead of failing.
* **Schema migrations** keyed off ``PRAGMA user_version``, applied on connect.
* **Denormalised counters** on ``topics`` (``entry_count``, ``live_tokens``) so
  the compaction trigger and topic list are single-row reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from crucible.errors import MemoryError_
from crucible.logging_setup import get_logger
from crucible.memory.models import (
    Entry,
    EntryKind,
    Run,
    Topic,
    Usage,
    from_iso,
    to_iso,
    utcnow,
)
from crucible.memory.store import EntryFilter, MemoryStore

log = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS topics (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    entry_count  INTEGER NOT NULL DEFAULT 0,
    live_tokens  INTEGER NOT NULL DEFAULT 0,
    tags         TEXT NOT NULL DEFAULT '',
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entries (
    id            TEXT PRIMARY KEY,
    topic         TEXT NOT NULL REFERENCES topics(name) ON DELETE CASCADE,
    agent         TEXT NOT NULL,
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    run_id        TEXT,
    parent_id     TEXT,
    created_at    TEXT NOT NULL,
    salience      REAL NOT NULL DEFAULT 0.5,
    archived      INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT,
    tokens        INTEGER NOT NULL DEFAULT 0,
    model         TEXT,
    latency_ms    INTEGER,
    metadata      TEXT NOT NULL DEFAULT '{}',
    tags          TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_entries_topic_time ON entries(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entries_kind       ON entries(topic, kind);
CREATE INDEX IF NOT EXISTS idx_entries_run        ON entries(run_id);
CREATE INDEX IF NOT EXISTS idx_entries_live       ON entries(topic, archived, created_at DESC);

CREATE TABLE IF NOT EXISTS embeddings (
    entry_id TEXT PRIMARY KEY REFERENCES entries(id) ON DELETE CASCADE,
    dims     INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    preset      TEXT NOT NULL DEFAULT '',
    query       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT '',
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    usage       TEXT NOT NULL DEFAULT '{}',
    node_states TEXT NOT NULL DEFAULT '{}',
    error       TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_topic_time ON runs(topic, started_at DESC);
"""

_FTS_V1 = """
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    content,
    agent UNINDEXED,
    topic UNINDEXED,
    content='entries',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
    INSERT INTO entries_fts(rowid, content, agent, topic)
    VALUES (new.rowid, new.content, new.agent, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, content, agent, topic)
    VALUES ('delete', old.rowid, old.content, old.agent, old.topic);
END;

CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE OF content ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, content, agent, topic)
    VALUES ('delete', old.rowid, old.content, old.agent, old.topic);
    INSERT INTO entries_fts(rowid, content, agent, topic)
    VALUES (new.rowid, new.content, new.agent, new.topic);
END;
"""

_FTS_SPECIALS = str.maketrans(dict.fromkeys('"()*:^-+', " "))


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    User input goes through this, never into MATCH directly: FTS5 treats a pile
    of punctuation as syntax and raises on malformed expressions. Each surviving
    term is quoted and OR-ed, with a prefix wildcard on the last term so partial
    words still match while typing.
    """
    terms = [t for t in text.translate(_FTS_SPECIALS).split() if t]
    if not terms:
        return ""
    quoted = [f'"{t}"' for t in terms[:-1]]
    quoted.append(f'"{terms[-1]}"*')
    return " OR ".join(quoted)


class SqliteMemoryStore(MemoryStore):
    """Durable store over a single SQLite file.

    Thread-safe: one connection is shared with ``check_same_thread=False`` and all
    access is serialised by a re-entrant lock. That is the right trade for this
    workload, where contention is a handful of agents rather than thousands of
    requests, and it keeps transaction semantics obvious.

    Pass ``":memory:"`` for an ephemeral store (tests use this).
    """

    def __init__(self, path: str | Path = "data/crucible.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self.fts_enabled = False
        self._configure()
        self._migrate()

    # -- plumbing ------------------------------------------------------------

    def _configure(self) -> None:
        cur = self._conn.cursor()
        # WAL lets readers (UI, API) proceed while a run writes. Not available for
        # in-memory databases, where the pragma is simply ignored.
        if self.path != ":memory:":
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=10000")
        cur.close()

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            version = cur.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise MemoryError_(
                    f"database at {self.path} uses schema v{version}, "
                    f"but this build understands v{SCHEMA_VERSION}",
                    hint="Upgrade crucible, or point CRUCIBLE_DATABASE_PATH at a new file.",
                )
            if version < 1:
                cur.executescript(_SCHEMA_V1)
                cur.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                log.info("initialised memory schema", extra={"path": self.path, "version": 1})
            self._conn.commit()
            self._install_fts(cur)
            cur.close()

    def _install_fts(self, cur: sqlite3.Cursor) -> None:
        try:
            cur.executescript(_FTS_V1)
            self._conn.commit()
            self.fts_enabled = True
        except sqlite3.OperationalError as exc:
            # Interpreters built without the FTS5 extension: degrade, don't die.
            log.warning("FTS5 unavailable, falling back to LIKE search", extra={"error": str(exc)})
            self._conn.rollback()
            self.fts_enabled = False

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        """Run a block in a transaction, committing on success."""
        if self._closed:
            raise MemoryError_("store is closed")
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Cursor]:
        if self._closed:
            raise MemoryError_("store is closed")
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.commit()
                self._conn.close()
                self._closed = True

    # -- topics --------------------------------------------------------------

    def ensure_topic(self, name: str, *, description: str = "", tags: Sequence[str] = ()) -> Topic:
        name = name.strip()
        if not name:
            raise MemoryError_("topic name cannot be empty")
        existing = self.get_topic(name)
        if existing is not None:
            if (description and not existing.description) or tags:
                merged = tuple(dict.fromkeys((*existing.tags, *tags)))
                with self._tx() as cur:
                    cur.execute(
                        "UPDATE topics SET description=?, tags=?, updated_at=? WHERE name=?",
                        (
                            existing.description or description,
                            ",".join(merged),
                            to_iso(utcnow()),
                            name,
                        ),
                    )
                return self.require_topic(name)
            return existing

        topic = Topic(name=name, description=description, tags=tuple(tags))
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO topics
                   (id, name, description, created_at, updated_at, entry_count, live_tokens, tags, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(name) DO NOTHING""",
                (
                    topic.id,
                    topic.name,
                    topic.description,
                    to_iso(topic.created_at),
                    to_iso(topic.updated_at),
                    0,
                    0,
                    ",".join(topic.tags),
                    json.dumps(topic.metadata),
                ),
            )
        return self.require_topic(name)

    def get_topic(self, name: str) -> Topic | None:
        with self._read() as cur:
            row = cur.execute("SELECT * FROM topics WHERE name=?", (name,)).fetchone()
        return _row_to_topic(row) if row else None

    def list_topics(self, *, limit: int = 100, search: str | None = None) -> list[Topic]:
        sql = "SELECT * FROM topics"
        params: list[Any] = []
        if search:
            sql += " WHERE name LIKE ? OR description LIKE ?"
            like = f"%{search}%"
            params += [like, like]
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._read() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [_row_to_topic(r) for r in rows]

    def delete_topic(self, name: str) -> int:
        with self._tx() as cur:
            count = cur.execute("SELECT COUNT(*) FROM entries WHERE topic=?", (name,)).fetchone()[0]
            cur.execute("DELETE FROM entries WHERE topic=?", (name,))
            cur.execute("DELETE FROM runs WHERE topic=?", (name,))
            cur.execute("DELETE FROM topics WHERE name=?", (name,))
        return int(count)

    def rename_topic(self, old: str, new: str) -> Topic:
        """Rename a topic, carrying its entries and runs with it."""
        new = new.strip()
        if not new:
            raise MemoryError_("new topic name cannot be empty")
        self.require_topic(old)
        if self.get_topic(new) is not None:
            raise MemoryError_(f"topic {new!r} already exists")
        with self._tx() as cur:
            # entries.topic references topics(name), so renaming the parent
            # orphans the children mid-statement. Deferring the check to COMMIT
            # lets both sides move within one transaction.
            cur.execute("PRAGMA defer_foreign_keys = ON")
            cur.execute(
                "UPDATE topics SET name=?, updated_at=? WHERE name=?", (new, to_iso(utcnow()), old)
            )
            cur.execute("UPDATE entries SET topic=? WHERE topic=?", (new, old))
            cur.execute("UPDATE runs SET topic=? WHERE topic=?", (new, old))
        return self.require_topic(new)

    # -- entries -------------------------------------------------------------

    def add_entry(self, entry: Entry) -> Entry:
        self.ensure_topic(entry.topic)
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO entries
                   (id, topic, agent, kind, content, run_id, parent_id, created_at,
                    salience, archived, superseded_by, tokens, model, latency_ms, metadata, tags)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.id,
                    entry.topic,
                    entry.agent,
                    entry.kind.value,
                    entry.content,
                    entry.run_id,
                    entry.parent_id,
                    to_iso(entry.created_at),
                    entry.salience,
                    int(entry.archived),
                    entry.superseded_by,
                    entry.tokens,
                    entry.model,
                    entry.latency_ms,
                    json.dumps(entry.metadata),
                    ",".join(entry.tags),
                ),
            )
            cur.execute(
                """UPDATE topics
                      SET entry_count = entry_count + 1,
                          live_tokens = live_tokens + ?,
                          updated_at  = ?
                    WHERE name = ?""",
                (0 if entry.archived else entry.tokens, to_iso(entry.created_at), entry.topic),
            )
        return entry

    def get_entry(self, entry_id: str) -> Entry | None:
        with self._read() as cur:
            row = cur.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return _row_to_entry(row) if row else None

    def query_entries(self, spec: EntryFilter) -> list[Entry]:
        clauses: list[str] = []
        params: list[Any] = []
        if spec.topic is not None:
            clauses.append("topic = ?")
            params.append(spec.topic)
        if spec.kinds:
            marks = ",".join("?" * len(spec.kinds))
            clauses.append(f"kind IN ({marks})")
            params += [k.value if isinstance(k, EntryKind) else str(k) for k in spec.kinds]
        if spec.agents:
            marks = ",".join("?" * len(spec.agents))
            clauses.append(f"agent IN ({marks})")
            params += list(spec.agents)
        if spec.run_id is not None:
            clauses.append("run_id = ?")
            params.append(spec.run_id)
        if not spec.include_archived:
            clauses.append("archived = 0")
        if spec.since is not None:
            clauses.append("created_at >= ?")
            params.append(to_iso(spec.since))

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if spec.newest_first else "ASC"
        sql = f"SELECT * FROM entries{where} ORDER BY created_at {order}, rowid {order} LIMIT ? OFFSET ?"
        params += [spec.limit, spec.offset]
        with self._read() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [_row_to_entry(r) for r in rows]

    def keyword_search(
        self, query: str, *, topic: str | None = None, limit: int = 50
    ) -> list[tuple[Entry, float]]:
        query = query.strip()
        if not query:
            return []
        if self.fts_enabled:
            match = _fts_query(query)
            if not match:
                return []
            sql = """
                SELECT e.*, bm25(entries_fts) AS rank
                  FROM entries_fts
                  JOIN entries e ON e.rowid = entries_fts.rowid
                 WHERE entries_fts MATCH ?
            """
            params: list[Any] = [match]
            if topic is not None:
                sql += " AND e.topic = ?"
                params.append(topic)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            try:
                with self._read() as cur:
                    rows = cur.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:  # malformed MATCH slipped through
                log.warning("FTS query failed, using LIKE", extra={"error": str(exc)})
                return self._like_search(query, topic=topic, limit=limit)
            if not rows:
                return []
            # bm25() is negative with better matches more negative; flip and scale
            # to [0, 1] against the best hit so weights stay comparable.
            relevances = [-float(r["rank"]) for r in rows]
            best = max(relevances) or 1.0
            return [
                (_row_to_entry(r), max(0.0, rel / best))
                for r, rel in zip(rows, relevances, strict=True)
            ]
        return self._like_search(query, topic=topic, limit=limit)

    def _like_search(
        self, query: str, *, topic: str | None, limit: int
    ) -> list[tuple[Entry, float]]:
        terms = [t for t in query.lower().split() if len(t) > 1][:6]
        if not terms:
            return []
        clauses = " OR ".join(["LOWER(content) LIKE ?"] * len(terms))
        params: list[Any] = [f"%{t}%" for t in terms]
        sql = f"SELECT * FROM entries WHERE ({clauses})"
        if topic is not None:
            sql += " AND topic = ?"
            params.append(topic)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit * 3)
        with self._read() as cur:
            rows = cur.execute(sql, params).fetchall()
        scored: list[tuple[Entry, float]] = []
        for row in rows:
            entry = _row_to_entry(row)
            haystack = entry.content.lower()
            covered = sum(1 for t in terms if t in haystack) / len(terms)
            scored.append((entry, covered))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def archive_entries(self, entry_ids: Sequence[str], *, superseded_by: str | None = None) -> int:
        if not entry_ids:
            return 0
        marks = ",".join("?" * len(entry_ids))
        with self._tx() as cur:
            rows = cur.execute(
                f"SELECT topic, tokens FROM entries WHERE id IN ({marks}) AND archived = 0",
                list(entry_ids),
            ).fetchall()
            if not rows:
                return 0
            reclaimed: dict[str, int] = {}
            for row in rows:
                reclaimed[row["topic"]] = reclaimed.get(row["topic"], 0) + int(row["tokens"])
            cur.execute(
                f"UPDATE entries SET archived = 1, superseded_by = ? WHERE id IN ({marks}) AND archived = 0",
                [superseded_by, *entry_ids],
            )
            touched = cur.rowcount
            for topic, tokens in reclaimed.items():
                cur.execute(
                    "UPDATE topics SET live_tokens = MAX(0, live_tokens - ?) WHERE name = ?",
                    (tokens, topic),
                )
        return int(touched)

    def set_salience(self, entry_id: str, salience: float) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE entries SET salience=? WHERE id=?",
                (min(1.0, max(0.0, float(salience))), entry_id),
            )

    def add_tags(self, entry_id: str, tags: Sequence[str]) -> None:
        entry = self.get_entry(entry_id)
        if entry is None:
            return
        merged = tuple(dict.fromkeys((*entry.tags, *tags)))
        with self._tx() as cur:
            cur.execute("UPDATE entries SET tags=? WHERE id=?", (",".join(merged), entry_id))

    # -- embeddings ----------------------------------------------------------

    def set_embedding(self, entry_id: str, vector: Sequence[float]) -> None:
        blob = array("f", [float(v) for v in vector]).tobytes()
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO embeddings (entry_id, dims, vector) VALUES (?,?,?) "
                "ON CONFLICT(entry_id) DO UPDATE SET dims=excluded.dims, vector=excluded.vector",
                (entry_id, len(vector), sqlite3.Binary(blob)),
            )

    def get_embeddings(self, entry_ids: Sequence[str]) -> dict[str, list[float]]:
        if not entry_ids:
            return {}
        out: dict[str, list[float]] = {}
        # Chunked to stay clear of SQLite's variable limit on large recalls.
        with self._read() as cur:
            for start in range(0, len(entry_ids), 400):
                chunk = list(entry_ids[start : start + 400])
                marks = ",".join("?" * len(chunk))
                rows = cur.execute(
                    f"SELECT entry_id, vector FROM embeddings WHERE entry_id IN ({marks})", chunk
                ).fetchall()
                for row in rows:
                    vec = array("f")
                    vec.frombytes(row["vector"])
                    out[row["entry_id"]] = list(vec)
        return out

    def entries_missing_embeddings(
        self, *, topic: str | None = None, limit: int = 500
    ) -> list[Entry]:
        sql = """
            SELECT e.* FROM entries e
             LEFT JOIN embeddings m ON m.entry_id = e.id
             WHERE m.entry_id IS NULL
        """
        params: list[Any] = []
        if topic is not None:
            sql += " AND e.topic = ?"
            params.append(topic)
        sql += " ORDER BY e.created_at ASC LIMIT ?"
        params.append(limit)
        with self._read() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [_row_to_entry(r) for r in rows]

    # -- runs ----------------------------------------------------------------

    def save_run(self, run: Run) -> Run:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO runs
                   (id, topic, preset, query, status, model, started_at, ended_at,
                    usage, node_states, error, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       status=excluded.status,
                       ended_at=excluded.ended_at,
                       usage=excluded.usage,
                       node_states=excluded.node_states,
                       error=excluded.error,
                       metadata=excluded.metadata""",
                (
                    run.id,
                    run.topic,
                    run.preset,
                    run.query,
                    run.status.value,
                    run.model,
                    to_iso(run.started_at),
                    to_iso(run.ended_at) if run.ended_at else None,
                    json.dumps(run.usage.to_dict()),
                    json.dumps(run.node_states),
                    run.error,
                    json.dumps(run.metadata),
                ),
            )
        return run

    def get_run(self, run_id: str) -> Run | None:
        with self._read() as cur:
            row = cur.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None

    def list_runs(self, *, topic: str | None = None, limit: int = 50) -> list[Run]:
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        if topic is not None:
            sql += " WHERE topic=?"
            params.append(topic)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._read() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [_row_to_run(r) for r in rows]

    # -- aggregate -----------------------------------------------------------

    def stats(self, *, topic: str | None = None) -> dict[str, Any]:
        scope = " WHERE topic = ?" if topic else ""
        params: list[Any] = [topic] if topic else []
        with self._read() as cur:
            entries = cur.execute(
                f"""SELECT COUNT(*) AS n,
                           COALESCE(SUM(tokens), 0) AS tokens,
                           COALESCE(SUM(archived), 0) AS archived
                      FROM entries{scope}""",
                params,
            ).fetchone()
            by_kind = {
                r["kind"]: r["n"]
                for r in cur.execute(
                    f"SELECT kind, COUNT(*) AS n FROM entries{scope} GROUP BY kind ORDER BY n DESC",
                    params,
                ).fetchall()
            }
            by_agent = {
                r["agent"]: r["n"]
                for r in cur.execute(
                    f"SELECT agent, COUNT(*) AS n FROM entries{scope} GROUP BY agent ORDER BY n DESC",
                    params,
                ).fetchall()
            }
            topics = cur.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            run_rows = cur.execute(f"SELECT usage FROM runs{scope}", params).fetchall()

        usage = Usage()
        for row in run_rows:
            usage = usage + Usage.from_dict(json.loads(row["usage"] or "{}"))
        return {
            "topics": 1 if topic else int(topics),
            "entries": int(entries["n"]),
            "archived": int(entries["archived"]),
            "tokens": int(entries["tokens"]),
            "runs": len(run_rows),
            "by_kind": by_kind,
            "by_agent": by_agent,
            "usage": usage.to_dict(),
            "fts": self.fts_enabled,
            "path": self.path,
        }

    def vacuum(self) -> None:
        """Reclaim space and rebuild the text index. Safe to run periodically."""
        with self._lock:
            if self.fts_enabled:
                self._conn.execute("INSERT INTO entries_fts(entries_fts) VALUES ('rebuild')")
            self._conn.commit()
            self._conn.execute("VACUUM")

    def activity_by_day(self, *, topic: str | None = None, days: int = 30) -> list[tuple[str, int]]:
        """``(YYYY-MM-DD, entries)`` pairs for the analytics view, oldest first."""
        sql = "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n FROM entries"
        params: list[Any] = []
        if topic is not None:
            sql += " WHERE topic = ?"
            params.append(topic)
        sql += " GROUP BY day ORDER BY day DESC LIMIT ?"
        params.append(days)
        with self._read() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [(r["day"], int(r["n"])) for r in reversed(rows)]


# -- row mapping -------------------------------------------------------------


def _split_tags(raw: str | None) -> tuple[str, ...]:
    return tuple(t for t in (raw or "").split(",") if t)


def _row_to_topic(row: sqlite3.Row) -> Topic:
    return Topic(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=from_iso(row["created_at"]),
        updated_at=from_iso(row["updated_at"]),
        entry_count=int(row["entry_count"]),
        live_tokens=int(row["live_tokens"]),
        tags=_split_tags(row["tags"]),
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _row_to_entry(row: sqlite3.Row) -> Entry:
    return Entry(
        id=row["id"],
        topic=row["topic"],
        agent=row["agent"],
        kind=EntryKind(row["kind"]),
        content=row["content"],
        run_id=row["run_id"],
        parent_id=row["parent_id"],
        created_at=from_iso(row["created_at"]),
        salience=float(row["salience"]),
        archived=bool(row["archived"]),
        superseded_by=row["superseded_by"],
        tokens=int(row["tokens"]),
        model=row["model"],
        latency_ms=row["latency_ms"],
        metadata=json.loads(row["metadata"] or "{}"),
        tags=_split_tags(row["tags"]),
    )


def _row_to_run(row: sqlite3.Row) -> Run:
    data: dict[str, Any] = {
        "id": row["id"],
        "topic": row["topic"],
        "preset": row["preset"],
        "query": row["query"],
        "status": row["status"],
        "model": row["model"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "usage": json.loads(row["usage"] or "{}"),
        "node_states": json.loads(row["node_states"] or "{}"),
        "error": row["error"],
        "metadata": json.loads(row["metadata"] or "{}"),
    }
    return Run.from_dict(data)


def isoformat_day(value: datetime) -> str:  # pragma: no cover - helper for callers
    return value.strftime("%Y-%m-%d")
