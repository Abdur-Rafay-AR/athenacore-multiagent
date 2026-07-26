"""Memory compaction.

An append-only log grows without bound, and eventually no useful subset of it
fits in a context window. Compaction is the answer: fold the oldest run of
entries into one durable summary, archive the originals (never delete them), and
let recall work against a much denser log.

The policy is intentionally conservative:

* Only triggers above ``compaction_threshold_tokens`` of *live* tokens.
* Never touches the newest ``compaction_keep_recent`` entries - recent context is
  where most value is.
* Never folds away protected kinds (``summary``, ``decision``, ``note``): human
  notes and prior conclusions survive verbatim, forever.
* Originals are archived with ``superseded_by`` pointing at the summary, so the
  full history is still auditable and one SQL update can undo a compaction.

The summariser is injected as a plain callable, which keeps this module free of
any dependency on the LLM layer and makes it trivially testable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from athenacore.logging_setup import get_logger
from athenacore.memory.models import Entry, EntryKind, estimate_tokens
from athenacore.memory.store import EntryFilter, MemoryStore

log = get_logger(__name__)

Summarizer = Callable[[str], str]
"""Takes a block of prior entries, returns condensed prose."""

COMPACTION_AGENT = "compactor"

_PROMPT = """You are compacting the long-term memory of a multi-agent research \
system so that it still fits in a context window.

Condense the entries below into a dense factual briefing. Rules:
- Preserve every claim, number, name, date and open question. Losing a fact is \
the only unacceptable error.
- Preserve disagreement explicitly: say who argued what, do not average views \
into a bland consensus.
- Drop pleasantries, restatements and meta-commentary about the process.
- Write plain prose or tight bullets. No preamble, no "here is the summary".
- Aim for roughly {target} words.

ENTRIES TO COMPACT
------------------
{body}
"""


@dataclass(slots=True)
class CompactionResult:
    """What a compaction pass did. ``performed=False`` means the policy declined."""

    performed: bool
    reason: str
    summary_entry: Entry | None = None
    archived_ids: tuple[str, ...] = ()
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def ratio(self) -> float:
        """Compression achieved, ``1.0`` meaning no change."""
        if not self.tokens_before:
            return 1.0
        return self.tokens_after / self.tokens_before

    def to_dict(self) -> dict[str, object]:
        return {
            "performed": self.performed,
            "reason": self.reason,
            "summary_entry_id": self.summary_entry.id if self.summary_entry else None,
            "archived": list(self.archived_ids),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "ratio": round(self.ratio, 3),
        }


class MemoryCompactor:
    """Applies the compaction policy to a topic."""

    def __init__(
        self,
        store: MemoryStore,
        summarizer: Summarizer,
        *,
        threshold_tokens: int = 12_000,
        keep_recent: int = 8,
        target_ratio: float = 0.25,
        min_entries_to_fold: int = 4,
    ) -> None:
        self.store = store
        self.summarizer = summarizer
        self.threshold_tokens = threshold_tokens
        self.keep_recent = max(1, keep_recent)
        self.target_ratio = target_ratio
        self.min_entries_to_fold = max(2, min_entries_to_fold)

    def should_compact(self, topic: str) -> bool:
        record = self.store.get_topic(topic)
        return bool(record and record.live_tokens > self.threshold_tokens)

    def maybe_compact(self, topic: str, *, force: bool = False) -> CompactionResult:
        """Compact if the policy says so. Safe and cheap to call after every run."""
        record = self.store.get_topic(topic)
        if record is None:
            return CompactionResult(False, "topic does not exist")
        if not force and record.live_tokens <= self.threshold_tokens:
            return CompactionResult(
                False,
                f"under threshold ({record.live_tokens} <= {self.threshold_tokens} tokens)",
                tokens_before=record.live_tokens,
                tokens_after=record.live_tokens,
            )
        return self.compact(topic)

    def compact(self, topic: str) -> CompactionResult:
        live = self.store.query_entries(
            EntryFilter(topic=topic, include_archived=False, limit=10_000, newest_first=False)
        )
        tokens_before = sum(e.tokens for e in live)
        foldable = self._select_foldable(live)

        if len(foldable) < self.min_entries_to_fold:
            return CompactionResult(
                False,
                f"only {len(foldable)} foldable entries, need {self.min_entries_to_fold}",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        body = "\n\n".join(e.as_prompt_block() for e in foldable)
        folded_tokens = sum(e.tokens for e in foldable)
        target_words = max(120, int(folded_tokens * self.target_ratio * 0.75))
        summary_text = self.summarizer(_PROMPT.format(target=target_words, body=body)).strip()

        if not summary_text:
            return CompactionResult(
                False,
                "summariser returned nothing; memory left untouched",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )
        if estimate_tokens(summary_text) >= folded_tokens:
            # A "summary" longer than its input would make memory worse. Refuse.
            return CompactionResult(
                False,
                "summary was not smaller than the source entries",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        window = f"{foldable[0].created_at.date()} → {foldable[-1].created_at.date()}"
        summary = Entry(
            topic=topic,
            agent=COMPACTION_AGENT,
            kind=EntryKind.SUMMARY,
            content=summary_text,
            salience=0.85,  # condensed history should outrank ordinary chatter
            tags=("compaction",),
            metadata={
                "compacted_entry_ids": [e.id for e in foldable],
                "compacted_count": len(foldable),
                "source_tokens": folded_tokens,
                "window": window,
                "agents": sorted({e.agent for e in foldable}),
            },
        )
        self.store.add_entry(summary)
        archived = self.store.archive_entries([e.id for e in foldable], superseded_by=summary.id)

        record = self.store.get_topic(topic)
        tokens_after = record.live_tokens if record else tokens_before
        log.info(
            "compacted topic memory",
            extra={
                "topic": topic,
                "folded": archived,
                "before": tokens_before,
                "after": tokens_after,
            },
        )
        return CompactionResult(
            True,
            f"folded {archived} entries into one summary",
            summary_entry=summary,
            archived_ids=tuple(e.id for e in foldable),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def _select_foldable(self, live: Sequence[Entry]) -> list[Entry]:
        """Oldest-first entries eligible for folding, respecting the policy."""
        if len(live) <= self.keep_recent:
            return []
        older = list(live[: len(live) - self.keep_recent])
        return [e for e in older if not e.kind.protected]

    def undo(self, summary_entry_id: str) -> int:
        """Restore entries folded into a summary. Returns how many came back.

        Compaction is lossy by design, so being able to reverse it is what makes
        it safe to run automatically.
        """
        summary = self.store.get_entry(summary_entry_id)
        if summary is None or summary.kind is not EntryKind.SUMMARY:
            return 0
        ids: list[str] = list(summary.metadata.get("compacted_entry_ids") or [])
        restored = 0
        for entry_id in ids:
            entry = self.store.get_entry(entry_id)
            if entry is not None and entry.archived and entry.superseded_by == summary_entry_id:
                self.store.set_salience(entry_id, entry.salience)
                restored += self._unarchive(entry_id)
        self.store.archive_entries([summary_entry_id])
        return restored

    def _unarchive(self, entry_id: str) -> int:
        """Un-archiving is not part of the abstract store contract, so this uses
        the SQL path when available and degrades to a no-op otherwise."""
        conn = getattr(self.store, "_conn", None)
        if conn is None:
            entry = self.store.get_entry(entry_id)
            if entry is not None:
                entry.archived = False
                entry.superseded_by = None
                return 1
            return 0
        lock = getattr(self.store, "_lock", None)
        ctx = lock if lock is not None else _NullContext()
        with ctx:
            cur = conn.cursor()
            cur.execute("SELECT topic, tokens FROM entries WHERE id=? AND archived=1", (entry_id,))
            row = cur.fetchone()
            if row is None:
                cur.close()
                return 0
            cur.execute("UPDATE entries SET archived=0, superseded_by=NULL WHERE id=?", (entry_id,))
            cur.execute(
                "UPDATE topics SET live_tokens = live_tokens + ? WHERE name = ?",
                (int(row["tokens"]), row["topic"]),
            )
            conn.commit()
            cur.close()
        return 1


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
