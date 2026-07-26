"""Memory: the store contract, retrieval ranking and compaction policy."""

from __future__ import annotations

from datetime import timedelta

import pytest

from crucible.config import RetrievalSettings
from crucible.errors import TopicNotFound
from crucible.memory.compaction import MemoryCompactor
from crucible.memory.embeddings import cosine
from crucible.memory.models import Entry, EntryKind, Usage, estimate_tokens, utcnow
from crucible.memory.retrieval import MemoryRetriever
from crucible.memory.store import EntryFilter

# -- store conformance (runs against every implementation) -------------------


class TestStoreConformance:
    """Behaviour both stores must share. Parametrised via the ``any_store`` fixture."""

    def test_ensure_topic_is_idempotent(self, any_store):
        first = any_store.ensure_topic("t", description="desc")
        second = any_store.ensure_topic("t")
        assert first.name == second.name == "t"
        assert len(any_store.list_topics()) == 1

    def test_add_entry_creates_topic_and_updates_counters(self, any_store):
        any_store.add_entry(Entry(topic="new", agent="a", content="hello world"))
        topic = any_store.get_topic("new")
        assert topic is not None
        assert topic.entry_count == 1
        assert topic.live_tokens > 0

    def test_query_filters(self, any_store):
        any_store.add_entry(Entry(topic="t", agent="a", content="alpha", kind=EntryKind.RESEARCH))
        any_store.add_entry(Entry(topic="t", agent="b", content="beta", kind=EntryKind.CRITIQUE))

        by_kind = any_store.query_entries(EntryFilter(topic="t", kinds=[EntryKind.CRITIQUE]))
        assert [e.agent for e in by_kind] == ["b"]

        by_agent = any_store.query_entries(EntryFilter(topic="t", agents=["a"]))
        assert [e.content for e in by_agent] == ["alpha"]

    def test_archiving_hides_entries_and_reclaims_tokens(self, any_store):
        entry = any_store.add_entry(Entry(topic="t", agent="a", content="x " * 100))
        before = any_store.get_topic("t").live_tokens
        assert any_store.archive_entries([entry.id]) == 1

        assert any_store.query_entries(EntryFilter(topic="t")) == []
        assert len(any_store.query_entries(EntryFilter(topic="t", include_archived=True))) == 1
        assert any_store.get_topic("t").live_tokens < before

    def test_archiving_is_not_double_counted(self, any_store):
        entry = any_store.add_entry(Entry(topic="t", agent="a", content="x " * 50))
        any_store.archive_entries([entry.id])
        assert any_store.archive_entries([entry.id]) == 0
        assert any_store.get_topic("t").live_tokens == 0

    def test_keyword_search_ranks_matches_first(self, any_store):
        any_store.add_entry(Entry(topic="t", agent="a", content="lithium brine water extraction"))
        any_store.add_entry(
            Entry(topic="t", agent="b", content="unrelated commentary about bicycles")
        )
        hits = any_store.keyword_search("lithium water", topic="t")
        assert hits, "expected at least one match"
        assert "lithium" in hits[0][0].content

    def test_timeline_is_chronological(self, any_store):
        for i in range(3):
            any_store.add_entry(Entry(topic="t", agent="a", content=f"entry {i}"))
        contents = [e.content for e in any_store.timeline("t")]
        assert contents == ["entry 0", "entry 1", "entry 2"]

    def test_delete_topic_removes_entries(self, any_store):
        any_store.add_entry(Entry(topic="t", agent="a", content="x"))
        assert any_store.delete_topic("t") == 1
        assert any_store.get_topic("t") is None

    def test_require_topic_raises_for_unknown(self, any_store):
        with pytest.raises(TopicNotFound):
            any_store.require_topic("nope")

    def test_embeddings_round_trip(self, any_store):
        entry = any_store.add_entry(Entry(topic="t", agent="a", content="vector me"))
        # Compared by id: timestamps persist at millisecond precision, so a
        # round-tripped entry is not byte-identical to the in-memory one.
        assert [e.id for e in any_store.entries_missing_embeddings()] == [entry.id]
        any_store.set_embedding(entry.id, [0.5, -0.25, 0.125])
        stored = any_store.get_embeddings([entry.id])[entry.id]
        assert stored == pytest.approx([0.5, -0.25, 0.125], abs=1e-6)
        assert any_store.entries_missing_embeddings() == []


# -- sqlite specifics --------------------------------------------------------


class TestSqliteStore:
    def test_persists_across_connections(self, tmp_path):
        from crucible.memory.sqlite_store import SqliteMemoryStore

        path = tmp_path / "persist.sqlite3"
        first = SqliteMemoryStore(path)
        first.add_entry(Entry(topic="t", agent="a", content="durable"))
        first.close()

        second = SqliteMemoryStore(path)
        try:
            assert [e.content for e in second.timeline("t")] == ["durable"]
        finally:
            second.close()

    def test_fts_is_available(self, store):
        # The suite would still pass without FTS5 via the LIKE fallback, but a
        # silent downgrade is worth knowing about.
        assert store.fts_enabled, "FTS5 unavailable; search quality is degraded"

    def test_search_survives_punctuation(self, store):
        """User input goes straight into FTS5 MATCH, which raises on bad syntax."""
        store.add_entry(Entry(topic="t", agent="a", content="cost is 40% (high) per unit"))
        for query in ["40% (high)", '"unbalanced', "a AND OR b", "***", "NEAR("]:
            store.keyword_search(query, topic="t")  # must not raise

    def test_rename_topic_moves_entries(self, store):
        store.add_entry(Entry(topic="old", agent="a", content="x"))
        store.rename_topic("old", "new")
        assert store.get_topic("old") is None
        assert len(store.timeline("new")) == 1

    def test_stats_and_activity(self, seeded):
        stats = seeded.stats(topic="lithium")
        assert stats["entries"] == 5
        assert stats["by_agent"]["research"] == 2
        assert seeded.activity_by_day(topic="lithium")


# -- models ------------------------------------------------------------------


class TestModels:
    def test_entry_round_trips(self):
        entry = Entry(
            topic="t",
            agent="a",
            content="content",
            kind=EntryKind.INSIGHT,
            tags=("x", "y"),
            metadata={"k": 1},
        )
        restored = Entry.from_dict(entry.to_dict())
        assert restored.id == entry.id
        assert restored.kind is EntryKind.INSIGHT
        assert restored.tags == ("x", "y")
        assert restored.metadata == {"k": 1}

    def test_salience_is_clamped(self):
        assert Entry(topic="t", agent="a", content="c", salience=5.0).salience == 1.0
        assert Entry(topic="t", agent="a", content="c", salience=-1.0).salience == 0.0

    def test_protected_kinds(self):
        assert EntryKind.SUMMARY.protected
        assert EntryKind.DECISION.protected
        assert not EntryKind.RESEARCH.protected

    def test_usage_adds(self):
        total = Usage(prompt_tokens=10, completion_tokens=5, calls=1) + Usage(
            prompt_tokens=3, completion_tokens=2, calls=1
        )
        assert total.total_tokens == 20
        assert total.calls == 2

    def test_token_estimate_is_nonzero_for_text(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello world") >= 2


# -- embeddings --------------------------------------------------------------


class TestEmbeddings:
    def test_deterministic_and_normalised(self, embedder):
        a = embedder.embed("lithium refining capacity")
        b = embedder.embed("lithium refining capacity")
        assert a == b
        assert sum(v * v for v in a) == pytest.approx(1.0, abs=1e-6)

    def test_similar_text_scores_higher_than_unrelated(self, embedder):
        base = embedder.embed("lithium brine extraction uses a lot of water")
        similar = embedder.embed("water usage in lithium brine extraction is high")
        unrelated = embedder.embed("the referee awarded a penalty in the second half")
        assert cosine(base, similar) > cosine(base, unrelated)

    def test_handles_empty_and_mismatched_input(self, embedder):
        assert embedder.embed("") == [0.0] * embedder.dims
        assert cosine([1.0, 0.0], []) == 0.0
        assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# -- retrieval ---------------------------------------------------------------


class TestRetrieval:
    def test_recall_returns_scored_entries(self, seeded, retriever):
        result = retriever.recall("lithium", "how much water does extraction use?")
        assert result
        assert all(0.0 <= item.score <= 1.0 for item in result.entries)
        assert result.candidates_considered >= len(result)

    def test_recall_respects_max_entries(self, seeded, retriever):
        assert len(retriever.recall("lithium", "water", max_entries=2)) == 2

    def test_recall_respects_token_budget(self, seeded, retriever):
        result = retriever.recall("lithium", "water", token_budget=20)
        assert result.tokens <= max(20, result.entries[0].entry.tokens)
        assert result.truncated

    def test_kind_filter(self, seeded, retriever):
        result = retriever.recall("lithium", "water", kinds=[EntryKind.CRITIQUE])
        assert {item.entry.kind for item in result.entries} == {EntryKind.CRITIQUE}

    def test_exclusion(self, seeded, retriever):
        first = retriever.recall("lithium", "water").entries[0].entry.id
        again = retriever.recall("lithium", "water", exclude_ids=[first])
        assert first not in {item.entry.id for item in again.entries}

    def test_recency_weighting_prefers_new_entries(self, store, embedder):
        old = Entry(topic="t", agent="a", content="shared subject matter alpha")
        old.created_at = utcnow() - timedelta(days=30)
        store.add_entry(old)
        store.add_entry(Entry(topic="t", agent="b", content="shared subject matter alpha"))

        retriever = MemoryRetriever(
            store,
            settings=RetrievalSettings(
                max_entries=2,
                candidate_pool=10,
                keyword_weight=0.0,
                semantic_weight=0.0,
                salience_weight=0.0,
                recency_weight=1.0,
            ),
            embedder=embedder,
        )
        top = retriever.recall("t", "shared subject").entries[0].entry
        assert top.agent == "b"

    def test_mmr_drops_near_duplicates(self, store, embedder):
        duplicate = "Refining capacity is the binding constraint on lithium supply growth."
        for agent in "abcd":
            store.add_entry(Entry(topic="t", agent=agent, content=duplicate))
        store.add_entry(
            Entry(
                topic="t",
                agent="e",
                content="Water consumption in Atacama brine ponds is contested.",
            )
        )

        diverse = MemoryRetriever(
            store,
            settings=RetrievalSettings(max_entries=2, candidate_pool=20, mmr_lambda=0.0),
            embedder=embedder,
        ).recall("t", "lithium supply")
        contents = {item.entry.content for item in diverse.entries}
        assert len(contents) == 2, "diversity-first selection should not return duplicates"

    def test_prompt_context_is_chronological_and_cited(self, seeded, retriever):
        result = retriever.recall("lithium", "water")
        block = result.as_prompt_context()
        assert "PRIOR MEMORY" in block
        assert "[1]" in block
        citations = result.citation_map()
        assert set(citations) == set(range(1, len(result) + 1))

    def test_empty_topic_recalls_nothing(self, store, retriever):
        assert not retriever.recall("does-not-exist", "anything")

    def test_index_pending_embeds_everything(self, seeded, retriever):
        assert retriever.index_pending(topic="lithium") == 5
        assert retriever.index_pending(topic="lithium") == 0


# -- compaction --------------------------------------------------------------


class TestCompaction:
    def _compactor(self, store, summary="Condensed briefing covering the prior findings."):
        return MemoryCompactor(
            store,
            lambda prompt: summary,
            threshold_tokens=10,
            keep_recent=1,
            min_entries_to_fold=2,
        )

    def test_declines_below_threshold(self, seeded):
        compactor = MemoryCompactor(seeded, lambda p: "x", threshold_tokens=10**9)
        result = compactor.maybe_compact("lithium")
        assert not result.performed
        assert "under threshold" in result.reason

    def test_folds_old_entries_into_a_summary(self, seeded):
        result = self._compactor(seeded).maybe_compact("lithium")
        assert result.performed
        assert result.tokens_after < result.tokens_before
        assert result.summary_entry is not None
        assert result.summary_entry.kind is EntryKind.SUMMARY

    def test_protected_kinds_survive(self, seeded):
        self._compactor(seeded).maybe_compact("lithium")
        live = seeded.query_entries(EntryFilter(topic="lithium", limit=50))
        # The user's note is protected and must still be live.
        assert any(e.kind is EntryKind.NOTE for e in live)

    def test_keeps_recent_entries(self, seeded):
        newest = seeded.timeline("lithium")[-1]
        self._compactor(seeded).maybe_compact("lithium")
        assert seeded.get_entry(newest.id).archived is False

    def test_refuses_a_summary_larger_than_its_input(self, seeded):
        compactor = self._compactor(seeded, summary="verbose " * 5000)
        result = compactor.maybe_compact("lithium")
        assert not result.performed
        assert "not smaller" in result.reason

    def test_refuses_an_empty_summary(self, seeded):
        result = self._compactor(seeded, summary="   ").maybe_compact("lithium")
        assert not result.performed

    def test_undo_restores_entries(self, seeded):
        compactor = self._compactor(seeded)
        result = compactor.maybe_compact("lithium")
        assert result.performed

        restored = compactor.undo(result.summary_entry.id)
        assert restored == len(result.archived_ids)
        live_ids = {e.id for e in seeded.query_entries(EntryFilter(topic="lithium", limit=50))}
        assert set(result.archived_ids).issubset(live_ids)
