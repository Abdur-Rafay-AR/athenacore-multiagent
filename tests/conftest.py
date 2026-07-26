"""Shared fixtures.

Every test runs fully offline against the deterministic Echo provider and a
temporary SQLite file, so the suite needs no model, no daemon and no network.
"""

from __future__ import annotations

import pytest

from athenacore.config import RetrievalSettings, Settings
from athenacore.llm.providers import EchoProvider, ScriptedProvider
from athenacore.memory.embeddings import HashingEmbedder
from athenacore.memory.models import Entry, EntryKind
from athenacore.memory.retrieval import MemoryRetriever
from athenacore.memory.sqlite_store import SqliteMemoryStore
from athenacore.memory.store import InMemoryMemoryStore


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        model="echo:test",
        database_path=tmp_path / "memory.sqlite3",
        data_dir=tmp_path,
        embeddings_enabled=True,
        embedding_dims=64,
        max_parallel_agents=3,
        retrieval=RetrievalSettings(max_entries=6, candidate_pool=40),
    )


@pytest.fixture
def store(tmp_path) -> SqliteMemoryStore:
    store = SqliteMemoryStore(tmp_path / "memory.sqlite3")
    yield store
    store.close()


@pytest.fixture(params=["sqlite", "memory"])
def any_store(request, tmp_path):
    """Both store implementations, so the conformance suite covers each."""
    if request.param == "sqlite":
        store = SqliteMemoryStore(tmp_path / "conformance.sqlite3")
        yield store
        store.close()
    else:
        yield InMemoryMemoryStore()


@pytest.fixture
def provider() -> EchoProvider:
    return EchoProvider("test")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dims=64)


@pytest.fixture
def retriever(store, embedder) -> MemoryRetriever:
    return MemoryRetriever(
        store,
        settings=RetrievalSettings(max_entries=6, candidate_pool=40),
        embedder=embedder,
    )


@pytest.fixture
def scripted():
    """Factory for a provider returning exact, pre-written text."""

    def make(*responses: str) -> ScriptedProvider:
        return ScriptedProvider(list(responses))

    return make


SAMPLE_ENTRIES = [
    (
        "research",
        EntryKind.RESEARCH,
        "Lithium brine extraction in Chile's Atacama consumes roughly 500,000 litres of water "
        "per tonne of lithium carbonate and takes 18 months to evaporate.",
    ),
    (
        "research",
        EntryKind.RESEARCH,
        "Hard rock spodumene mining in Western Australia supplies about 47 percent of global "
        "lithium, with a far shorter production cycle than brine.",
    ),
    (
        "critic",
        EntryKind.CRITIQUE,
        "The water consumption figure ignores closed-loop direct lithium extraction pilots "
        "in Argentina, which recycle most of the brine.",
    ),
    (
        "insight",
        EntryKind.INSIGHT,
        "Refining capacity rather than raw extraction is the binding constraint; China refines "
        "the majority of the world's spodumene concentrate.",
    ),
    (
        "user",
        EntryKind.NOTE,
        "Keep the analysis focused on 2030 electric vehicle demand scenarios.",
    ),
]


@pytest.fixture
def seeded(store) -> SqliteMemoryStore:
    """A store preloaded with a small, realistic topic."""
    store.ensure_topic("lithium", description="Battery supply chain", tags=["energy"])
    for index, (agent, kind, content) in enumerate(SAMPLE_ENTRIES):
        store.add_entry(
            Entry(
                topic="lithium",
                agent=agent,
                kind=kind,
                content=content,
                salience=0.4 + index * 0.1,
            )
        )
    return store
