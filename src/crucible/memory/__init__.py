"""Persistent memory: models, stores, retrieval and compaction."""

from __future__ import annotations

from crucible.memory.compaction import CompactionResult, MemoryCompactor
from crucible.memory.embeddings import CallableEmbedder, Embedder, HashingEmbedder, cosine
from crucible.memory.models import (
    Entry,
    EntryKind,
    Run,
    RunStatus,
    Topic,
    Usage,
    estimate_tokens,
)
from crucible.memory.retrieval import MemoryRetriever, RecallResult
from crucible.memory.sqlite_store import SqliteMemoryStore
from crucible.memory.store import (
    EntryFilter,
    InMemoryMemoryStore,
    MemoryStore,
    ScoredEntry,
)

__all__ = [
    "CallableEmbedder",
    "CompactionResult",
    "Embedder",
    "Entry",
    "EntryFilter",
    "EntryKind",
    "HashingEmbedder",
    "InMemoryMemoryStore",
    "MemoryCompactor",
    "MemoryRetriever",
    "MemoryStore",
    "RecallResult",
    "Run",
    "RunStatus",
    "ScoredEntry",
    "SqliteMemoryStore",
    "Topic",
    "Usage",
    "cosine",
    "estimate_tokens",
]
