"""Crucible: a multi-agent collaboration engine with persistent memory.

The public surface is deliberately small. Most callers need four things::

    from crucible import Settings, SqliteMemoryStore, build_preset, Orchestrator

    settings = Settings.from_env()
    store = SqliteMemoryStore(settings.database_path)
    graph = build_preset("research-critique-synthesis", settings=settings)
    report = Orchestrator(store=store, settings=settings).run(graph, topic="lithium supply")

Everything else (providers, tools, retrieval knobs, the agent registry) is
importable from its own module and is documented in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from crucible.config import RetrievalSettings, Settings
from crucible.errors import (
    ConfigurationError,
    CrucibleError,
    GraphError,
    ProviderError,
    ToolError,
)
from crucible.memory.models import Entry, EntryKind, Run, RunStatus, Topic
from crucible.memory.sqlite_store import SqliteMemoryStore
from crucible.memory.store import MemoryStore
from crucible.orchestration.graph import AgentGraph, GraphNode
from crucible.orchestration.orchestrator import Orchestrator, RunReport
from crucible.orchestration.presets import PRESETS, build_preset

__version__ = "0.3.0"

__all__ = [
    "PRESETS",
    "AgentGraph",
    "ConfigurationError",
    "CrucibleError",
    "Entry",
    "EntryKind",
    "GraphError",
    "GraphNode",
    "MemoryStore",
    "Orchestrator",
    "ProviderError",
    "RetrievalSettings",
    "Run",
    "RunReport",
    "RunStatus",
    "Settings",
    "SqliteMemoryStore",
    "ToolError",
    "Topic",
    "__version__",
    "build_preset",
]
