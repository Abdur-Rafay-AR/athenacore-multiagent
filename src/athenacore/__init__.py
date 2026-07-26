"""AthenaCore: a multi-agent collaboration engine with persistent memory.

The public surface is deliberately small. Most callers need four things::

    from athenacore import Settings, SqliteMemoryStore, build_preset, Orchestrator

    settings = Settings.from_env()
    store = SqliteMemoryStore(settings.database_path)
    graph = build_preset("research-critique-synthesis", settings=settings)
    report = Orchestrator(store=store, settings=settings).run(graph, topic="lithium supply")

Everything else (providers, tools, retrieval knobs, the agent registry) is
importable from its own module and is documented in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from athenacore.config import RetrievalSettings, Settings
from athenacore.errors import (
    AthenaCoreError,
    ConfigurationError,
    GraphError,
    ProviderError,
    ToolError,
)
from athenacore.memory.models import Entry, EntryKind, Run, RunStatus, Topic
from athenacore.memory.sqlite_store import SqliteMemoryStore
from athenacore.memory.store import MemoryStore
from athenacore.orchestration.graph import AgentGraph, GraphNode
from athenacore.orchestration.orchestrator import Orchestrator, RunReport
from athenacore.orchestration.presets import PRESETS, build_preset

__version__ = "0.2.0"

__all__ = [
    "PRESETS",
    "AgentGraph",
    "AthenaCoreError",
    "ConfigurationError",
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
