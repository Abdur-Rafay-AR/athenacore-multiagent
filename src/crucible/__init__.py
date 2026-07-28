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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

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

# Single-sourced from the installed distribution metadata, so pyproject.toml is
# the only place a version number is written. Duplicating it in code is the
# classic way to ship a release whose reported version is a lie.
try:
    __version__ = _pkg_version("crucible-agents")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0.dev0"

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
