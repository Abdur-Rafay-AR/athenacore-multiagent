"""Orchestration: graphs, parallel execution, debate and events."""

from __future__ import annotations

from crucible.orchestration.debate import DebateOrchestrator, DebateReport, DebateRound
from crucible.orchestration.events import (
    CancellationToken,
    Event,
    EventBus,
    EventQueue,
    EventType,
    console_printer,
)
from crucible.orchestration.graph import AgentGraph, GraphNode
from crucible.orchestration.orchestrator import Orchestrator, RunReport
from crucible.orchestration.presets import PRESETS, build_preset, describe_presets

__all__ = [
    "PRESETS",
    "AgentGraph",
    "CancellationToken",
    "DebateOrchestrator",
    "DebateReport",
    "DebateRound",
    "Event",
    "EventBus",
    "EventQueue",
    "EventType",
    "GraphNode",
    "Orchestrator",
    "RunReport",
    "build_preset",
    "console_printer",
    "describe_presets",
]
