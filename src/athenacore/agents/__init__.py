"""Agents: roles, recall policies and prompts over a shared execution cycle."""

from __future__ import annotations

from athenacore.agents.base import (
    Agent,
    AgentContext,
    AgentResult,
    RecallPolicy,
    format_transcript,
)
from athenacore.agents.builtin import (
    CriticAgent,
    FactCheckAgent,
    InsightAgent,
    PlannerAgent,
    ResearchAgent,
    SummarizerAgent,
    SynthesizerAgent,
)
from athenacore.agents.registry import (
    AgentFactory,
    available_agents,
    describe_agents,
    get_agent_class,
    register_agent,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentFactory",
    "AgentResult",
    "CriticAgent",
    "FactCheckAgent",
    "InsightAgent",
    "PlannerAgent",
    "RecallPolicy",
    "ResearchAgent",
    "SummarizerAgent",
    "SynthesizerAgent",
    "available_agents",
    "describe_agents",
    "format_transcript",
    "get_agent_class",
    "register_agent",
]
