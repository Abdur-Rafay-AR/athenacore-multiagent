"""Ready-made graphs.

Presets are the fast path: they encode workflows that are known to work so a new
user gets a good result before learning the graph API. Each one is an ordinary
:class:`AgentGraph`, so ``build_preset(...).to_dict()`` is also the best way to
learn how to write your own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from athenacore.config import Settings
from athenacore.errors import ConfigurationError
from athenacore.orchestration.graph import AgentGraph, GraphNode


@dataclass(frozen=True, slots=True)
class Preset:
    key: str
    title: str
    summary: str
    build: Callable[[Settings], AgentGraph]
    cost: str = "medium"
    """Rough token appetite: low / medium / high."""

    def to_dict(self) -> dict[str, Any]:
        graph = self.build(Settings())
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "cost": self.cost,
            "nodes": [n.name for n in graph],
            "levels": graph.levels(),
            "mermaid": graph.to_mermaid(),
        }


def _solo(agent: str) -> Callable[[Settings], AgentGraph]:
    def build(settings: Settings) -> AgentGraph:
        return AgentGraph(
            [GraphNode(name=agent, agent=agent)],
            name=f"solo:{agent}",
            description=f"Run {agent} alone.",
        )

    return build


def _research_critique_synthesis(settings: Settings) -> AgentGraph:
    """The workhorse: investigate, attack, then adjudicate.

    Research and fact-checking are independent, so they run concurrently; the
    critic waits for research; the synthesizer waits for both.
    """
    return AgentGraph(
        [
            GraphNode(name="research", agent="research"),
            GraphNode(
                name="factcheck",
                agent="factcheck",
                depends_on=("research",),
                optional=True,  # a clean audit is nice-to-have, not load-bearing
            ),
            GraphNode(name="critic", agent="critic", depends_on=("research",)),
            GraphNode(
                name="synthesizer",
                agent="synthesizer",
                depends_on=("critic", "factcheck"),
            ),
        ],
        name="research-critique-synthesis",
        description="Investigate a question, attack the findings, then adjudicate.",
    )


def _deep_dive(settings: Settings) -> AgentGraph:
    """Everything, ending in a plan. Expensive but thorough."""
    return AgentGraph(
        [
            GraphNode(name="research", agent="research"),
            GraphNode(name="critic", agent="critic", depends_on=("research",)),
            GraphNode(name="factcheck", agent="factcheck", depends_on=("research",), optional=True),
            GraphNode(name="insight", agent="insight", depends_on=("research", "critic")),
            GraphNode(
                name="synthesizer",
                agent="synthesizer",
                depends_on=("critic", "insight", "factcheck"),
            ),
            GraphNode(name="planner", agent="planner", depends_on=("synthesizer",)),
        ],
        name="deep-dive",
        description="Full pipeline: research, critique, audit, insight, synthesis, plan.",
    )


def _brief(settings: Settings) -> AgentGraph:
    """Cheap: one research pass, then a summary. Good for local 7B models."""
    return AgentGraph(
        [
            GraphNode(name="research", agent="research"),
            GraphNode(name="summarizer", agent="summarizer", depends_on=("research",)),
        ],
        name="brief",
        description="A single research pass condensed into a briefing.",
    )


def _catch_up(settings: Settings) -> AgentGraph:
    """No new research — just re-read the topic and report its state.

    This is the preset that shows off persistent memory: run it on a topic you
    worked on last week and it tells you where things stand.
    """
    return AgentGraph(
        [
            GraphNode(
                name="summarizer",
                agent="summarizer",
                task="Summarise the current state of {topic} for someone returning to it.",
            ),
            GraphNode(
                name="insight",
                agent="insight",
                depends_on=("summarizer",),
                task="What are the most decision-relevant implications of the state of {topic}?",
            ),
        ],
        name="catch-up",
        description="Re-read a topic's memory and report where things stand.",
    )


def _red_team(settings: Settings) -> AgentGraph:
    """Two independent critics attacking in parallel, then adjudication.

    Two critics at different temperatures find meaningfully different objections
    than one critic asked twice — the parallel structure is doing real work here.
    """
    return AgentGraph(
        [
            GraphNode(name="research", agent="research"),
            GraphNode(
                name="critic_assumptions",
                agent="critic",
                depends_on=("research",),
                task="Attack the assumptions and causal claims behind: {query}",
                overrides={"temperature": 0.4},
            ),
            GraphNode(
                name="critic_failure_modes",
                agent="critic",
                depends_on=("research",),
                task="Attack the failure modes, incentives and second-order risks in: {query}",
                overrides={"temperature": 0.8},
            ),
            GraphNode(name="factcheck", agent="factcheck", depends_on=("research",), optional=True),
            GraphNode(
                name="synthesizer",
                agent="synthesizer",
                depends_on=("critic_assumptions", "critic_failure_modes", "factcheck"),
            ),
        ],
        name="red-team",
        description="Research, then two independent critics, then adjudication.",
    )


PRESETS: dict[str, Preset] = {
    preset.key: preset
    for preset in [
        Preset(
            key="brief",
            title="Brief",
            summary="One research pass, condensed. Cheapest useful run.",
            build=_brief,
            cost="low",
        ),
        Preset(
            key="research-critique-synthesis",
            title="Research → Critique → Synthesis",
            summary="The default. Investigate, attack the findings, adjudicate.",
            build=_research_critique_synthesis,
            cost="medium",
        ),
        Preset(
            key="red-team",
            title="Red Team",
            summary="Two independent critics attack in parallel before adjudication.",
            build=_red_team,
            cost="high",
        ),
        Preset(
            key="deep-dive",
            title="Deep Dive",
            summary="Everything, ending in an actionable plan.",
            build=_deep_dive,
            cost="high",
        ),
        Preset(
            key="catch-up",
            title="Catch Up",
            summary="No new research: report where an existing topic stands.",
            build=_catch_up,
            cost="low",
        ),
    ]
}

# Single-agent presets, so `--preset solo:critic` works for any registered agent.
SOLO_PREFIX = "solo:"


def build_preset(key: str, *, settings: Settings | None = None) -> AgentGraph:
    """Resolve a preset key (or ``solo:<agent>``) into a validated graph."""
    settings = settings or Settings()
    key = key.strip()
    if key.startswith(SOLO_PREFIX):
        agent = key[len(SOLO_PREFIX) :].strip().lower()
        from athenacore.agents.registry import get_agent_class

        get_agent_class(agent)  # raises with a helpful hint if unknown
        graph = _solo(agent)(settings)
        graph.validate()
        return graph

    preset = PRESETS.get(key)
    if preset is None:
        raise ConfigurationError(
            f"unknown preset {key!r}",
            hint=f"Available: {', '.join(sorted(PRESETS))}, or solo:<agent>",
        )
    graph = preset.build(settings)
    graph.validate()
    return graph


def describe_presets() -> list[dict[str, Any]]:
    return [PRESETS[key].to_dict() for key in sorted(PRESETS)]
