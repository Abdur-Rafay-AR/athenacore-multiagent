"""The agent graph.

A run is a DAG of agent invocations. Nodes declare their dependencies, and the
executor runs each *level* of the topological order in parallel - so a research
agent and a fact-checker with no relationship to each other genuinely run at the
same time, while a synthesizer waits for both.

The graph is data, not code: it can be built in Python, loaded from JSON/YAML, or
assembled by a preset, and rendered as Mermaid for the UI.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.errors import GraphError


@dataclass(slots=True)
class GraphNode:
    """One agent invocation inside a graph."""

    name: str
    """Unique node id. Also the key upstream outputs are passed under."""

    agent: str
    """Registered agent name to instantiate."""

    depends_on: tuple[str, ...] = ()
    task: str | None = None
    """Node-specific instruction. ``None`` inherits the run's query.
    Supports ``{query}`` and ``{topic}`` placeholders."""

    optional: bool = False
    """When True, a failure here does not fail dependents or the run."""

    condition: str | None = None
    """Name of a predicate registered on the graph; the node is skipped when it
    evaluates false. Used for 'only run the planner if we reached a conclusion'."""

    overrides: dict[str, Any] = field(default_factory=dict)
    """Attributes patched onto the agent instance (e.g. ``temperature``)."""

    def render_task(self, *, query: str, topic: str) -> str:
        if self.task is None:
            return query
        try:
            return self.task.format(query=query, topic=topic)
        except (KeyError, IndexError):
            # A stray brace in a hand-written task should not crash the run.
            return self.task

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agent": self.agent,
            "depends_on": list(self.depends_on),
            "task": self.task,
            "optional": self.optional,
            "condition": self.condition,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls(
            name=data["name"],
            agent=data.get("agent", data["name"]),
            depends_on=tuple(data.get("depends_on") or ()),
            task=data.get("task"),
            optional=bool(data.get("optional", False)),
            condition=data.get("condition"),
            overrides=dict(data.get("overrides") or {}),
        )


class AgentGraph:
    """A validated DAG of agent nodes."""

    def __init__(
        self,
        nodes: Iterable[GraphNode] = (),
        *,
        name: str = "custom",
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._nodes: dict[str, GraphNode] = {}
        for node in nodes:
            self.add(node)

    # -- construction --------------------------------------------------------

    def add(self, node: GraphNode) -> AgentGraph:
        if node.name in self._nodes:
            raise GraphError(f"duplicate node name {node.name!r}")
        self._nodes[node.name] = node
        return self

    def then(self, name: str, agent: str | None = None, **kwargs: Any) -> AgentGraph:
        """Append a node depending on all current leaves - the linear-chain
        shorthand that covers most hand-built graphs::

            AgentGraph().then("research").then("critic").then("synthesizer")
        """
        depends = kwargs.pop("depends_on", None)
        if depends is None:
            depends = tuple(self.leaves())
        self.add(GraphNode(name=name, agent=agent or name, depends_on=tuple(depends), **kwargs))
        return self

    def parallel(
        self, *names: str, depends_on: Sequence[str] | None = None, **kwargs: Any
    ) -> AgentGraph:
        """Add several independent nodes sharing the same dependencies."""
        deps = tuple(depends_on) if depends_on is not None else tuple(self.leaves())
        for name in names:
            self.add(GraphNode(name=name, agent=name, depends_on=deps, **kwargs))
        return self

    # -- inspection ----------------------------------------------------------

    @property
    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def get(self, name: str) -> GraphNode | None:
        return self._nodes.get(name)

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[GraphNode]:
        return iter(self._nodes.values())

    def __contains__(self, name: object) -> bool:
        return name in self._nodes

    def leaves(self) -> list[str]:
        """Nodes nothing depends on - the graph's outputs."""
        depended = {dep for node in self._nodes.values() for dep in node.depends_on}
        return [name for name in self._nodes if name not in depended]

    def roots(self) -> list[str]:
        return [name for name, node in self._nodes.items() if not node.depends_on]

    def dependents(self, name: str) -> list[str]:
        return [n.name for n in self._nodes.values() if name in n.depends_on]

    def descendants(self, name: str) -> set[str]:
        """Every node reachable from ``name`` - used to skip a failed branch."""
        seen: set[str] = set()
        stack = list(self.dependents(name))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.dependents(current))
        return seen

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Check the graph is non-empty, references only known nodes, and is acyclic."""
        if not self._nodes:
            raise GraphError("graph has no nodes")
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise GraphError(
                        f"node {node.name!r} depends on unknown node {dep!r}",
                        hint=f"Known nodes: {', '.join(self._nodes)}",
                    )
            if node.name in node.depends_on:
                raise GraphError(f"node {node.name!r} depends on itself")
        self.levels()  # raises on a cycle

    def levels(self) -> list[list[str]]:
        """Topological order grouped into parallelisable levels (Kahn's algorithm).

        Returning levels rather than a flat order is the whole point: everything
        inside one level is independent and can be dispatched concurrently.
        """
        indegree = {name: len(node.depends_on) for name, node in self._nodes.items()}
        ready = sorted(name for name, degree in indegree.items() if degree == 0)
        if not ready and self._nodes:
            raise GraphError("graph contains a cycle: every node has a dependency")

        levels: list[list[str]] = []
        resolved: set[str] = set()
        while ready:
            levels.append(ready)
            resolved.update(ready)
            nxt: list[str] = []
            for name in ready:
                for dependent in self.dependents(name):
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        nxt.append(dependent)
            ready = sorted(nxt)

        if len(resolved) != len(self._nodes):
            unresolved = sorted(set(self._nodes) - resolved)
            raise GraphError(
                f"graph contains a cycle involving: {', '.join(unresolved)}",
                hint="Remove a dependency to break the loop.",
            )
        return levels

    def order(self) -> list[str]:
        """Flat topological order."""
        return [name for level in self.levels() for name in level]

    @property
    def max_width(self) -> int:
        """Widest level - the most agents that can run at once."""
        return max((len(level) for level in self.levels()), default=0)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self._nodes.values()],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentGraph:
        graph = cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
        )
        for node in data.get("nodes") or []:
            graph.add(GraphNode.from_dict(node))
        graph.validate()
        return graph

    @classmethod
    def from_json(cls, raw: str) -> AgentGraph:
        return cls.from_dict(json.loads(raw))

    @classmethod
    def linear(cls, *agents: str, name: str = "linear") -> AgentGraph:
        """Build a simple chain: each agent depends on the previous one."""
        graph = cls(name=name)
        previous: tuple[str, ...] = ()
        for agent in agents:
            graph.add(GraphNode(name=agent, agent=agent, depends_on=previous))
            previous = (agent,)
        graph.validate()
        return graph

    # -- rendering -----------------------------------------------------------

    def to_mermaid(self, *, states: dict[str, str] | None = None) -> str:
        """Mermaid flowchart source, optionally tinted by per-node state.

        The UI renders this directly, so a user can see the topology of the run
        they are about to start and watch it light up as it executes.
        """
        states = states or {}
        lines = ["flowchart LR"]
        for node in self._nodes.values():
            label = node.name if node.name == node.agent else f"{node.name}<br/><i>{node.agent}</i>"
            lines.append(f'    {_safe_id(node.name)}["{label}"]')
        for node in self._nodes.values():
            for dep in node.depends_on:
                lines.append(f"    {_safe_id(dep)} --> {_safe_id(node.name)}")

        palette = {
            "succeeded": "#1f9d55,#e6f6ee",
            "running": "#2b6cb0,#e6f0fa",
            "failed": "#c53030,#fbeaea",
            "skipped": "#718096,#edf2f7",
        }
        for name, state in states.items():
            if name in self._nodes and state in palette:
                stroke, fill = palette[state].split(",")
                lines.append(
                    f"    style {_safe_id(name)} fill:{fill},stroke:{stroke},stroke-width:2px"
                )
        return "\n".join(lines)

    def to_ascii(self) -> str:
        """Plain-text topology for terminals that cannot render Mermaid."""
        out = []
        for depth, level in enumerate(self.levels()):
            prefix = "  " * depth
            for name in level:
                node = self._nodes[name]
                deps = f"  <- {', '.join(node.depends_on)}" if node.depends_on else ""
                flag = " (optional)" if node.optional else ""
                out.append(f"{prefix}{name} [{node.agent}]{flag}{deps}")
        return "\n".join(out)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<AgentGraph {self.name!r} nodes={len(self._nodes)} depth={len(self.levels())}>"


def _safe_id(name: str) -> str:
    """Mermaid node ids must be alphanumeric-ish."""
    return "n_" + "".join(c if c.isalnum() else "_" for c in name)
