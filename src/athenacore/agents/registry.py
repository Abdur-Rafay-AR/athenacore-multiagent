"""Agent registry and construction.

Agents are looked up by name everywhere - CLI flags, graph definitions, API
payloads, UI dropdowns - so they live in one registry that also discovers
third-party agents through the ``athenacore.agents`` entry-point group. Shipping
an agent as a separate pip package therefore requires no changes here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

from athenacore.config import Settings
from athenacore.errors import ConfigurationError
from athenacore.logging_setup import get_logger
from athenacore.memory.retrieval import MemoryRetriever
from athenacore.memory.store import MemoryStore

if TYPE_CHECKING:  # pragma: no cover
    from athenacore.agents.base import Agent
    from athenacore.llm.base import LLMProvider
    from athenacore.tools.base import ToolRegistry

log = get_logger(__name__)

_REGISTRY: dict[str, type[Agent]] = {}
_PLUGINS_LOADED = False

A = TypeVar("A", bound="type[Agent]")


def register_agent(cls: A) -> A:
    """Class decorator that adds an agent to the registry under its ``name``."""
    name = getattr(cls, "name", "").strip().lower()
    if not name or name == "agent":
        raise ConfigurationError(f"{cls.__name__} must define a unique class-level `name`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ConfigurationError(
            f"agent name {name!r} is already registered by {_REGISTRY[name].__name__}"
        )
    _REGISTRY[name] = cls
    return cls


def _load_plugins() -> None:
    """Import agents contributed by other installed packages. Best effort."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        from importlib.metadata import entry_points

        for entry in entry_points(group="athenacore.agents"):
            try:
                loaded = entry.load()
            except Exception as exc:  # a broken plugin must not break startup
                log.warning(
                    "agent plugin failed to load", extra={"plugin": entry.name, "error": str(exc)}
                )
                continue
            plugin_name = getattr(loaded, "name", None)
            if isinstance(loaded, type) and isinstance(plugin_name, str) and plugin_name:
                _REGISTRY.setdefault(plugin_name.lower(), cast("type[Agent]", loaded))
                log.debug("loaded agent plugin", extra={"agent": plugin_name})
    except Exception as exc:  # pragma: no cover - importlib.metadata edge cases
        log.debug("plugin discovery skipped", extra={"error": str(exc)})


def _ensure_builtins() -> None:
    # Importing for side effects: the decorators populate the registry.
    import athenacore.agents.builtin  # noqa: F401

    _load_plugins()


def available_agents() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)


def get_agent_class(name: str) -> type[Agent]:
    _ensure_builtins()
    key = name.strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ConfigurationError(
            f"unknown agent {name!r}",
            hint=f"Available agents: {', '.join(available_agents())}",
        )
    return cls


def describe_agents() -> list[dict[str, Any]]:
    """Metadata for every registered agent - powers ``athenacore agents`` and
    the UI's agent picker."""
    _ensure_builtins()
    out = []
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        out.append(
            {
                "name": name,
                "role": getattr(cls, "role", ""),
                "description": (
                    getattr(cls, "description", "") or (cls.__doc__ or "").strip().split("\n")[0]
                ),
                "kind": cls.entry_kind.value,
                "temperature": getattr(cls, "temperature", None),
                "uses_tools": getattr(cls, "uses_tools", True),
            }
        )
    return out


class AgentFactory:
    """Builds configured agents, sharing one provider, store and retriever.

    Sharing the retriever matters: it holds the embedder, so a single embedding
    model (and its cache) is reused across every agent in a run.
    """

    def __init__(
        self,
        provider: LLMProvider,
        store: MemoryStore,
        *,
        settings: Settings | None = None,
        retriever: MemoryRetriever | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.settings = settings or Settings()
        self.retriever = retriever or MemoryRetriever(
            store,
            settings=self.settings.retrieval,
            token_budget=self.settings.context_token_budget,
        )
        self.tools = tools
        self._cache: dict[str, Agent] = {}

    def create(self, name: str, **overrides: Any) -> Agent:
        cls = get_agent_class(name)
        agent = cls(
            self.provider,
            self.store,
            retriever=self.retriever,
            tools=self.tools,
            max_tool_calls=self.settings.max_tool_calls_per_turn,
        )
        for key, value in overrides.items():
            setattr(agent, key, value)
        return agent

    def get(self, name: str) -> Agent:
        """Cached variant - repeated graph nodes reuse one instance."""
        key = name.strip().lower()
        if key not in self._cache:
            self._cache[key] = self.create(key)
        return self._cache[key]

    def create_many(self, names: Sequence[str]) -> list[Agent]:
        return [self.create(n) for n in names]
