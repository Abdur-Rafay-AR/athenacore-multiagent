"""The orchestrator.

Executes an :class:`~athenacore.orchestration.graph.AgentGraph` against a topic:
resolves each level's nodes, runs them in a thread pool, wires each node's output
into its dependents, persists everything, and emits events throughout.

Design commitments worth knowing about:

* **Failure is isolated.** A node that fails marks its whole downstream branch
  skipped, but unrelated branches finish and the run reports ``partial`` rather
  than losing all the work.
* **Threads, not asyncio.** Every provider call is a blocking HTTP request and
  every store write is blocking sqlite3. A thread pool matches that reality
  without forcing async through the entire codebase.
* **Memory is written as it happens**, not at the end, so a crashed or cancelled
  run still leaves behind whatever the completed agents learned.
* **Compaction runs after the graph**, so a long-lived topic stays inside its
  context budget without the operator thinking about it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from athenacore.agents.base import AgentContext, AgentResult
from athenacore.agents.registry import AgentFactory
from athenacore.config import Settings
from athenacore.errors import AthenaCoreError, RunCancelled
from athenacore.llm.base import LLMProvider, Message
from athenacore.llm.registry import build_provider
from athenacore.logging_setup import get_logger
from athenacore.memory.compaction import CompactionResult, MemoryCompactor
from athenacore.memory.embeddings import Embedder, HashingEmbedder
from athenacore.memory.models import Entry, EntryKind, Run, RunStatus, Usage, utcnow
from athenacore.memory.retrieval import MemoryRetriever
from athenacore.memory.store import MemoryStore
from athenacore.orchestration.events import (
    CancellationToken,
    EventBus,
    EventType,
)
from athenacore.orchestration.graph import AgentGraph
from athenacore.tools.base import ToolRegistry
from athenacore.tools.builtin import default_registry

log = get_logger(__name__)

Condition = Callable[["RunReport"], bool]


@dataclass(slots=True)
class RunReport:
    """Everything that happened during a run."""

    run: Run
    results: dict[str, AgentResult] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    compaction: CompactionResult | None = None

    @property
    def status(self) -> RunStatus:
        return self.run.status

    @property
    def usage(self) -> Usage:
        return self.run.usage

    @property
    def ok(self) -> bool:
        return self.run.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL}

    @property
    def failures(self) -> dict[str, str]:
        return {name: r.error or "unknown" for name, r in self.results.items() if not r.ok}

    def output(self, node: str | None = None) -> str:
        """The headline answer: a named node, or the last successful one."""
        if node is not None:
            result = self.results.get(node)
            return result.content if result else ""
        for name in reversed(list(self.results)):
            if self.results[name].ok and self.results[name].content:
                return self.results[name].content
        return ""

    def transcript(self) -> str:
        from athenacore.agents.base import format_transcript

        return format_transcript(list(self.results.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "skipped": self.skipped,
            "compaction": self.compaction.to_dict() if self.compaction else None,
            "output": self.output(),
        }


class Orchestrator:
    """Runs agent graphs against a topic's shared memory."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        bus: EventBus | None = None,
        embedder: Embedder | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store
        self.bus = bus or EventBus()
        self.provider = provider or build_provider(self.settings.model, self.settings)

        if embedder is None and self.settings.embeddings_enabled:
            embedder = HashingEmbedder(dims=self.settings.embedding_dims)
        self.embedder = embedder

        self.retriever = MemoryRetriever(
            store,
            settings=self.settings.retrieval,
            embedder=embedder,
            token_budget=self.settings.context_token_budget,
        )
        self._tools_override = tools
        self.compactor = MemoryCompactor(
            store,
            self._summarize_for_compaction,
            threshold_tokens=self.settings.compaction_threshold_tokens,
            keep_recent=self.settings.compaction_keep_recent,
        )
        self._conditions: dict[str, Condition] = {}

    # -- public API ----------------------------------------------------------

    def register_condition(self, name: str, predicate: Condition) -> None:
        """Register a predicate that :attr:`GraphNode.condition` can name."""
        self._conditions[name] = predicate

    def run(
        self,
        graph: AgentGraph,
        *,
        topic: str,
        query: str = "",
        cancel: CancellationToken | None = None,
        compact: bool = True,
        record_query: bool = True,
    ) -> RunReport:
        """Execute ``graph`` against ``topic``. Does not raise on node failure."""
        graph.validate()
        self.store.ensure_topic(topic)

        run = Run(
            topic=topic,
            preset=graph.name,
            query=query,
            status=RunStatus.RUNNING,
            model=f"{self.provider.name}:{self.provider.model}",
            node_states={node.name: "pending" for node in graph},
        )
        self.store.save_run(run)
        report = RunReport(run=run)
        cancel = cancel or CancellationToken()

        self.bus.publish(
            EventType.RUN_STARTED,
            run_id=run.id,
            message=f"{graph.name} · {len(graph)} nodes · topic {topic!r}",
            topic=topic,
            preset=graph.name,
            nodes=[n.name for n in graph],
            levels=graph.levels(),
            model=run.model,
        )

        if record_query and query.strip():
            self.store.add_entry(
                Entry(
                    topic=topic,
                    agent="user",
                    kind=EntryKind.QUESTION,
                    content=query.strip(),
                    run_id=run.id,
                    salience=0.75,  # the question frames everything after it
                )
            )

        factory = AgentFactory(
            self.provider,
            self.store,
            settings=self.settings,
            retriever=self.retriever,
            tools=self._tools_for(topic),
        )

        started = time.monotonic()
        try:
            self._execute_levels(graph, factory, report, topic, query, cancel)
            run.status = RunStatus.PARTIAL if report.failures else RunStatus.SUCCEEDED
        except RunCancelled:
            run.status = RunStatus.CANCELLED
            self.bus.publish(EventType.RUN_CANCELLED, run_id=run.id, message="cancelled")
        except AthenaCoreError as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            self.bus.publish(EventType.RUN_FAILED, run_id=run.id, message=str(exc))
        finally:
            run.ended_at = utcnow()
            run.usage = _sum_usage(report.results.values())
            self.store.save_run(run)

        if compact and run.status.terminal and run.status is not RunStatus.FAILED:
            report.compaction = self._maybe_compact(topic, run.id)

        # Keep the vector index warm so the next run's recall is instant.
        if self.embedder is not None:
            self.retriever.index_pending(topic=topic)

        if run.status is not RunStatus.FAILED:
            self.bus.publish(
                EventType.RUN_FINISHED,
                run_id=run.id,
                message=(
                    f"{run.status.value} in {int((time.monotonic() - started) * 1000)}ms · "
                    f"{run.usage.total_tokens} tokens"
                ),
                status=run.status.value,
                usage=run.usage.to_dict(),
                failures=report.failures,
            )
        log.info(
            "run finished",
            extra={
                "run": run.id,
                "topic": topic,
                "status": run.status.value,
                "tokens": run.usage.total_tokens,
                "ms": run.duration_ms,
            },
        )
        return report

    def run_agent(
        self,
        agent_name: str,
        *,
        topic: str,
        query: str = "",
    ) -> RunReport:
        """Convenience path for a single agent - the old ``run_agent`` behaviour."""
        graph = AgentGraph.linear(agent_name, name=f"solo:{agent_name}")
        return self.run(graph, topic=topic, query=query)

    # -- execution -----------------------------------------------------------

    def _execute_levels(
        self,
        graph: AgentGraph,
        factory: AgentFactory,
        report: RunReport,
        topic: str,
        query: str,
        cancel: CancellationToken,
    ) -> None:
        run = report.run
        skipped: set[str] = set()
        workers = max(1, min(self.settings.max_parallel_agents, graph.max_width))

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="athena") as pool:
            for level in graph.levels():
                cancel.raise_if_cancelled()
                runnable = [name for name in level if name not in skipped]

                for name in level:
                    if name in skipped:
                        run.node_states[name] = "skipped"
                        report.skipped.append(name)
                        self.bus.publish(
                            EventType.NODE_SKIPPED,
                            run_id=run.id,
                            node=name,
                            message="skipped: an upstream node failed",
                        )

                # Evaluate conditions before dispatching, so a condition can look
                # at everything produced by earlier levels.
                dispatched: dict[str, Future[AgentResult]] = {}
                for name in runnable:
                    node = graph.get(name)
                    assert node is not None
                    if node.condition and not self._check_condition(node.condition, report):
                        run.node_states[name] = "skipped"
                        report.skipped.append(name)
                        self.bus.publish(
                            EventType.NODE_SKIPPED,
                            run_id=run.id,
                            node=name,
                            message=f"skipped: condition {node.condition!r} was false",
                        )
                        skipped.update(graph.descendants(name))
                        continue

                    ctx = AgentContext(
                        topic=topic,
                        task=node.render_task(query=query, topic=topic),
                        run_id=run.id,
                        upstream={
                            dep: report.results[dep].content
                            for dep in node.depends_on
                            if dep in report.results and report.results[dep].ok
                        },
                    )
                    agent = factory.create(node.agent, **node.overrides)
                    run.node_states[name] = "running"
                    self.bus.publish(
                        EventType.NODE_STARTED,
                        run_id=run.id,
                        node=name,
                        message=f"{agent.role} working…",
                        agent=node.agent,
                        role=agent.role,
                    )
                    dispatched[name] = pool.submit(agent.run, ctx)

                for name, future in dispatched.items():
                    try:
                        result = future.result()
                    except Exception as exc:  # a crash inside an agent
                        result = AgentResult(
                            agent=name,
                            content="",
                            kind=EntryKind.ERROR,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    report.results[name] = result
                    node = graph.get(name)
                    assert node is not None
                    self._finish_node(report, graph, node, result, skipped)

                self.store.save_run(run)

    def _finish_node(
        self,
        report: RunReport,
        graph: AgentGraph,
        node: Any,
        result: AgentResult,
        skipped: set[str],
    ) -> None:
        run = report.run
        if result.ok:
            run.node_states[node.name] = "succeeded"
            for tool_result in result.tool_results:
                self.bus.publish(
                    EventType.TOOL_CALLED,
                    run_id=run.id,
                    node=node.name,
                    message=f"{tool_result.call.name} → {'ok' if tool_result.ok else 'error'}",
                    **tool_result.to_dict(),
                )
            if result.recall:
                self.bus.publish(
                    EventType.MEMORY_RECALLED,
                    run_id=run.id,
                    node=node.name,
                    message=f"recalled {len(result.recall)} entries (~{result.recall.tokens} tokens)",
                    entries=[s.to_dict() for s in result.recall.entries],
                )
            if result.entry:
                self.bus.publish(
                    EventType.MEMORY_WRITTEN,
                    run_id=run.id,
                    node=node.name,
                    message=f"wrote {result.entry.kind.value} entry",
                    entry_id=result.entry.id,
                )
            self.bus.publish(
                EventType.NODE_FINISHED,
                run_id=run.id,
                node=node.name,
                message=f"done in {result.duration_ms}ms ({result.usage.total_tokens} tokens)",
                content=result.content,
                usage=result.usage.to_dict(),
                citations=result.citations,
            )
            return

        run.node_states[node.name] = "failed"
        self.bus.publish(
            EventType.NODE_FAILED,
            run_id=run.id,
            node=node.name,
            message=result.error or "failed",
            optional=node.optional,
        )
        if not node.optional:
            # Only this branch dies; independent branches keep going.
            skipped.update(graph.descendants(node.name))

    def _check_condition(self, name: str, report: RunReport) -> bool:
        predicate = self._conditions.get(name)
        if predicate is None:
            log.warning("unknown graph condition, running node anyway", extra={"condition": name})
            return True
        try:
            return bool(predicate(report))
        except Exception as exc:
            log.warning(
                "condition raised, running node anyway",
                extra={"condition": name, "error": str(exc)},
            )
            return True

    # -- collaborators -------------------------------------------------------

    def _tools_for(self, topic: str) -> ToolRegistry:
        if self._tools_override is not None:
            return self._tools_override
        if not self.settings.tools_enabled:
            return ToolRegistry()
        return default_registry(
            self.store,
            topic=topic,
            web_search=self.settings.web_search_enabled,
        )

    def _summarize_for_compaction(self, prompt: str) -> str:
        """Compaction's summariser: one direct, low-temperature model call."""
        completion = self.provider.complete(
            [
                Message.system("You compress technical memory without losing facts."),
                Message.user(prompt),
            ],
            temperature=0.2,
        )
        return completion.text

    def _maybe_compact(self, topic: str, run_id: str) -> CompactionResult | None:
        try:
            result = self.compactor.maybe_compact(topic)
        except AthenaCoreError as exc:
            # Compaction is an optimisation; never fail a run over it.
            log.warning("compaction failed", extra={"topic": topic, "error": str(exc)})
            return None
        if result.performed:
            self.bus.publish(
                EventType.COMPACTED,
                run_id=run_id,
                message=(
                    f"compacted {len(result.archived_ids)} entries "
                    f"({result.tokens_before} → {result.tokens_after} tokens)"
                ),
                **result.to_dict(),
            )
        return result


def _sum_usage(results: Sequence[AgentResult] | Any) -> Usage:
    total = Usage()
    for result in results:
        total = total + result.usage
    return total
