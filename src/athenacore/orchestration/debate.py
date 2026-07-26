"""Multi-round debate with automatic convergence detection.

A DAG runs each agent once. A debate runs them repeatedly against each other's
latest positions, which is where multi-agent systems actually earn their cost:
the critic responds to the researcher's *revision*, not its opening statement.

The hard part is knowing when to stop. Fixed round counts either burn tokens on
agents agreeing with each other or cut off while positions are still moving, so
this module measures it: each round's combined output is embedded and compared
with the previous round's. Once cosine similarity crosses
``debate_convergence_threshold``, positions have stopped moving and the debate
ends early. That check costs nothing - the embedder is local and offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from athenacore.agents.base import AgentContext, AgentResult
from athenacore.agents.registry import AgentFactory
from athenacore.config import Settings
from athenacore.errors import RunCancelled
from athenacore.logging_setup import get_logger
from athenacore.memory.embeddings import Embedder, HashingEmbedder, cosine
from athenacore.memory.models import Entry, EntryKind, Run, RunStatus, Usage, utcnow
from athenacore.orchestration.events import CancellationToken, EventBus, EventType
from athenacore.orchestration.orchestrator import Orchestrator

log = get_logger(__name__)


@dataclass(slots=True)
class DebateRound:
    """One exchange between the participating agents."""

    index: int
    results: dict[str, AgentResult] = field(default_factory=dict)
    similarity_to_previous: float | None = None
    """Cosine similarity of this round's combined text to the last round's.
    ``None`` for the first round."""

    @property
    def text(self) -> str:
        return "\n\n".join(
            f"{name}: {result.content}" for name, result in self.results.items() if result.ok
        )

    @property
    def usage(self) -> Usage:
        total = Usage()
        for result in self.results.values():
            total = total + result.usage
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "similarity_to_previous": self.similarity_to_previous,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


@dataclass(slots=True)
class DebateReport:
    """The full debate: every round, why it stopped, and the verdict."""

    run: Run
    rounds: list[DebateRound] = field(default_factory=list)
    verdict: AgentResult | None = None
    converged: bool = False
    stop_reason: str = ""

    @property
    def usage(self) -> Usage:
        total = self.verdict.usage if self.verdict else Usage()
        for round_ in self.rounds:
            total = total + round_.usage
        return total

    def output(self) -> str:
        if self.verdict and self.verdict.ok:
            return self.verdict.content
        return self.rounds[-1].text if self.rounds else ""

    def transcript(self) -> str:
        blocks = []
        for round_ in self.rounds:
            header = f"# Round {round_.index}"
            if round_.similarity_to_previous is not None:
                header += f"  (similarity to previous: {round_.similarity_to_previous:.2f})"
            blocks.append(header)
            for name, result in round_.results.items():
                body = result.content if result.ok else f"[failed] {result.error}"
                blocks.append(f"## {name}\n\n{body}")
        if self.verdict:
            blocks.append(f"# Verdict\n\n{self.verdict.content}")
        return "\n\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "output": self.output(),
        }


class DebateOrchestrator:
    """Runs a structured, converging debate between agents."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        participants: Sequence[str] = ("research", "critic"),
        judge: str | None = "synthesizer",
        embedder: Embedder | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.store = orchestrator.store
        self.settings: Settings = orchestrator.settings
        self.bus: EventBus = orchestrator.bus
        self.participants = list(participants)
        self.judge = judge
        self.embedder = embedder or orchestrator.embedder or HashingEmbedder(dims=128)
        if len(self.participants) < 2:
            raise ValueError("a debate needs at least two participants")

    def run(
        self,
        *,
        topic: str,
        query: str,
        rounds: int | None = None,
        cancel: CancellationToken | None = None,
    ) -> DebateReport:
        max_rounds = rounds or self.settings.debate_rounds
        cancel = cancel or CancellationToken()
        self.store.ensure_topic(topic)

        run = Run(
            topic=topic,
            preset="debate",
            query=query,
            status=RunStatus.RUNNING,
            model=f"{self.orchestrator.provider.name}:{self.orchestrator.provider.model}",
            metadata={
                "participants": self.participants,
                "judge": self.judge,
                "max_rounds": max_rounds,
            },
        )
        self.store.save_run(run)
        report = DebateReport(run=run)

        self.bus.publish(
            EventType.RUN_STARTED,
            run_id=run.id,
            message=f"debate · {' vs '.join(self.participants)} · up to {max_rounds} rounds",
            topic=topic,
            participants=self.participants,
        )

        if query.strip():
            self.store.add_entry(
                Entry(
                    topic=topic,
                    agent="user",
                    kind=EntryKind.QUESTION,
                    content=query.strip(),
                    run_id=run.id,
                    salience=0.75,
                )
            )

        factory = AgentFactory(
            self.orchestrator.provider,
            self.store,
            settings=self.settings,
            retriever=self.orchestrator.retriever,
            tools=self.orchestrator._tools_for(topic),
        )

        try:
            self._debate(report, factory, topic, query, max_rounds, cancel)
            if self.judge:
                report.verdict = self._adjudicate(report, factory, topic, query)
            run.status = RunStatus.SUCCEEDED
        except RunCancelled:
            run.status = RunStatus.CANCELLED
            report.stop_reason = "cancelled"
            self.bus.publish(EventType.RUN_CANCELLED, run_id=run.id, message="cancelled")
        finally:
            run.ended_at = utcnow()
            run.usage = report.usage
            run.metadata["converged"] = report.converged
            run.metadata["stop_reason"] = report.stop_reason
            run.metadata["rounds_run"] = len(report.rounds)
            self.store.save_run(run)

        self.bus.publish(
            EventType.RUN_FINISHED,
            run_id=run.id,
            message=f"{len(report.rounds)} round(s) · {report.stop_reason}",
            status=run.status.value,
            usage=run.usage.to_dict(),
            converged=report.converged,
        )
        return report

    # -- internals -----------------------------------------------------------

    def _debate(
        self,
        report: DebateReport,
        factory: AgentFactory,
        topic: str,
        query: str,
        max_rounds: int,
        cancel: CancellationToken,
    ) -> None:
        previous_vector: list[float] | None = None
        report.stop_reason = f"completed {max_rounds} round(s)"

        for index in range(1, max_rounds + 1):
            cancel.raise_if_cancelled()
            self.bus.publish(
                EventType.ROUND_STARTED,
                run_id=report.run.id,
                message=f"round {index} of {max_rounds}",
                round=index,
            )
            round_ = DebateRound(index=index)

            # Sequential within a round, on purpose: each participant must see
            # what the previous one just said. Parallelism here would mean
            # everyone talking past each other.
            spoken: dict[str, str] = {}
            for name in self.participants:
                cancel.raise_if_cancelled()
                agent = factory.get(name)
                ctx = AgentContext(
                    topic=topic,
                    task=self._task_for(name, query, index, spoken),
                    run_id=report.run.id,
                    upstream=dict(spoken),
                    round_index=index,
                )
                self.bus.publish(
                    EventType.NODE_STARTED,
                    run_id=report.run.id,
                    node=name,
                    message=f"round {index}: {agent.role} responding…",
                    round=index,
                )
                result = agent.run(ctx)
                round_.results[name] = result
                if result.ok and result.content:
                    spoken[name] = result.content
                self.bus.publish(
                    EventType.NODE_FINISHED if result.ok else EventType.NODE_FAILED,
                    run_id=report.run.id,
                    node=name,
                    message=result.error
                    or f"round {index} done ({result.usage.total_tokens} tokens)",
                    round=index,
                    content=result.content,
                )

            vector = self.embedder.embed(round_.text)
            if previous_vector is not None:
                similarity = cosine(previous_vector, vector)
                round_.similarity_to_previous = similarity
                report.rounds.append(round_)
                if similarity >= self.settings.debate_convergence_threshold:
                    report.converged = True
                    report.stop_reason = (
                        f"converged after round {index} "
                        f"(similarity {similarity:.2f} ≥ {self.settings.debate_convergence_threshold})"
                    )
                    self.bus.publish(
                        EventType.CONVERGED,
                        run_id=report.run.id,
                        message=report.stop_reason,
                        similarity=similarity,
                        round=index,
                    )
                    return
            else:
                report.rounds.append(round_)
            previous_vector = vector

    def _task_for(self, name: str, query: str, index: int, spoken: dict[str, str]) -> str:
        """Frame each turn so participants respond rather than restate.

        Round one is an opening statement; later rounds explicitly demand
        movement - concede, sharpen or hold with new support - which is what stops
        a debate from becoming two monologues.
        """
        if index == 1:
            return query
        if not spoken:
            return (
                f"{query}\n\nThis is round {index}. Do not repeat your earlier position. "
                "Advance it: what has changed in light of the exchange so far?"
            )
        opponents = ", ".join(spoken)
        return (
            f"{query}\n\nThis is round {index}. {opponents} has just responded (above).\n"
            "You must do one of three things for each point at issue, and say which:\n"
            "- CONCEDE: they are right; state what you now believe instead.\n"
            "- SHARPEN: they are partly right; state the narrower claim that survives.\n"
            "- HOLD: they are wrong; give the specific evidence or reasoning they missed.\n"
            "Do not restate agreement you have already expressed."
        )

    def _adjudicate(
        self, report: DebateReport, factory: AgentFactory, topic: str, query: str
    ) -> AgentResult:
        judge = factory.get(self.judge or "synthesizer")
        transcript = "\n\n".join(f"[Round {r.index}] {r.text}" for r in report.rounds if r.text)
        ctx = AgentContext(
            topic=topic,
            task=(
                f"{query}\n\nThe debate above ran for {len(report.rounds)} round(s) and "
                f"{'converged' if report.converged else 'did not converge'}. "
                "Deliver the verdict: what is established, what remains genuinely open, "
                "and which side had the better of each disputed point."
            ),
            run_id=report.run.id,
            upstream={"debate": transcript},
        )
        self.bus.publish(
            EventType.NODE_STARTED,
            run_id=report.run.id,
            node=judge.name,
            message="adjudicating…",
        )
        result = judge.run(ctx)
        self.bus.publish(
            EventType.NODE_FINISHED if result.ok else EventType.NODE_FAILED,
            run_id=report.run.id,
            node=judge.name,
            message=result.error or "verdict delivered",
            content=result.content,
        )
        return result
