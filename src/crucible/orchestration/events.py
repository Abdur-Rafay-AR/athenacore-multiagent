"""Run events.

The orchestrator emits events rather than printing, which is what lets one
engine drive a CLI progress display, an SSE stream to a browser and a Streamlit
timeline without knowing any of them exist.

The bus is deliberately synchronous and non-throwing: a subscriber that raises
must never take down the run that is feeding it.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from crucible.logging_setup import get_logger
from crucible.memory.models import to_iso, utcnow

log = get_logger(__name__)


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    NODE_STARTED = "node.started"
    NODE_FINISHED = "node.finished"
    NODE_FAILED = "node.failed"
    NODE_SKIPPED = "node.skipped"
    TOKEN = "token"
    TOOL_CALLED = "tool.called"
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_WRITTEN = "memory.written"
    COMPACTED = "memory.compacted"
    ROUND_STARTED = "round.started"
    CONVERGED = "debate.converged"
    LOG = "log"


@dataclass(slots=True)
class Event:
    """One thing that happened during a run."""

    type: EventType
    run_id: str = ""
    node: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: to_iso(utcnow()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "node": self.node,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe fan-out of events to subscribers.

    Also keeps a bounded history so a client that connects mid-run (a browser
    reloading the UI, say) can replay what it missed.
    """

    def __init__(self, *, history_limit: int = 1000) -> None:
        self._subscribers: list[Subscriber] = []
        self._lock = threading.RLock()
        self._history: list[Event] = []
        self._history_limit = history_limit

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register a callback. Returns a function that unsubscribes it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def emit(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception as exc:  # never let a listener break the run
                log.warning("event subscriber raised", extra={"error": str(exc)})

    def publish(
        self,
        type: EventType,
        *,
        run_id: str = "",
        node: str | None = None,
        message: str = "",
        **data: Any,
    ) -> Event:
        """Convenience constructor + emit."""
        event = Event(type=type, run_id=run_id, node=node, message=message, data=data)
        self.emit(event)
        return event

    def history(self, *, since: int = 0) -> list[Event]:
        with self._lock:
            return list(self._history[since:])

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


class EventQueue:
    """Adapts the push-based bus into a pull-based iterator.

    This is what the SSE endpoint and the Streamlit console consume: they need to
    pull events on their own schedule while the run pushes from a worker thread.
    """

    def __init__(self, bus: EventBus, *, maxsize: int = 2000) -> None:
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=maxsize)
        self._unsubscribe = bus.subscribe(self._on_event)
        self._closed = False

    def _on_event(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop rather than block: a slow consumer must not stall the run.
            log.debug("event queue full, dropping event", extra={"type": event.type.value})

    def __iter__(self) -> Iterator[Event]:
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item
            if item.type in {
                EventType.RUN_FINISHED,
                EventType.RUN_FAILED,
                EventType.RUN_CANCELLED,
            }:
                return

    @property
    def empty(self) -> bool:
        """Whether nothing is currently buffered. Used by pollers such as the UI."""
        return self._queue.empty()

    def drain(self) -> list[Event]:
        """Everything currently buffered, without blocking."""
        out: list[Event] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return out
            if item is not None:
                out.append(item)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._unsubscribe()
            with contextlib.suppress(queue.Full):  # pragma: no cover
                self._queue.put_nowait(None)

    def __enter__(self) -> EventQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CancellationToken:
    """Cooperative cancellation, checked between graph nodes.

    Cooperative rather than pre-emptive because killing a thread mid-HTTP-call
    would leak connections and could write a half-formed entry to memory.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        from crucible.errors import RunCancelled

        if self._event.is_set():
            raise RunCancelled("run was cancelled")


def console_printer(*, verbose: bool = False) -> Subscriber:
    """A subscriber that prints a readable live trace to stdout."""
    symbols = {
        EventType.RUN_STARTED: "▶",
        EventType.RUN_FINISHED: "■",
        EventType.RUN_FAILED: "✖",
        EventType.RUN_CANCELLED: "◼",
        EventType.NODE_STARTED: "·",
        EventType.NODE_FINISHED: "✓",
        EventType.NODE_FAILED: "✖",
        EventType.NODE_SKIPPED: "→",
        EventType.ROUND_STARTED: "◆",
        EventType.CONVERGED: "≈",
        EventType.COMPACTED: "▽",
        EventType.TOOL_CALLED: "🔧",
    }
    quiet = {EventType.TOKEN, EventType.MEMORY_RECALLED, EventType.LOG}

    def printer(event: Event) -> None:
        if event.type in quiet and not verbose:
            return
        symbol = symbols.get(event.type, "•")
        label = f"{event.node}: " if event.node else ""
        print(f"  {symbol} {label}{event.message}")

    return printer
