"""HTTP API.

A thin FastAPI layer over the engine: no business logic lives here, it only maps
requests onto the same objects the CLI uses. The interesting endpoint is
``POST /runs/stream``, which runs a graph in a worker thread and forwards the
event bus to the client as Server-Sent Events — so a browser sees each agent
start, recall memory, call tools and finish, live.

Install with ``pip install 'athenacore[api]'`` and start with ``athenacore serve``.
Interactive docs are at ``/docs``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

from athenacore.config import Settings
from athenacore.errors import AthenaCoreError, ConfigurationError
from athenacore.logging_setup import get_logger
from athenacore.memory.sqlite_store import SqliteMemoryStore

log = get_logger(__name__)

try:
    from fastapi import Depends, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional extra
    raise ImportError(
        "The API needs extra packages. Install with: pip install 'athenacore[api]'"
    ) from exc


# -- schemas -----------------------------------------------------------------


class RunRequest(BaseModel):
    topic: str = Field(..., description="Memory topic; shared across runs.")
    query: str = Field("", description="The question to work on.")
    preset: str = Field("research-critique-synthesis", description="Preset key or solo:<agent>.")
    model: str | None = Field(None, description="Override the model, e.g. ollama:llama3.1")


class DebateRequest(BaseModel):
    topic: str
    query: str
    rounds: int | None = None
    participants: list[str] = Field(default_factory=lambda: ["research", "critic"])
    judge: str | None = "synthesizer"
    model: str | None = None


class EntryPatch(BaseModel):
    salience: float | None = Field(None, ge=0.0, le=1.0)
    tags: list[str] | None = None


class TopicRequest(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


# -- app ---------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Accepts settings so tests can inject an isolated store."""
    settings = settings or Settings.from_env()
    settings.ensure_dirs()

    app = FastAPI(
        title="AthenaCore",
        version="0.2.0",
        description=(
            "Multi-agent collaboration with persistent, searchable memory. "
            "Agents share one memory per topic; runs are graphs of agent invocations."
        ),
    )
    # Wide-open CORS is right for a local-first tool people run on their own
    # machine. Tighten it before exposing this to a network.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings

    def get_store() -> Iterator[SqliteMemoryStore]:
        store = SqliteMemoryStore(settings.database_path)
        try:
            yield store
        finally:
            store.close()

    def _orchestrator(store: SqliteMemoryStore, model: str | None = None):
        from athenacore.orchestration.orchestrator import Orchestrator

        active = settings.with_overrides(model=model) if model else settings
        return Orchestrator(store, settings=active)

    # -- meta ---------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, Any]:
        store = SqliteMemoryStore(settings.database_path)
        try:
            stats = store.stats()
        finally:
            store.close()
        return {
            "status": "ok",
            "version": "0.2.0",
            "model": settings.model,
            "fts": stats.get("fts", False),
            "entries": stats["entries"],
        }

    @app.get("/config", tags=["meta"])
    def config() -> dict[str, Any]:
        """Effective settings, with secrets redacted."""
        return settings.redacted()

    @app.get("/agents", tags=["meta"])
    def agents() -> list[dict[str, Any]]:
        from athenacore.agents.registry import describe_agents

        return describe_agents()

    @app.get("/presets", tags=["meta"])
    def presets() -> list[dict[str, Any]]:
        from athenacore.orchestration.presets import describe_presets

        return describe_presets()

    # -- topics -------------------------------------------------------------

    @app.get("/topics", tags=["memory"])
    def list_topics(
        limit: int = Query(50, ge=1, le=500),
        search: str | None = None,
        store: SqliteMemoryStore = Depends(get_store),
    ) -> list[dict[str, Any]]:
        return [t.to_dict() for t in store.list_topics(limit=limit, search=search)]

    @app.post("/topics", tags=["memory"], status_code=201)
    def create_topic(
        payload: TopicRequest, store: SqliteMemoryStore = Depends(get_store)
    ) -> dict[str, Any]:
        return store.ensure_topic(
            payload.name, description=payload.description, tags=payload.tags
        ).to_dict()

    @app.get("/topics/{name}", tags=["memory"])
    def get_topic(name: str, store: SqliteMemoryStore = Depends(get_store)) -> dict[str, Any]:
        topic = store.get_topic(name)
        if topic is None:
            raise HTTPException(404, f"topic {name!r} not found")
        return {"topic": topic.to_dict(), "stats": store.stats(topic=name)}

    @app.delete("/topics/{name}", tags=["memory"])
    def delete_topic(name: str, store: SqliteMemoryStore = Depends(get_store)) -> dict[str, Any]:
        if store.get_topic(name) is None:
            raise HTTPException(404, f"topic {name!r} not found")
        return {"deleted": store.delete_topic(name)}

    @app.get("/topics/{name}/timeline", tags=["memory"])
    def timeline(
        name: str,
        limit: int = Query(100, ge=1, le=1000),
        include_archived: bool = False,
        store: SqliteMemoryStore = Depends(get_store),
    ) -> list[dict[str, Any]]:
        entries = store.timeline(name, limit=limit, include_archived=include_archived)
        return [e.to_dict() for e in entries]

    # -- memory -------------------------------------------------------------

    @app.get("/memory/recall", tags=["memory"])
    def recall(
        topic: str,
        query: str = "",
        limit: int = Query(10, ge=1, le=50),
        store: SqliteMemoryStore = Depends(get_store),
    ) -> dict[str, Any]:
        """Run the same hybrid recall the agents use, with score breakdowns."""
        from athenacore.memory.embeddings import HashingEmbedder
        from athenacore.memory.retrieval import MemoryRetriever

        embedder = (
            HashingEmbedder(dims=settings.embedding_dims) if settings.embeddings_enabled else None
        )
        retriever = MemoryRetriever(
            store,
            settings=settings.retrieval,
            embedder=embedder,
            token_budget=settings.context_token_budget,
        )
        return retriever.recall(topic, query, max_entries=limit).to_dict()

    @app.get("/memory/search", tags=["memory"])
    def search(
        query: str,
        topic: str | None = None,
        limit: int = Query(20, ge=1, le=100),
        store: SqliteMemoryStore = Depends(get_store),
    ) -> list[dict[str, Any]]:
        """Raw full-text search, without the ranking blend."""
        return [
            {"entry": entry.to_dict(), "relevance": round(score, 4)}
            for entry, score in store.keyword_search(query, topic=topic, limit=limit)
        ]

    @app.patch("/memory/entries/{entry_id}", tags=["memory"])
    def patch_entry(
        entry_id: str, payload: EntryPatch, store: SqliteMemoryStore = Depends(get_store)
    ) -> dict[str, Any]:
        entry = store.get_entry(entry_id)
        if entry is None:
            raise HTTPException(404, f"entry {entry_id!r} not found")
        if payload.salience is not None:
            store.set_salience(entry_id, payload.salience)
        if payload.tags:
            store.add_tags(entry_id, payload.tags)
        updated = store.get_entry(entry_id)
        assert updated is not None
        return updated.to_dict()

    @app.post("/memory/entries/{entry_id}/archive", tags=["memory"])
    def archive_entry(
        entry_id: str, store: SqliteMemoryStore = Depends(get_store)
    ) -> dict[str, Any]:
        if store.get_entry(entry_id) is None:
            raise HTTPException(404, f"entry {entry_id!r} not found")
        return {"archived": store.archive_entries([entry_id])}

    # -- runs ---------------------------------------------------------------

    @app.get("/runs", tags=["runs"])
    def list_runs(
        topic: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        store: SqliteMemoryStore = Depends(get_store),
    ) -> list[dict[str, Any]]:
        return [r.to_dict() for r in store.list_runs(topic=topic, limit=limit)]

    @app.get("/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str, store: SqliteMemoryStore = Depends(get_store)) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id!r} not found")
        from athenacore.memory.store import EntryFilter

        entries = store.query_entries(EntryFilter(run_id=run_id, limit=200, newest_first=False))
        return {"run": run.to_dict(), "entries": [e.to_dict() for e in entries]}

    @app.post("/runs", tags=["runs"])
    def create_run(
        payload: RunRequest, store: SqliteMemoryStore = Depends(get_store)
    ) -> dict[str, Any]:
        """Run a graph and return the whole report once it finishes."""
        from athenacore.orchestration.presets import build_preset

        try:
            orchestrator = _orchestrator(store, payload.model)
            graph = build_preset(payload.preset, settings=orchestrator.settings)
        except (ConfigurationError, AthenaCoreError) as exc:
            raise HTTPException(400, str(exc)) from exc
        report = orchestrator.run(graph, topic=payload.topic, query=payload.query)
        return report.to_dict()

    @app.post("/runs/stream", tags=["runs"])
    def stream_run(payload: RunRequest, store: SqliteMemoryStore = Depends(get_store)):
        """Run a graph, streaming progress as Server-Sent Events.

        The graph executes on a worker thread while this generator drains the
        event bus, so the client sees each node start and finish rather than
        waiting on one long request.
        """
        from athenacore.orchestration.events import EventQueue
        from athenacore.orchestration.presets import build_preset

        try:
            orchestrator = _orchestrator(store, payload.model)
            graph = build_preset(payload.preset, settings=orchestrator.settings)
        except (ConfigurationError, AthenaCoreError) as exc:
            raise HTTPException(400, str(exc)) from exc

        queue = EventQueue(orchestrator.bus)
        holder: dict[str, Any] = {}

        def work() -> None:
            try:
                holder["report"] = orchestrator.run(graph, topic=payload.topic, query=payload.query)
            except Exception as exc:  # surfaced to the client as an error event
                holder["error"] = str(exc)
                log.exception("streamed run failed")
            finally:
                queue.close()

        worker = threading.Thread(target=work, name="athena-api-run", daemon=True)
        worker.start()

        def events() -> Iterator[str]:
            try:
                for event in queue:
                    yield f"event: {event.type.value}\ndata: {json.dumps(event.to_dict(), default=str)}\n\n"
                worker.join(timeout=5)
                if "error" in holder:
                    yield f"event: error\ndata: {json.dumps({'message': holder['error']})}\n\n"
                elif "report" in holder:
                    payload_json = json.dumps(holder["report"].to_dict(), default=str)
                    yield f"event: report\ndata: {payload_json}\n\n"
            finally:
                queue.close()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/debates", tags=["runs"])
    def create_debate(
        payload: DebateRequest, store: SqliteMemoryStore = Depends(get_store)
    ) -> dict[str, Any]:
        from athenacore.orchestration.debate import DebateOrchestrator

        try:
            debate = DebateOrchestrator(
                _orchestrator(store, payload.model),
                participants=payload.participants,
                judge=payload.judge,
            )
        except (ValueError, AthenaCoreError) as exc:
            raise HTTPException(400, str(exc)) from exc
        report = debate.run(topic=payload.topic, query=payload.query, rounds=payload.rounds)
        return report.to_dict()

    # -- analytics ----------------------------------------------------------

    @app.get("/stats", tags=["meta"])
    def stats(
        topic: str | None = None, store: SqliteMemoryStore = Depends(get_store)
    ) -> dict[str, Any]:
        payload = store.stats(topic=topic)
        payload["activity"] = store.activity_by_day(topic=topic, days=30)
        return payload

    return app


app = None  # populated lazily by `athenacore serve`


def main() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
