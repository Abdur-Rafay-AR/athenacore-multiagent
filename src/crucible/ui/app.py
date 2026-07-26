"""Streamlit control room.

Five views over one engine:

* **Console** - configure and launch a run, watch the live event trace, read each
  agent's output, and see the graph light up as nodes complete.
* **Memory** - search the topic's memory the way agents do, with the ranking
  breakdown shown per result, and adjust salience or archive entries by hand.
* **Timeline** - the append-only history of a topic, including archived entries.
* **Runs** - past runs with their status, cost and node states.
* **Analytics** - where the tokens went, which agents contribute, activity over
  time, and the memory compaction picture.

Run it with ``crucible ui`` (or ``streamlit run src/crucible/ui/app.py``).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import streamlit as st

from crucible.agents.registry import available_agents, describe_agents
from crucible.config import Settings
from crucible.errors import CrucibleError
from crucible.llm.registry import available_providers
from crucible.memory.embeddings import HashingEmbedder
from crucible.memory.models import EntryKind
from crucible.memory.retrieval import MemoryRetriever
from crucible.memory.sqlite_store import SqliteMemoryStore
from crucible.memory.store import EntryFilter
from crucible.orchestration.debate import DebateOrchestrator
from crucible.orchestration.events import EventQueue
from crucible.orchestration.orchestrator import Orchestrator
from crucible.orchestration.presets import PRESETS, build_preset
from crucible.ui import theme

st.set_page_config(
    page_title="Crucible",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- cached resources --------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_store(path: str) -> SqliteMemoryStore:
    """One store per database path, shared across reruns.

    ``cache_resource`` is essential here: Streamlit re-executes the whole script
    on every interaction, and opening a fresh SQLite connection each time would
    thrash the WAL and lose the connection-level pragmas.
    """
    return SqliteMemoryStore(path)


@st.cache_data(ttl=5, show_spinner=False)
def cached_stats(path: str, topic: str | None) -> dict[str, Any]:
    return get_store(path).stats(topic=topic)


@st.cache_data(ttl=5, show_spinner=False)
def cached_activity(path: str, topic: str | None) -> list[tuple[str, int]]:
    return get_store(path).activity_by_day(topic=topic, days=30)


def settings_from_state() -> Settings:
    base = Settings.from_env()
    return base.with_overrides(
        model=st.session_state.get("model", base.model),
        temperature=st.session_state.get("temperature", base.temperature),
        tools_enabled=st.session_state.get("tools_enabled", base.tools_enabled),
        web_search_enabled=st.session_state.get("web_search", base.web_search_enabled),
        max_parallel_agents=st.session_state.get("parallelism", base.max_parallel_agents),
        debate_rounds=st.session_state.get("rounds", base.debate_rounds),
    )


# -- sidebar -----------------------------------------------------------------


def render_sidebar(settings: Settings) -> tuple[str, str]:
    store = get_store(str(settings.database_path))

    st.sidebar.markdown("## ⚗️ Crucible")
    st.sidebar.caption("Multi-agent collaboration with persistent memory")

    topics = store.list_topics(limit=200)
    names = [t.name for t in topics]
    options = ["+ New topic", *names]
    choice = st.sidebar.selectbox("Topic", options, index=1 if names else 0)

    if choice == "+ New topic":
        topic = st.sidebar.text_input(
            "New topic name", placeholder="e.g. battery-supply-chain"
        ).strip()
    else:
        topic = choice
        record = next((t for t in topics if t.name == topic), None)
        if record:
            st.sidebar.caption(
                f"{record.entry_count} entries · {record.live_tokens} live tokens · "
                f"updated {record.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )

    st.sidebar.divider()
    st.sidebar.markdown("### Model")
    st.session_state.setdefault("model", settings.model)
    st.sidebar.text_input(
        "Model spec",
        key="model",
        help="provider:model - " + ", ".join(available_providers()),
    )
    st.sidebar.slider("Temperature", 0.0, 1.5, float(settings.temperature), 0.05, key="temperature")

    with st.sidebar.expander("Advanced", expanded=False):
        st.checkbox("Enable tools", value=settings.tools_enabled, key="tools_enabled")
        st.checkbox(
            "Enable web search",
            value=settings.web_search_enabled,
            key="web_search",
            help="Requires: pip install 'crucible-agents[search]'",
        )
        st.slider("Max parallel agents", 1, 8, settings.max_parallel_agents, key="parallelism")
        st.caption(f"Database: `{settings.database_path}`")
        stats = cached_stats(str(settings.database_path), None)
        st.caption(f"Full-text index: {'enabled' if stats.get('fts') else 'unavailable'}")

    view = st.sidebar.radio(
        "View",
        ["Console", "Memory", "Timeline", "Runs", "Analytics"],
        label_visibility="collapsed",
    )
    return topic, view


# -- console -----------------------------------------------------------------


def run_with_live_trace(orchestrator: Orchestrator, graph, topic: str, query: str) -> Any:
    """Execute a run on a worker thread, painting events as they arrive.

    Streamlit has no event loop of its own, so the run goes to a thread and this
    function polls the event queue, redrawing the trace and the graph. That is
    what turns a 90-second wait into something you can watch.
    """
    events = EventQueue(orchestrator.bus)
    holder: dict[str, Any] = {}

    def work() -> None:
        try:
            holder["report"] = orchestrator.run(graph, topic=topic, query=query)
        except Exception as exc:  # shown in the UI rather than swallowed
            holder["error"] = exc
        finally:
            events.close()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()

    trace_box = st.empty()
    graph_box = st.empty()
    progress = st.progress(0.0, text="starting…")

    seen: list[dict[str, Any]] = []
    states: dict[str, str] = {node.name: "pending" for node in graph}
    total = len(graph)

    while worker.is_alive() or not events.empty:
        drained = events.drain()
        if not drained:
            time.sleep(0.15)
            continue
        for event in drained:
            data = event.to_dict()
            seen.append(data)
            node = data.get("node")
            if node:
                mapping = {
                    "node.started": "running",
                    "node.finished": "succeeded",
                    "node.failed": "failed",
                    "node.skipped": "skipped",
                }
                if data["type"] in mapping:
                    states[node] = mapping[data["type"]]
        done = sum(1 for s in states.values() if s in {"succeeded", "failed", "skipped"})
        progress.progress(min(1.0, done / max(1, total)), text=f"{done}/{total} agents done")
        trace_box.markdown(
            '<div class="ac-trace">' + "".join(theme.trace_row(e) for e in seen[-60:]) + "</div>",
            unsafe_allow_html=True,
        )
        graph_box.markdown(f"```mermaid\n{graph.to_mermaid(states=states)}\n```")

    worker.join(timeout=10)
    progress.empty()
    if "error" in holder:
        st.error(f"Run failed: {holder['error']}")
        return None
    return holder.get("report")


def view_console(settings: Settings, topic: str) -> None:
    store = get_store(str(settings.database_path))
    st.markdown("### Console")

    if not topic:
        st.markdown(
            theme.empty_state(
                "Name a topic in the sidebar to begin.",
                "A topic is a long-lived memory thread - every run against it builds on the last.",
            ),
            unsafe_allow_html=True,
        )
        return

    mode = st.radio("Mode", ["Graph run", "Debate"], horizontal=True, label_visibility="collapsed")

    if mode == "Graph run":
        col_left, col_right = st.columns([3, 2])
        with col_left:
            preset_key = st.selectbox(
                "Workflow",
                [*PRESETS.keys(), *[f"solo:{a}" for a in available_agents()]],
                format_func=lambda k: (
                    PRESETS[k].title if k in PRESETS else f"Single agent: {k[5:]}"
                ),
            )
        with col_right:
            if preset_key in PRESETS:
                preset = PRESETS[preset_key]
                st.caption(
                    f"**{preset.title}** - {preset.summary}  \nToken appetite: {preset.cost}"
                )

        query = st.text_area(
            "Question",
            placeholder="What do you want the team to work on?",
            height=100,
        )

        try:
            graph = build_preset(preset_key, settings=settings)
        except CrucibleError as exc:
            st.error(str(exc))
            return

        with st.expander("Workflow graph", expanded=False):
            st.markdown(f"```mermaid\n{graph.to_mermaid()}\n```")
            st.caption(
                f"{len(graph)} agents · {len(graph.levels())} stages · "
                f"up to {graph.max_width} running in parallel"
            )

        if st.button("▶ Run", type="primary", use_container_width=True, disabled=not query.strip()):
            orchestrator = Orchestrator(store, settings=settings)
            report = run_with_live_trace(orchestrator, graph, topic, query.strip())
            if report is not None:
                render_report(report)
                cached_stats.clear()

    else:
        query = st.text_area(
            "Debate question", height=100, placeholder="A question with two defensible sides."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            participants = st.multiselect(
                "Participants", available_agents(), default=["research", "critic"]
            )
        with col_b:
            judge = st.selectbox("Judge", ["synthesizer", *available_agents(), "(none)"])
        with col_c:
            rounds = st.slider("Max rounds", 1, 8, settings.debate_rounds, key="rounds")

        st.caption(
            "Rounds stop early once positions stop moving - measured by cosine similarity "
            f"between consecutive rounds (threshold {settings.debate_convergence_threshold})."
        )

        can_run = bool(query.strip()) and len(participants) >= 2
        if st.button(
            "▶ Start debate", type="primary", use_container_width=True, disabled=not can_run
        ):
            orchestrator = Orchestrator(store, settings=settings)
            debate = DebateOrchestrator(
                orchestrator,
                participants=participants,
                judge=None if judge == "(none)" else judge,
            )
            with st.spinner(f"Debating (up to {rounds} rounds)…"):
                report = debate.run(topic=topic, query=query.strip(), rounds=rounds)
            render_debate(report)
            cached_stats.clear()
        elif not can_run and query.strip():
            st.info("A debate needs at least two participants.")


def render_report(report: Any) -> None:
    usage = report.usage
    st.markdown(
        theme.stat_tiles(
            [
                ("status", report.status.value),
                ("agents", len(report.results)),
                ("tokens", usage.total_tokens),
                ("model calls", usage.calls),
                ("duration", f"{report.run.duration_ms} ms"),
            ]
            + ([("cost", f"${usage.cost_usd:.4f}")] if usage.cost_usd else [])
        ),
        unsafe_allow_html=True,
    )

    if report.failures:
        for name, error in report.failures.items():
            st.warning(f"**{name}** failed: {error}")
    if report.skipped:
        st.info(f"Skipped: {', '.join(report.skipped)}")
    if report.compaction and report.compaction.performed:
        st.success(
            f"🗜️ Memory compacted - {report.compaction.reason} "
            f"({report.compaction.tokens_before} → {report.compaction.tokens_after} tokens)"
        )

    names = [n for n, r in report.results.items() if r.ok and r.content]
    if not names:
        st.warning("No agent produced output.")
        return

    tabs = st.tabs([*names, "Transcript"])
    # Not strict: `tabs` has one extra entry for the Transcript tab, filled below.
    for tab, name in zip(tabs, names, strict=False):
        result = report.results[name]
        with tab:
            st.markdown(result.content)
            bits = [f"{result.usage.total_tokens} tokens", f"{result.duration_ms} ms"]
            if result.recall:
                bits.append(f"recalled {len(result.recall)} entries")
            if result.tool_results:
                bits.append(f"tools: {', '.join(t.call.name for t in result.tool_results)}")
            if result.citations:
                bits.append(f"cited {len(result.citations)} memories")
            st.caption(" · ".join(bits))

            if result.recall and result.recall.entries:
                with st.expander(f"What {name} recalled", expanded=False):
                    for item in result.recall.entries:
                        st.markdown(
                            theme.entry_card(
                                item.entry.to_dict(),
                                signals={
                                    "kw": item.keyword,
                                    "sem": item.semantic,
                                    "rec": item.recency,
                                    "sal": item.salience,
                                },
                                score=item.score,
                            ),
                            unsafe_allow_html=True,
                        )
    with tabs[-1]:
        st.code(report.transcript(), language="markdown")


def render_debate(report: Any) -> None:
    st.markdown(
        theme.stat_tiles(
            [
                ("rounds", len(report.rounds)),
                ("converged", "yes" if report.converged else "no"),
                ("tokens", report.usage.total_tokens),
                ("calls", report.usage.calls),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption(report.stop_reason)

    if report.verdict and report.verdict.ok:
        st.markdown("#### ⚖️ Verdict")
        st.markdown(report.verdict.content)
        st.divider()

    for round_ in report.rounds:
        header = f"Round {round_.index}"
        if round_.similarity_to_previous is not None:
            header += f" · similarity to previous {round_.similarity_to_previous:.2f}"
        with st.expander(header, expanded=round_.index == len(report.rounds)):
            for name, result in round_.results.items():
                st.markdown(f"**{name}**")
                st.markdown(result.content if result.ok else f"_failed: {result.error}_")


# -- memory ------------------------------------------------------------------


def view_memory(settings: Settings, topic: str) -> None:
    store = get_store(str(settings.database_path))
    st.markdown("### Memory")
    if not topic:
        st.markdown(theme.empty_state("Select a topic first."), unsafe_allow_html=True)
        return

    st.caption(
        "This runs the same hybrid recall the agents use: BM25 + embedding similarity + "
        "recency decay + salience, de-duplicated with MMR. Every result shows why it ranked."
    )

    col_query, col_limit = st.columns([4, 1])
    with col_query:
        query = st.text_input("Search memory", placeholder="e.g. water usage in brine extraction")
    with col_limit:
        limit = st.number_input("Results", 1, 50, 10)

    with st.expander("Ranking weights", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        keyword = c1.slider("Keyword", 0.0, 2.0, settings.retrieval.keyword_weight, 0.1)
        semantic = c2.slider("Semantic", 0.0, 2.0, settings.retrieval.semantic_weight, 0.1)
        recency = c3.slider("Recency", 0.0, 2.0, settings.retrieval.recency_weight, 0.1)
        salience = c4.slider("Salience", 0.0, 2.0, settings.retrieval.salience_weight, 0.1)
        mmr = st.slider(
            "MMR λ (1 = pure relevance, 0 = maximum diversity)",
            0.0,
            1.0,
            settings.retrieval.mmr_lambda,
            0.05,
        )
        include_archived = st.checkbox("Include archived entries", value=False)

    retrieval = settings.retrieval.__class__(
        max_entries=int(limit),
        candidate_pool=max(60, int(limit) * 4),
        keyword_weight=keyword,
        semantic_weight=semantic,
        recency_weight=recency,
        salience_weight=salience,
        mmr_lambda=mmr,
        include_archived=include_archived,
    )
    retriever = MemoryRetriever(
        store,
        settings=retrieval,
        embedder=HashingEmbedder(dims=settings.embedding_dims)
        if settings.embeddings_enabled
        else None,
        token_budget=settings.context_token_budget,
    )
    result = retriever.recall(topic, query, max_entries=int(limit))

    if not result:
        st.markdown(
            theme.empty_state(
                "Nothing recalled yet.",
                "Run an agent on this topic to give it something to remember.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.caption(
        f"{len(result)} of {result.candidates_considered} candidates · ~{result.tokens} tokens"
        + (" · truncated by the token budget" if result.truncated else "")
    )

    for item in result.entries:
        entry = item.entry
        st.markdown(
            theme.entry_card(
                entry.to_dict(),
                signals={
                    "kw": item.keyword,
                    "sem": item.semantic,
                    "rec": item.recency,
                    "sal": item.salience,
                },
                score=item.score,
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Adjust", expanded=False):
            c1, c2 = st.columns([3, 1])
            new_salience = c1.slider(
                "Salience", 0.0, 1.0, float(entry.salience), 0.05, key=f"sal-{entry.id}"
            )
            if c1.button("Save salience", key=f"save-{entry.id}"):
                store.set_salience(entry.id, new_salience)
                st.success("Updated - this changes how strongly it surfaces in future recalls.")
            if c2.button("Archive", key=f"arch-{entry.id}"):
                store.archive_entries([entry.id])
                st.success("Archived. It stays in the history but is excluded from recall.")


# -- timeline ----------------------------------------------------------------


def view_timeline(settings: Settings, topic: str) -> None:
    store = get_store(str(settings.database_path))
    st.markdown("### Timeline")
    if not topic:
        st.markdown(theme.empty_state("Select a topic first."), unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    kinds = c1.multiselect("Kinds", [k.value for k in EntryKind], default=[])
    agents = c2.multiselect(
        "Agents", [a["name"] for a in describe_agents()] + ["user", "compactor"]
    )
    show_archived = c3.checkbox("Archived", value=True)

    entries = store.query_entries(
        EntryFilter(
            topic=topic,
            kinds=[EntryKind(k) for k in kinds] or None,
            agents=agents or None,
            include_archived=show_archived,
            limit=300,
            newest_first=False,
        )
    )
    if not entries:
        st.markdown(theme.empty_state("No entries match these filters."), unsafe_allow_html=True)
        return

    st.caption(f"{len(entries)} entries, oldest first - the full audit trail for this topic.")
    for entry in entries:
        st.markdown(theme.entry_card(entry.to_dict()), unsafe_allow_html=True)


# -- runs --------------------------------------------------------------------


def view_runs(settings: Settings, topic: str) -> None:
    store = get_store(str(settings.database_path))
    st.markdown("### Runs")
    runs = store.list_runs(topic=topic or None, limit=60)
    if not runs:
        st.markdown(theme.empty_state("No runs recorded yet."), unsafe_allow_html=True)
        return

    for run in runs:
        icon = {"succeeded": "✅", "partial": "⚠️", "failed": "❌", "cancelled": "⏹️"}.get(
            run.status.value, "•"
        )
        label = (
            f"{icon} {run.started_at.strftime('%Y-%m-%d %H:%M')} · {run.preset} · "
            f"{run.usage.total_tokens} tokens"
        )
        with st.expander(label, expanded=False):
            if run.query:
                st.markdown(f"**Question:** {run.query}")
            st.markdown(
                theme.stat_tiles(
                    [
                        ("status", run.status.value),
                        ("duration", f"{run.duration_ms} ms"),
                        ("calls", run.usage.calls),
                        ("tokens", run.usage.total_tokens),
                    ]
                ),
                unsafe_allow_html=True,
            )
            if run.node_states:
                st.write(
                    " ".join(
                        f":{'green' if v == 'succeeded' else 'red' if v == 'failed' else 'gray'}[{k}]"
                        for k, v in run.node_states.items()
                    )
                )
            if run.error:
                st.error(run.error)
            entries = store.query_entries(
                EntryFilter(run_id=run.id, limit=50, newest_first=False, include_archived=True)
            )
            for entry in entries:
                st.markdown(theme.entry_card(entry.to_dict()), unsafe_allow_html=True)


# -- analytics ---------------------------------------------------------------


def view_analytics(settings: Settings, topic: str) -> None:
    path = str(settings.database_path)
    st.markdown("### Analytics")
    stats = cached_stats(path, topic or None)

    st.markdown(
        theme.stat_tiles(
            [
                ("topics", stats["topics"]),
                ("entries", stats["entries"]),
                ("archived", stats["archived"]),
                ("memory tokens", f"{stats['tokens']:,}"),
                ("runs", stats["runs"]),
                ("model calls", stats["usage"]["calls"]),
            ]
        ),
        unsafe_allow_html=True,
    )
    if stats["usage"]["cost_usd"]:
        st.caption(f"Estimated spend: ${stats['usage']['cost_usd']:.4f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Contributions by agent")
        by_agent = stats["by_agent"]
        if by_agent:
            top = max(by_agent.values())
            html_rows = "".join(
                theme.bar_row(agent, count, top, theme.kind_colour("research"))
                for agent, count in list(by_agent.items())[:12]
            )
            st.markdown(html_rows, unsafe_allow_html=True)
        else:
            st.caption("No entries yet.")

    with col_right:
        st.markdown("#### Entry kinds")
        by_kind = stats["by_kind"]
        if by_kind:
            top = max(by_kind.values())
            html_rows = "".join(
                theme.bar_row(kind, count, top, theme.kind_colour(kind))
                for kind, count in by_kind.items()
            )
            st.markdown(html_rows, unsafe_allow_html=True)
        else:
            st.caption("No entries yet.")

    st.markdown("#### Activity")
    activity = cached_activity(path, topic or None)
    st.markdown(theme.activity_chart(activity), unsafe_allow_html=True)

    if topic:
        store = get_store(path)
        record = store.get_topic(topic)
        if record:
            st.markdown("#### Compaction pressure")
            ratio = min(1.0, record.live_tokens / max(1, settings.compaction_threshold_tokens))
            st.progress(ratio)
            st.caption(
                f"{record.live_tokens:,} live tokens of a {settings.compaction_threshold_tokens:,} "
                "token budget. At 100%, older entries are folded into a durable summary "
                "and archived (never deleted)."
            )


# -- main --------------------------------------------------------------------


def main() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    base = Settings.from_env()
    base.ensure_dirs()
    topic, view = render_sidebar(base)
    settings = settings_from_state()

    try:
        if view == "Console":
            view_console(settings, topic)
        elif view == "Memory":
            view_memory(settings, topic)
        elif view == "Timeline":
            view_timeline(settings, topic)
        elif view == "Runs":
            view_runs(settings, topic)
        else:
            view_analytics(settings, topic)
    except CrucibleError as exc:
        st.error(str(exc))


main()
