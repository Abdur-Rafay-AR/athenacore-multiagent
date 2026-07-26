"""Command-line interface.

Built on ``argparse`` rather than Typer/Click so the CLI works from a bare
interpreter with nothing installed - which is the same reason the core engine has
no dependencies.

    athenacore run "question" --topic energy --preset red-team
    athenacore debate "question" --topic energy --rounds 4
    athenacore recall "water usage" --topic energy --explain
    athenacore topics / timeline / runs / stats / export
    athenacore agents / presets / graph / doctor / serve / ui
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from athenacore import __version__
from athenacore.config import Settings
from athenacore.errors import AthenaCoreError
from athenacore.logging_setup import configure_logging

# Windows terminals default to a legacy code page that cannot encode the symbols
# used in the live trace. Force UTF-8 before anything prints.
if hasattr(sys.stdout, "reconfigure"):  # pragma: no cover - platform dependent
    try:
        cast("Any", sys.stdout).reconfigure(encoding="utf-8", errors="replace")
        cast("Any", sys.stderr).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
CYAN, GREEN, YELLOW, RED = "\033[36m", "\033[32m", "\033[33m", "\033[31m"


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def _heading(text: str) -> str:
    return _colour(text, BOLD + CYAN)


# -- shared plumbing ---------------------------------------------------------


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "database", None):
        overrides["database_path"] = Path(args.database)
    if getattr(args, "temperature", None) is not None:
        overrides["temperature"] = args.temperature
    if getattr(args, "no_tools", False):
        overrides["tools_enabled"] = False
    if getattr(args, "web_search", False):
        overrides["web_search_enabled"] = True
    if getattr(args, "log_level", None):
        overrides["log_level"] = args.log_level

    settings = Settings.from_env(**overrides)
    settings.ensure_dirs()
    configure_logging(settings.log_level, json_output=settings.log_json)
    return settings


def _open_store(settings: Settings):
    from athenacore.memory.sqlite_store import SqliteMemoryStore

    return SqliteMemoryStore(settings.database_path)


def _build_orchestrator(settings: Settings, *, quiet: bool, verbose: bool):
    from athenacore.orchestration.events import console_printer
    from athenacore.orchestration.orchestrator import Orchestrator

    store = _open_store(settings)
    orchestrator = Orchestrator(store, settings=settings)
    if not quiet:
        orchestrator.bus.subscribe(console_printer(verbose=verbose))
    return orchestrator, store


def _emit(payload: Any, *, as_json: bool, renderer=None) -> None:
    """Print either machine-readable JSON or the human rendering."""
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    elif renderer is not None:
        renderer(payload)


# -- commands ----------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    from athenacore.orchestration.presets import build_preset

    orchestrator, store = _build_orchestrator(settings, quiet=args.json, verbose=args.verbose)
    try:
        graph = build_preset(args.preset, settings=settings)
        if not args.json:
            print(_heading(f"\n{graph.name}  ·  topic {args.topic!r}  ·  {settings.model}\n"))
        report = orchestrator.run(graph, topic=args.topic, query=args.query)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0 if report.ok else 1

        print()
        for name, result in report.results.items():
            if result.ok and result.content:
                print(_heading(f"── {name} ({result.kind.value}) ──"))
                print(result.content)
                print()
        if report.failures:
            for name, error in report.failures.items():
                print(_colour(f"  ✖ {name}: {error}", RED))
        if report.compaction and report.compaction.performed:
            print(_colour(f"  ▽ memory compacted: {report.compaction.reason}", DIM))
        usage = report.usage
        print(
            _colour(
                f"\n{report.status.value} · {usage.total_tokens} tokens · "
                f"{usage.calls} calls · {report.run.duration_ms}ms"
                + (f" · ${usage.cost_usd:.4f}" if usage.cost_usd else ""),
                DIM,
            )
        )
        return 0 if report.ok else 1
    finally:
        store.close()


def cmd_debate(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    from athenacore.orchestration.debate import DebateOrchestrator

    orchestrator, store = _build_orchestrator(settings, quiet=args.json, verbose=args.verbose)
    try:
        debate = DebateOrchestrator(
            orchestrator,
            participants=args.participants,
            judge=None if args.no_judge else args.judge,
        )
        report = debate.run(topic=args.topic, query=args.query, rounds=args.rounds)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0
        print("\n" + report.transcript())
        print(_colour(f"\n{report.stop_reason} · {report.usage.total_tokens} tokens", DIM))
        return 0
    finally:
        store.close()


def cmd_recall(args: argparse.Namespace) -> int:
    """Query memory exactly the way agents do - the debugging tool for recall."""
    settings = _settings_from_args(args)
    from athenacore.memory.embeddings import HashingEmbedder
    from athenacore.memory.retrieval import MemoryRetriever

    store = _open_store(settings)
    try:
        embedder = (
            HashingEmbedder(dims=settings.embedding_dims) if settings.embeddings_enabled else None
        )
        retriever = MemoryRetriever(
            store,
            settings=settings.retrieval,
            embedder=embedder,
            token_budget=settings.context_token_budget,
        )
        result = retriever.recall(args.topic, args.query, max_entries=args.limit)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
            return 0
        if not result:
            print("No matching memory.")
            return 0
        print(
            _heading(
                f"\n{len(result)} entries · ~{result.tokens} tokens · from {result.candidates_considered} candidates\n"
            )
        )
        for item in result.entries:
            entry = item.entry
            stamp = entry.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"{_colour(entry.agent, BOLD)} · {entry.kind.value} · {stamp}")
            if args.explain:
                print(_colour(f"  {item.explain()}", DIM))
            print(f"  {entry.preview(300)}\n")
        return 0
    finally:
        store.close()


def cmd_topics(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    store = _open_store(settings)
    try:
        topics = store.list_topics(limit=args.limit, search=args.search)
        if args.json:
            print(json.dumps([t.to_dict() for t in topics], indent=2, default=str))
            return 0
        if not topics:
            print('No topics yet. Start one:  athenacore run "your question" --topic mytopic')
            return 0
        print(_heading(f"\n{'TOPIC':<28} {'ENTRIES':>8} {'TOKENS':>9}  UPDATED"))
        for topic in topics:
            print(
                f"{topic.name[:27]:<28} {topic.entry_count:>8} {topic.live_tokens:>9}  "
                f"{topic.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )
        print()
        return 0
    finally:
        store.close()


def cmd_timeline(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    store = _open_store(settings)
    try:
        entries = store.timeline(args.topic, limit=args.limit, include_archived=args.all)
        if args.json:
            print(json.dumps([e.to_dict() for e in entries], indent=2, default=str))
            return 0
        if not entries:
            print(f"No entries for topic {args.topic!r}.")
            return 0
        print(_heading(f"\nTimeline · {args.topic} · {len(entries)} entries\n"))
        for entry in entries:
            flag = _colour(" [archived]", DIM) if entry.archived else ""
            stamp = entry.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"{_colour(entry.agent, BOLD)} · {entry.kind.value} · {stamp}{flag}")
            print(f"  {entry.preview(240)}\n")
        return 0
    finally:
        store.close()


def cmd_runs(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    store = _open_store(settings)
    try:
        runs = store.list_runs(topic=args.topic, limit=args.limit)
        if args.json:
            print(json.dumps([r.to_dict() for r in runs], indent=2, default=str))
            return 0
        if not runs:
            print("No runs recorded yet.")
            return 0
        print(
            _heading(f"\n{'STARTED':<17} {'TOPIC':<20} {'PRESET':<28} {'STATUS':<10} {'TOKENS':>8}")
        )
        for run in runs:
            print(
                f"{run.started_at.strftime('%Y-%m-%d %H:%M'):<17} {run.topic[:19]:<20} "
                f"{run.preset[:27]:<28} {run.status.value:<10} {run.usage.total_tokens:>8}"
            )
        print()
        return 0
    finally:
        store.close()


def cmd_stats(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    store = _open_store(settings)
    try:
        stats = store.stats(topic=args.topic)
        if args.json:
            print(json.dumps(stats, indent=2, default=str))
            return 0
        print(_heading("\nMemory\n"))
        for key in ("topics", "entries", "archived", "tokens", "runs"):
            print(f"  {key:<12} {stats[key]}")
        print(f"  {'fts5':<12} {'enabled' if stats.get('fts') else 'unavailable'}")
        print(f"  {'database':<12} {stats.get('path')}")
        if stats["by_agent"]:
            print(_heading("\nBy agent\n"))
            width = max(stats["by_agent"].values())
            for agent, count in stats["by_agent"].items():
                bar = "█" * max(1, int(20 * count / width))
                print(f"  {agent:<14} {bar} {count}")
        usage = stats["usage"]
        print(_heading("\nModel usage\n"))
        print(f"  {'calls':<12} {usage['calls']}")
        print(f"  {'tokens':<12} {usage['total_tokens']}")
        if usage["cost_usd"]:
            print(f"  {'cost':<12} ${usage['cost_usd']:.4f}")
        print()
        return 0
    finally:
        store.close()


def cmd_export(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    store = _open_store(settings)
    try:
        entries = store.timeline(args.topic, limit=10_000, include_archived=args.all)
        if not entries:
            print(f"No entries for topic {args.topic!r}.", file=sys.stderr)
            return 1

        if args.format == "json":
            payload = json.dumps([e.to_dict() for e in entries], indent=2, default=str)
        else:
            topic = store.get_topic(args.topic)
            lines = [f"# {args.topic}", ""]
            if topic and topic.description:
                lines += [topic.description, ""]
            lines += [
                f"_{len(entries)} entries · exported {entries[-1].created_at.strftime('%Y-%m-%d')}_",
                "",
            ]
            for entry in entries:
                stamp = entry.created_at.strftime("%Y-%m-%d %H:%M")
                suffix = " *(archived)*" if entry.archived else ""
                lines += [
                    f"## {entry.kind.icon} {entry.agent} - {entry.kind.value}{suffix}",
                    f"_{stamp}_" + (f" · `{entry.model}`" if entry.model else ""),
                    "",
                    entry.content,
                    "",
                ]
            payload = "\n".join(lines)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"Wrote {len(entries)} entries to {args.output}")
        else:
            print(payload)
        return 0
    finally:
        store.close()


def cmd_forget(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    store = _open_store(settings)
    try:
        if store.get_topic(args.topic) is None:
            print(f"Topic {args.topic!r} does not exist.", file=sys.stderr)
            return 1
        if not args.yes:
            answer = input(f"Delete topic {args.topic!r} and all its entries? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled.")
                return 1
        removed = store.delete_topic(args.topic)
        print(f"Deleted topic {args.topic!r} ({removed} entries).")
        return 0
    finally:
        store.close()


def cmd_agents(args: argparse.Namespace) -> int:
    from athenacore.agents.registry import describe_agents

    agents = describe_agents()
    if args.json:
        print(json.dumps(agents, indent=2))
        return 0
    print(_heading("\nRegistered agents\n"))
    for agent in agents:
        print(f"  {_colour(agent['name'], BOLD):<24} {agent['role']}")
        print(f"  {'':<15} {_colour(agent['description'], DIM)}")
    print()
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    from athenacore.orchestration.presets import describe_presets

    presets = describe_presets()
    if args.json:
        print(json.dumps(presets, indent=2))
        return 0
    print(_heading("\nAvailable presets\n"))
    for preset in presets:
        print(f"  {_colour(preset['key'], BOLD):<38} {preset['title']} ({preset['cost']} cost)")
        print(f"    {_colour(preset['summary'], DIM)}")
        print(f"    {_colour(' → '.join(' | '.join(lvl) for lvl in preset['levels']), DIM)}")
    print(f"\n  Also: {_colour('solo:<agent>', BOLD)} to run any single agent.\n")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    from athenacore.orchestration.presets import build_preset

    graph = build_preset(args.preset, settings=settings)
    if args.format == "mermaid":
        print(graph.to_mermaid())
    elif args.format == "json":
        print(graph.to_json())
    else:
        print(graph.to_ascii())
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    from athenacore.memory.embeddings import HashingEmbedder
    from athenacore.memory.retrieval import MemoryRetriever

    store = _open_store(settings)
    try:
        retriever = MemoryRetriever(
            store,
            settings=settings.retrieval,
            embedder=HashingEmbedder(dims=settings.embedding_dims),
        )
        total = 0
        while True:
            done = retriever.index_pending(topic=args.topic, batch=200)
            total += done
            if done == 0:
                break
            print(f"  indexed {total}…", end="\r")
        print(f"Indexed {total} entries.        ")
        return 0
    finally:
        store.close()


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose the environment. The first thing to run when something is off."""
    settings = _settings_from_args(args)
    from athenacore.llm.registry import build_provider

    print(_heading("\nAthenaCore diagnostics\n"))
    ok = True

    print(f"  version        {__version__}")
    print(f"  python         {sys.version.split()[0]}")
    print(f"  model          {settings.model}")

    store = _open_store(settings)
    try:
        stats = store.stats()
        print(f"  database       {settings.database_path} ({stats['entries']} entries)")
        status = _colour("enabled", GREEN) if stats.get("fts") else _colour("UNAVAILABLE", YELLOW)
        print(f"  full-text      {status}")
        if not stats.get("fts"):
            print(_colour("                 recall falls back to substring matching", DIM))
    finally:
        store.close()

    print()
    try:
        provider = build_provider(settings.model, settings)
        reachable, detail = provider.health()
        if reachable:
            print(f"  provider       {_colour('reachable', GREEN)} ({provider.name})")
        else:
            ok = False
            print(f"  provider       {_colour('UNREACHABLE', RED)} ({provider.name})")
            for line in detail.splitlines():
                print(_colour(f"                 {line}", DIM))
    except AthenaCoreError as exc:
        ok = False
        print(f"  provider       {_colour('MISCONFIGURED', RED)}")
        print(_colour(f"                 {exc}", DIM))

    for extra, module, purpose in (
        ("api", "fastapi", "athenacore serve"),
        ("ui", "streamlit", "athenacore ui"),
        ("search", "duckduckgo_search", "web_search tool"),
    ):
        try:
            __import__(module)
            print(f"  {extra:<14} {_colour('installed', GREEN)}")
        except ImportError:
            print(
                f"  {extra:<14} not installed {_colour(f'(pip install athenacore[{extra}]) for {purpose}', DIM)}"
            )

    print()
    return 0 if ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    try:
        import uvicorn
    except ImportError:
        print("The API needs extra packages:  pip install 'athenacore[api]'", file=sys.stderr)
        return 1
    from athenacore.api.server import create_app

    os.environ.setdefault("ATHENA_DATABASE_PATH", str(settings.database_path))
    print(_heading(f"\nAthenaCore API on http://{args.host}:{args.port}  (docs at /docs)\n"))
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_level=settings.log_level.lower()
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The UI needs extra packages:  pip install 'athenacore[ui]'", file=sys.stderr)
        return 1
    import subprocess

    app = Path(__file__).parent / "ui" / "app.py"
    os.environ["ATHENA_DATABASE_PATH"] = str(settings.database_path)
    os.environ["ATHENA_MODEL"] = settings.model
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.port",
            str(args.port),
            "--server.headless",
            "true" if args.headless else "false",
        ]
    )


# -- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="athenacore",
        description="Multi-agent collaboration with persistent, searchable memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  athenacore run "Is refining the real EV bottleneck?" --topic batteries\n'
            '  athenacore run "..." --topic batteries --preset red-team --model ollama:llama3.1\n'
            '  athenacore debate "Will sodium-ion displace lithium?" --topic batteries --rounds 4\n'
            '  athenacore recall "water usage" --topic batteries --explain\n'
            "  athenacore export --topic batteries --format md -o report.md\n"
            "  athenacore doctor\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"athenacore {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--database", help="path to the SQLite memory file")
    common.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    common.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    model_opts = argparse.ArgumentParser(add_help=False)
    model_opts.add_argument("-m", "--model", help="provider:model, e.g. ollama:llama3.1")
    model_opts.add_argument("-t", "--temperature", type=float)
    model_opts.add_argument("--no-tools", action="store_true", help="disable tool calling")
    model_opts.add_argument("--web-search", action="store_true", help="enable the web search tool")
    model_opts.add_argument(
        "-v", "--verbose", action="store_true", help="show recall and token events"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common, model_opts], help="run an agent graph on a topic")
    run.add_argument("query", help="the question to work on")
    run.add_argument("--topic", required=True, help="memory topic (shared across runs)")
    run.add_argument("-p", "--preset", default="research-critique-synthesis")
    run.set_defaults(func=cmd_run)

    debate = sub.add_parser("debate", parents=[common, model_opts], help="run a converging debate")
    debate.add_argument("query")
    debate.add_argument("--topic", required=True)
    debate.add_argument("--rounds", type=int, help="maximum rounds (may stop earlier)")
    debate.add_argument("--participants", nargs="+", default=["research", "critic"])
    debate.add_argument("--judge", default="synthesizer")
    debate.add_argument("--no-judge", action="store_true")
    debate.set_defaults(func=cmd_debate)

    recall = sub.add_parser("recall", parents=[common], help="query memory the way agents do")
    recall.add_argument("query")
    recall.add_argument("--topic", required=True)
    recall.add_argument("-n", "--limit", type=int, default=10)
    recall.add_argument("--explain", action="store_true", help="show the ranking breakdown")
    recall.set_defaults(func=cmd_recall)

    topics = sub.add_parser("topics", parents=[common], help="list topics")
    topics.add_argument("-n", "--limit", type=int, default=50)
    topics.add_argument("--search")
    topics.set_defaults(func=cmd_topics)

    timeline = sub.add_parser("timeline", parents=[common], help="show a topic's history")
    timeline.add_argument("--topic", required=True)
    timeline.add_argument("-n", "--limit", type=int, default=50)
    timeline.add_argument("--all", action="store_true", help="include archived entries")
    timeline.set_defaults(func=cmd_timeline)

    runs = sub.add_parser("runs", parents=[common], help="list past runs")
    runs.add_argument("--topic")
    runs.add_argument("-n", "--limit", type=int, default=25)
    runs.set_defaults(func=cmd_runs)

    stats = sub.add_parser("stats", parents=[common], help="memory and usage statistics")
    stats.add_argument("--topic")
    stats.set_defaults(func=cmd_stats)

    export = sub.add_parser("export", parents=[common], help="export a topic")
    export.add_argument("--topic", required=True)
    export.add_argument("--format", choices=["md", "json"], default="md")
    export.add_argument("-o", "--output")
    export.add_argument("--all", action="store_true", help="include archived entries")
    export.set_defaults(func=cmd_export)

    forget = sub.add_parser("forget", parents=[common], help="delete a topic and its memory")
    forget.add_argument("--topic", required=True)
    forget.add_argument("-y", "--yes", action="store_true")
    forget.set_defaults(func=cmd_forget)

    agents = sub.add_parser("agents", parents=[common], help="list registered agents")
    agents.set_defaults(func=cmd_agents)

    presets = sub.add_parser("presets", parents=[common], help="list workflow presets")
    presets.set_defaults(func=cmd_presets)

    graph = sub.add_parser("graph", parents=[common], help="render a preset's graph")
    graph.add_argument("preset")
    graph.add_argument("--format", choices=["ascii", "mermaid", "json"], default="ascii")
    graph.set_defaults(func=cmd_graph)

    reindex = sub.add_parser(
        "reindex", parents=[common], help="build embeddings for existing memory"
    )
    reindex.add_argument("--topic")
    reindex.set_defaults(func=cmd_reindex)

    doctor = sub.add_parser("doctor", parents=[common, model_opts], help="diagnose the setup")
    doctor.set_defaults(func=cmd_doctor)

    serve = sub.add_parser("serve", parents=[common], help="start the REST/SSE API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    ui = sub.add_parser("ui", parents=[common], help="start the Streamlit UI")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--headless", action="store_true")
    ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AthenaCoreError as exc:
        print(_colour(f"\nerror: {exc}", RED), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
