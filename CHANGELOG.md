# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-26

### Changed
- **Renamed the project from AthenaCore to Crucible.** The former name belongs to
  a company that owns the original version of this work, so it could not be
  carried forward here.
- The import name, CLI command and logging namespace are all now `crucible`. The
  environment variable prefix is `CRUCIBLE_` (previously `ATHENA_`), and the
  default database is `data/crucible.sqlite3`.
- The distribution is published as `crucible-agents`, because `crucible` is
  already taken on PyPI. `pip install crucible-agents`, `import crucible`.

### Migration
Rename your environment variables from `ATHENA_*` to `CRUCIBLE_*`, and either
point `CRUCIBLE_DATABASE_PATH` at your existing file or rename it to
`data/crucible.sqlite3`. The database schema is unchanged, so no data migration
is needed.

## [0.2.0] - 2026-07-26

A rewrite. The prototype was four agents sharing a flat TinyDB log; this is an
engine with a real retrieval system, a parallel orchestrator and three interfaces.

### Added

**Memory**
- `SqliteMemoryStore`: WAL, schema migrations via `PRAGMA user_version`, an FTS5
  external-content index kept in sync by triggers, float32 embedding blobs and
  denormalised topic counters.
- `MemoryRetriever`: hybrid recall over BM25 + embedding cosine + exponential
  recency decay + salience, with MMR de-duplication and token budgeting. Every
  result carries its per-signal breakdown.
- `HashingEmbedder`: deterministic offline embeddings from the standard library.
  `CallableEmbedder` wraps any external model.
- `MemoryCompactor`: folds old entries into a durable summary once a topic exceeds
  its token budget, protects notes/decisions/summaries, and supports `undo()`.
- Typed append-only domain model (`Topic`, `Entry`, `Run`, `Usage`) with 11 entry
  kinds; `InMemoryMemoryStore` for tests.

**Agents**
- Seven agents (research, critic, factcheck, insight, synthesizer, summarizer,
  planner) over one shared execution cycle, each with its own recall policy.
- Critic and summarizer cannot see their own prior output, so they cannot
  re-litigate earlier positions.
- Registry with entry-point discovery for third-party agents.

**Orchestration**
- `AgentGraph`: validated DAG with cycle detection, level grouping for
  parallelism, conditions, JSON round-tripping and Mermaid/ASCII rendering.
- `Orchestrator`: thread-pool execution by level, branch-scoped failure isolation
  with `PARTIAL` status, optional nodes, cooperative cancellation, automatic
  post-run compaction.
- `DebateOrchestrator`: multi-round debate with concede/sharpen/hold framing and
  early stopping on measured round-to-round convergence.
- Five presets plus `solo:<agent>`, and an event bus driving all interfaces.

**Providers and tools**
- Ollama, OpenAI-compatible (with aliases for Groq, Together, DeepSeek,
  OpenRouter, Mistral, Fireworks, vLLM, LM Studio, llama.cpp), Anthropic, plus
  Echo and Scripted fakes. All over `urllib`; no SDKs.
- Retries with jittered exponential backoff, usage accounting, cost estimation
  and a streaming-to-blocking fallback.
- Text tool protocol that works on any model, with a whitelisted-AST calculator,
  memory search/write, a clock and opt-in web search.

**Interfaces**
- CLI with 15 subcommands, `--json` everywhere, and `doctor` for diagnostics.
- FastAPI REST API with SSE streaming of run events.
- Streamlit control room: live run trace, memory search with tunable weights,
  timeline, run history and analytics.

**Project**
- 176 offline tests, including a store conformance suite parametrised over both
  implementations and UI coverage via Streamlit's `AppTest`.
- CI across Linux/macOS/Windows on Python 3.10-3.12, with a job that enforces the
  zero-dependency core by installing without extras and driving a full run.
- Architecture and recipe documentation, contributing guide, `.env.example`.

### Changed
- Storage moved from TinyDB to SQLite. The core engine now has **zero**
  dependencies; FastAPI and Streamlit are optional extras.
- Installable package with a `crucible` entry point, replacing loose scripts.

### Removed
- The prototype `agents/`, `orchestrator.py` and `frontend.py` modules, the
  committed TinyDB store and the empty `requirements.txt`.

### Migration
There is no automatic migration from the 0.1 TinyDB file. The schemas share no
structure: 0.1 stored `{name, log: [{agent, content}]}` with no timestamps, kinds,
salience or runs. Start a fresh topic, or insert old entries yourself with
`store.add_entry(Entry(...))`.

## [0.1.0] - 2026-01-20
- Initial prototype: four agents, TinyDB-backed shared log, Streamlit UI.
