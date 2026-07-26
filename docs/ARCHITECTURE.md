# Architecture

This document explains how AthenaCore is put together and, more usefully, *why*.
Every design decision here was a trade; the reasoning matters more than the
structure, because the reasoning is what tells you whether a change you want to
make is safe.

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Interfaces      cli.py  ·  api/server.py  ·  ui/app.py      │
│                  (no business logic - all three call down)   │
├──────────────────────────────────────────────────────────────┤
│  Orchestration   graph · orchestrator · debate · presets     │
│                  events (fan-out to all three interfaces)    │
├──────────────────────────────────────────────────────────────┤
│  Agents          base (one execution cycle) · builtin · registry │
├──────────────────────────────────────────────────────────────┤
│  Capabilities    llm/ (providers)      tools/ (text protocol)│
├──────────────────────────────────────────────────────────────┤
│  Memory          models · store · retrieval · compaction     │
│                  embeddings                                  │
├──────────────────────────────────────────────────────────────┤
│  Foundation      config · errors · logging_setup             │
└──────────────────────────────────────────────────────────────┘
```

Dependencies point strictly downward. The practical test: `athenacore.memory`
imports nothing from `athenacore.agents`, and the interface layer contains no
logic you cannot also reach from Python. That is what makes the CLI, API and UI
genuinely equivalent rather than three drifting implementations.

## Memory

### Entries are immutable

An entry is never edited. Corrections are new entries; compaction *supersedes*
entries by archiving them with a pointer to their replacement. This costs storage
and buys the property the project is actually about: you can always reconstruct
how a topic's understanding evolved, and no agent can quietly rewrite history.

`EntryKind` drives three behaviours - which entries an agent prefers to recall,
how the UI renders them, and whether compaction may fold them away. `SUMMARY`,
`DECISION` and `NOTE` are `protected`: human notes and prior conclusions survive
compaction verbatim, forever.

### Why SQLite and not a vector database

A topic accumulates thousands of entries, not millions. At that scale:

- Cosine similarity over a few hundred candidate vectors in pure Python is
  microseconds. The bottleneck is the model, by four orders of magnitude.
- FTS5 gives real BM25 ranking, which for factual recall frequently *beats*
  embedding similarity. A hybrid of both beats either.
- One file means an operator can copy, back up, diff, or open the whole memory in
  any SQLite browser. Nothing to deploy, nothing to keep running.

If you outgrow this, `MemoryStore` is an ABC - implement it against pgvector and
nothing above the memory layer changes. The conformance suite in
`tests/test_memory.py` (`TestStoreConformance`) is parametrised over
implementations, so a new backend inherits a specification.

### Storage details worth knowing

- **WAL mode** so the API server, CLI and UI can share one file, with a 10s busy
  timeout for the rare write conflict.
- **FTS5 external-content index** synchronised by triggers, so search can never
  drift from the entries table. If the interpreter lacks FTS5, the store logs a
  warning and degrades to `LIKE` matching rather than failing.
- **Migrations** keyed off `PRAGMA user_version`, applied on connect. Opening a
  database written by a *newer* version fails loudly instead of corrupting it.
- **Denormalised counters** (`entry_count`, `live_tokens`) on `topics`, so the
  compaction trigger and the topic list are single-row reads.
- **One connection + an RLock**, not a pool. Contention here is a handful of
  agents; serialising keeps transaction semantics obvious.

### Retrieval

The pipeline is deliberately explicit rather than a single similarity call:

1. **Gather** - the union of BM25 hits and the most recent live entries. Including
   recency-only candidates is what lets recall work with *no query at all*, which
   the summarizer needs.
2. **Score** - four signals, each min-max normalised **across the candidate set**,
   then weighted. Normalising per-recall rather than globally means the weights
   mean "relative importance among these candidates", which stays stable as raw
   BM25 magnitudes drift with corpus size.
3. **Diversify** - MMR. Without it, recall returns five paraphrases of the same
   critique and wastes the budget. This matters far more here than in ordinary RAG,
   because agents restate each other constantly by design.
4. **Budget** - admit entries until the token budget is spent, so a prompt can
   never overflow the context window.

Each result carries its per-signal breakdown all the way to the UI and
`athenacore recall --explain`. An unexplainable ranker is one you cannot tune.

### Embeddings without a dependency

`HashingEmbedder` is the classic hashing trick over word unigrams, word bigrams
and character 4-grams, with signed buckets (so collisions cancel instead of
always inflating), sub-linear term frequency, and `blake2b` for reproducibility
across processes - `hash()` is salted per-process and would produce different
vectors on every run.

It will not capture "car" ≈ "automobile". It *will* capture paraphrase,
inflection, plurals and typos, which is most of what recall needs from the
*second* signal behind BM25. When you want more, `CallableEmbedder` wraps any
model in one line and the rest of the system is unchanged.

### Compaction

Triggered when a topic's live tokens exceed the threshold. Folds the oldest
non-protected entries into one summary, archives the originals with
`superseded_by` set, and refuses to act if the "summary" comes back empty or no
smaller than its input - a lossy operation that makes memory *worse* should not
happen silently. `undo()` restores the originals, which is what makes it safe to
run automatically after every run.

The summariser is injected as a plain callable, so this module has no dependency
on the LLM layer and is trivially testable.

## Agents

Every agent runs one cycle, defined once in `agents/base.py`: recall → compose →
generate (with the tool loop) → persist. Subclasses normally override only four
things: `role`, `system_prompt`, `entry_kind` and `recall_policy`.

That constraint is the design. It keeps each agent to a few dozen lines and keeps
the interesting behaviour - memory, tools, orchestration, provenance - shared
rather than copy-pasted seven times.

### Recall policies are per-role

A critic wants the claims to attack. A summarizer wants everything. A planner
mostly wants conclusions. Handing every agent the same context is the fastest way
to get seven bland restatements of the same paragraph.

Two policies do something subtler: `CriticAgent` and `SummarizerAgent` set
`include_own_previous=False`, so they cannot see their own prior output. The
critic therefore cannot re-litigate its earlier critiques, which is the single
most effective anti-degeneration measure in the system.

### Provenance

Each persisted entry records the recalled entry ids, the resolved `[n]` citations,
the tools used, the model, and latency. You can reconstruct exactly what an agent
saw when it said something - which is the difference between an auditable system
and a pile of text.

## Tools

Native function calling is unevenly supported across local models, and the
default path for this project is Ollama on a laptop. So tools use a text protocol:

```
TOOL: calculator {"expression": "1.4e6 * 0.47"}
```

Parsing is deliberately lenient (single quotes, trailing commas, missing argument
objects all accepted) because rejecting a nearly-correct call costs a full round
trip. Failures become `ToolResult` values that the model reads as an
`OBSERVATION`, never exceptions - a bad tool call should let the model recover,
not kill the run. Tools still declare JSON-Schema specs, so a native
function-calling backend can be added without touching any tool.

The calculator evaluates through a whitelisted AST walk, never `eval`, with
guards on exponent size. It is the only responsible way to put model output near
an expression evaluator.

## Orchestration

### Levels, not a flat order

`AgentGraph.levels()` returns the topological order **grouped**: everything in one
level is independent and dispatched to a thread pool together. Two critics with no
relationship to each other genuinely run at the same time.

### Failure isolation

A failed node marks only its own descendants skipped. Unrelated branches complete
and the run reports `PARTIAL`. Nodes marked `optional=True` (the fact-checker, for
example) do not skip their dependents at all. The alternative - abort everything:
throws away expensive work because one agent timed out.

### Threads, not asyncio

Every provider call is blocking HTTP through `urllib`; every store write is
blocking `sqlite3`. A thread pool matches that exactly. Async would force
`await` through the whole codebase and into every agent, and buy nothing at this
concurrency.

### Debate and measured convergence

A DAG runs each agent once. A debate runs them against each other's *latest*
positions, sequentially within a round so each participant actually sees what the
previous one just said.

Knowing when to stop is the hard part. Fixed round counts either burn tokens on
agents agreeing or cut off while positions are still moving. So each round's
combined output is embedded and compared with the previous round's; once cosine
similarity crosses the threshold, positions have stopped moving and the debate
ends. The check is free - the embedder is local.

Prompts for rounds after the first require each agent to `CONCEDE`, `SHARPEN` or
`HOLD` on each point, which is what stops a debate from becoming two monologues.

### Events

The orchestrator emits events rather than printing. One engine therefore drives
the CLI trace, the SSE stream and the live Streamlit timeline without knowing any
of them exist. The bus is synchronous and non-throwing: a subscriber that raises
gets logged, never propagated. `EventQueue` adapts push to pull for consumers that
need to poll, and drops rather than blocks when a slow consumer falls behind:
a stalled UI must not stall the run.

Cancellation is cooperative, checked between nodes. Pre-emptively killing a
thread mid-HTTP-call would leak connections and could persist a half-formed entry.

## Providers

One base class owns retries with **jittered** exponential backoff, timeouts,
usage accounting and the streaming-to-blocking fallback. Jitter matters more than
it looks: a graph fans several agents out at the same instant, and un-jittered
retries would keep them synchronised into the same rate limit.

Everything speaks HTTP through `urllib` - no SDKs, so no version conflicts and no
supply chain. The OpenAI-compatible provider is one class serving OpenAI, Groq,
Together, DeepSeek, OpenRouter, vLLM, llama.cpp and LM Studio.

`EchoProvider` deserves a note: it is deterministic, offline, and echoes salient
nouns from the prompt back in role-appropriate shapes. That last part is what
makes it useful rather than a stub - recall, MMR de-duplication and convergence
detection all have real signal to work on in tests, instead of a constant string
that would make every similarity 1.0.

## Testing

176 tests, entirely offline, with mypy enforced in CI. No model, no daemon, no network.

- Store tests are a **conformance suite** parametrised over both implementations.
- Retrieval tests assert *behaviour* - that recency weighting prefers new entries,
  that MMR drops duplicates - not implementation details.
- Compaction tests cover the refusal cases and `undo`, because the failure modes
  matter more than the happy path.
- The calculator is tested against code execution, file access and resource
  exhaustion attempts.
- Orchestration tests assert failure isolation: that a dead branch is skipped
  while an unrelated branch still completes.
- The UI is executed through Streamlit's `AppTest`, which catches the failure that
  matters most: a view crashing on empty state after a change elsewhere.

## Extending

| To add… | Do this |
|---|---|
| An agent | Subclass `Agent`, decorate with `@register_agent`. Available everywhere immediately. |
| An agent from another package | Register under the `athenacore.agents` entry-point group. |
| A tool | Subclass `Tool`, add it to a `ToolRegistry`. |
| A provider | Subclass `LLMProvider`, call `register_provider(name, factory)`. |
| A storage backend | Implement `MemoryStore`; run the conformance suite against it. |
| A workflow | Build an `AgentGraph`, or add a `Preset`. |
| A real embedding model | Wrap it in `CallableEmbedder`. |
