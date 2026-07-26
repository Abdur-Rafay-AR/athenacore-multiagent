# Recipes

Concrete things you can do with Crucible, in rough order of how likely you are
to want them.

## Return to a topic weeks later

The point of persistent memory. Run `catch-up` on a topic you have not touched
since last month - no new research, just a report of where things stand, built
from what the agents established and disputed.

```bash
crucible run "" --topic batteries --preset catch-up
crucible timeline --topic batteries        # or read the full history
```

## Understand why a memory surfaced

The single most useful debugging command in the project:

```bash
crucible recall "water usage" --topic batteries --explain
```

```
research · research · 2026-07-24 09:12
  score=0.812 (kw=1.00 sem=0.74 rec=0.61 sal=0.60)
  Lithium brine extraction in Chile's Atacama consumes roughly 500,000 litres…
```

Reading the breakdown tells you which weight to change. If irrelevant-but-recent
entries dominate, lower `CRUCIBLE_RECALL_W_RECENCY`. If you get five paraphrases of
one point, lower `CRUCIBLE_RECALL_MMR_LAMBDA`.

## Pin something the agents must not forget

Salience feeds the ranking directly, and notes are protected from compaction:

```python
from crucible import SqliteMemoryStore
from crucible.memory import Entry, EntryKind

store = SqliteMemoryStore("data/crucible.sqlite3")
store.add_entry(
    Entry(
        topic="batteries",
        agent="user",
        kind=EntryKind.DECISION,  # protected from compaction, forever
        content="We decided to scope this to LFP chemistry only.",
        salience=1.0,  # maximum recall priority
    )
)
```

Or from the UI: Memory view → expand an entry → adjust its salience.

## Run a debate that ends when it should

```bash
crucible debate "Will sodium-ion displace lithium in grid storage?" \
  --topic batteries --rounds 6
```

It will usually stop before round six. The transcript records the similarity
between consecutive rounds, so you can see *why* it stopped. If debates end too
early, raise `CRUCIBLE_DEBATE_CONVERGENCE` toward 0.98.

Change who argues:

```bash
crucible debate "..." --topic t --participants research critic insight --judge synthesizer
crucible debate "..." --topic t --no-judge     # leave it unresolved
```

## Build a custom workflow

```python
from crucible import AgentGraph, GraphNode, Orchestrator, Settings, SqliteMemoryStore

graph = AgentGraph(
    [
        GraphNode(name="research", agent="research"),
        # Two critics at different temperatures find different objections
        # than one critic asked twice.
        GraphNode(
            name="skeptic",
            agent="critic",
            depends_on=("research",),
            task="Attack the causal claims in: {query}",
            overrides={"temperature": 0.3},
        ),
        GraphNode(
            name="wildcard",
            agent="critic",
            depends_on=("research",),
            task="What is the least obvious way this fails?",
            overrides={"temperature": 0.95},
        ),
        GraphNode(name="audit", agent="factcheck", depends_on=("research",), optional=True),
        GraphNode(name="verdict", agent="synthesizer", depends_on=("skeptic", "wildcard", "audit")),
    ],
    name="dual-critic",
)

settings = Settings.from_env()
store = SqliteMemoryStore(settings.database_path)
report = Orchestrator(store, settings=settings).run(
    graph, topic="batteries", query="Is refining the binding constraint?"
)
print(report.output("verdict"))
```

Check the shape before spending tokens:

```python
print(graph.to_ascii())
print(graph.levels())  # what runs in parallel
```

## Skip a node unless something held

Conditions let a graph adapt to its own results:

```python
orchestrator.register_condition(
    "reached_a_conclusion",
    lambda report: "Bottom line" in report.output("synthesizer"),
)

graph.add(
    GraphNode(
        name="planner",
        agent="planner",
        depends_on=("synthesizer",),
        condition="reached_a_conclusion",  # no plan without a conclusion
    )
)
```

## Use a real embedding model

The default embedder is offline and dependency-free. When you want semantic
recall that understands synonyms:

```python
from sentence_transformers import SentenceTransformer
from crucible.memory import CallableEmbedder
from crucible.orchestration import Orchestrator

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = CallableEmbedder(
    lambda texts: model.encode(list(texts)).tolist(), dims=384, name="minilm"
)

orchestrator = Orchestrator(store, settings=settings, embedder=embedder)
```

Then re-index existing memory, since the vectors changed dimension:

```bash
crucible reindex
```

## Stream a run into your own application

```python
from crucible.orchestration import EventType


def on_event(event):
    if event.type is EventType.NODE_FINISHED:
        print(f"{event.node} finished: {event.data.get('usage')}")
    elif event.type is EventType.CONVERGED:
        print("debate settled:", event.message)


orchestrator.bus.subscribe(on_event)
```

Over HTTP, the same events arrive as SSE:

```bash
curl -N -X POST localhost:8000/runs/stream \
  -H 'content-type: application/json' \
  -d '{"topic":"batteries","query":"...","preset":"red-team"}'
```

## Script it

Every command speaks JSON:

```bash
# Just the adjudicated conclusion
crucible run "..." --topic t --json | jq -r '.results.synthesizer.content'

# What did this cost?
crucible run "..." --topic t --json | jq '.run.usage'

# Which agents failed?
crucible run "..." --topic t --json | jq '.results | to_entries
  | map(select(.value.error)) | map({(.key): .value.error})'

# Nightly digest across every topic
for topic in $(crucible topics --json | jq -r '.[].name'); do
  crucible export --topic "$topic" --format md -o "reports/$topic.md"
done
```

## Run entirely offline, forever

```bash
export CRUCIBLE_MODEL=ollama:llama3.1
export CRUCIBLE_WEB_SEARCH_ENABLED=false     # the default
export CRUCIBLE_EMBEDDINGS_ENABLED=true      # the built-in embedder is local
```

Nothing leaves the machine. The memory is one SQLite file you own.

## Tune for a small local model

7B-class models need a tighter context and cheaper workflows:

```bash
export CRUCIBLE_CONTEXT_TOKEN_BUDGET=2500     # less recalled memory per prompt
export CRUCIBLE_MAX_OUTPUT_TOKENS=700
export CRUCIBLE_RECALL_MAX_ENTRIES=6
export CRUCIBLE_MAX_PARALLEL_AGENTS=2         # avoid thrashing one GPU
crucible run "..." --topic t --preset brief
```

Larger hosted models can go the other way - raise the budget, use `deep-dive`,
and increase parallelism.

## Inspect the database directly

It is just SQLite. Nothing is hidden from you:

```bash
sqlite3 data/crucible.sqlite3 \
  "SELECT agent, kind, salience, substr(content,1,60)
     FROM entries WHERE topic='batteries' AND archived=0
     ORDER BY created_at DESC LIMIT 10;"
```

## Back up and move memory

```bash
cp data/crucible.sqlite3 backups/crucible-$(date +%F).sqlite3
crucible export --topic batteries --format json -o batteries.json
```
