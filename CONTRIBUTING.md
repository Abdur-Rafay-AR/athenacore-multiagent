# Contributing

Thanks for considering it. This project is small enough that a good pull request
can land quickly.

## Setup

```bash
git clone https://github.com/Abdur-Rafay-AR/crucible
cd crucible
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,api,ui]"
pytest
```

The suite is fully offline - it runs against the deterministic `echo` provider and
temporary SQLite files. If a test needs a real model or the network, it is in the
wrong place; mark it `@pytest.mark.slow` and expect it to be skipped in CI.

## Before opening a PR

```bash
ruff check . && ruff format . && mypy src && pytest
```

That is what CI runs, plus the same suite on Linux, macOS and Windows across
Python 3.10-3.12, and a job that installs the package with **no extras** and
drives a full CLI run.

## The one rule that matters

**The core must stay dependency-free.** `crucible.memory`,
`crucible.agents`, `crucible.llm`, `crucible.tools`,
`crucible.orchestration` and the CLI import only the standard library. That is
not aesthetic minimalism - it is why the project installs and runs anywhere, and
CI enforces it.

If you need a third-party package:

- Put it behind an extra in `pyproject.toml` (`api`, `ui`, `search`, or a new one).
- Import it lazily, inside the function that needs it.
- Degrade gracefully when it is missing, with a message naming the install command.
  `WebSearchTool` and `CallableEmbedder` are the patterns to copy.

## Adding an agent

An agent is a role, a recall policy and a prompt:

```python
from crucible.agents import Agent, RecallPolicy, register_agent
from crucible.memory import EntryKind


@register_agent
class MyAgent(Agent):
    name = "myagent"  # unique; how it is referenced everywhere
    role = "My Role"  # shown in the UI and logs
    entry_kind = EntryKind.INSIGHT
    temperature = 0.5
    recall_policy = RecallPolicy(
        kinds=(EntryKind.RESEARCH,),  # None means every kind
        max_entries=8,
        token_budget=2500,
        include_own_previous=False,  # stops it anchoring on its own past output
    )

    @property
    def system_prompt(self) -> str:
        return "My Role.\nWhat this agent does and the constraints it works under."
```

Prompt notes, learned the hard way against 7B local models:

- **Start with the role name on its own line.** Logs and the offline provider key
  off it.
- **Be blunt and specific.** "Give at most four insights, each one bold claim plus
  two sentences of support" beats "be insightful". Vague prompts produce vague
  output, and small models are much less forgiving than frontier ones.
- **Say what failure looks like.** "A restatement is a failure" measurably improves
  the insight agent.
- **Don't repeat the output contract** - `Agent.output_contract()` already handles
  formatting, citations and anti-hedging for every agent.

Add a test asserting the agent produces its declared `entry_kind` and writes to
memory. `tests/test_engine.py::TestAgents` has the pattern.

## Adding a tool

```python
from crucible.tools import Tool
from crucible.errors import ToolError


class MyTool(Tool):
    name = "my_tool"
    description = "One line the model reads to decide whether to call this."
    parameters = {"arg": {"type": "string", "description": "what it is"}}
    required = ("arg",)
    safe = True  # False for side effects, cost, or network egress

    def run(self, **kwargs) -> str:
        if not kwargs["arg"]:
            raise ToolError("arg cannot be empty", tool=self.name)
        return "result text the model will read"
```

Raise `ToolError` for expected failures; the registry converts it into an
observation the model can recover from. Anything with side effects, cost or
network access must set `safe = False` so it is opt-in.

## Style

- Type annotations on public functions.
- Docstrings that explain **why**, not what. `store.py` and `retrieval.py` are the
  standard to aim for. If a decision involved a trade-off, write down the trade.
- Comment the non-obvious and skip the obvious. `# increment the counter` is noise;
  "jitter matters because a graph fans agents out simultaneously" is not.
- Tests assert behaviour, not implementation. A test that breaks when you rename a
  private method is a liability.

## Commits and PRs

Write commit messages that explain the reasoning, not just the diff. One logical
change per commit.

In the PR, say what you changed, why, and how you verified it. If you changed
retrieval or prompts, include a before/after - `crucible recall --explain` is
the right evidence for ranking changes.

## Reporting bugs

Include the output of `crucible doctor`, the command you ran, and what you
expected. That covers the environmental causes, which is most of them.
