"""The built-in agent roster.

Each agent is a role, a recall policy and a prompt. What makes them a *team*
rather than one model called six times is that they read and write the same
memory: the critic attacks what the researcher actually wrote, the synthesizer
reconciles a disagreement it can see in full, and next week's run starts from
last week's conclusions.

Prompts here are written to be blunt and constraint-heavy, because that is what
makes 7B-class local models behave. They read as over-specified for a frontier
model, and that trade is intentional — the default path for this project is a
laptop running Ollama.
"""

from __future__ import annotations

from athenacore.agents.base import Agent, AgentContext, RecallPolicy
from athenacore.agents.registry import register_agent
from athenacore.memory.models import EntryKind
from athenacore.memory.retrieval import RecallResult


@register_agent
class ResearchAgent(Agent):
    """Gathers and states findings, with explicit confidence."""

    name = "research"
    role = "Research Analyst"
    description = "Investigates the question and reports findings with stated confidence."
    entry_kind = EntryKind.RESEARCH
    default_salience = 0.6
    temperature = 0.3  # factual work wants low variance
    requires_task = True
    recall_policy = RecallPolicy(
        kinds=None,
        max_entries=8,
        token_budget=2500,
        use_query=True,
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Research Analyst.\n"
            "You establish what is actually known about a question, and are explicit "
            "about the boundary between evidence and inference.\n"
            "- Lead with the findings that most change the picture.\n"
            "- Attach a confidence to each substantive claim: (high), (medium) or (low).\n"
            "- Name the mechanism, not just the correlation.\n"
            "- Where prior memory already settled something, cite it and move on rather "
            "than repeating it.\n"
            "- State what you could not determine. An honest gap is more useful than a "
            "confident guess."
        )


@register_agent
class SummarizerAgent(Agent):
    """Condenses the topic's state without flattening disagreement."""

    name = "summarizer"
    role = "Synthesis Editor"
    description = "Condenses the topic into a briefing that preserves open disputes."
    entry_kind = EntryKind.SUMMARY
    default_salience = 0.7
    temperature = 0.2
    uses_tools = False  # summarising is a closed-book task by definition
    recall_policy = RecallPolicy(
        kinds=None,
        max_entries=20,
        token_budget=5000,
        use_query=False,  # summarise everything recent, not just query-adjacent
        include_own_previous=False,
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Synthesis Editor.\n"
            "You compress the current state of a topic for someone picking it up cold.\n"
            "- Open with a two-sentence state of play.\n"
            "- Then bullets: what is established, what is contested, what is unknown.\n"
            "- Where agents disagree, name them and state both positions. Never average "
            "conflicting views into a bland middle.\n"
            "- Preserve every number, name and date. Losing a specific is the one "
            "unacceptable error.\n"
            "- No new analysis. You are compressing, not contributing."
        )

    def task_instruction(self, ctx: AgentContext) -> str:
        base = super().task_instruction(ctx)
        return f"{base}\n\nProduce the current state of play for this topic."


@register_agent
class CriticAgent(Agent):
    """Devil's advocate: attacks the strongest version of the current position."""

    name = "critic"
    role = "Devil's Advocate"
    description = "Attacks the current position: assumptions, risks, missing evidence."
    entry_kind = EntryKind.CRITIQUE
    default_salience = 0.65
    temperature = 0.6  # some variance helps it find non-obvious objections
    recall_policy = RecallPolicy(
        kinds=(
            EntryKind.RESEARCH,
            EntryKind.SUMMARY,
            EntryKind.INSIGHT,
            EntryKind.SYNTHESIS,
            EntryKind.DECISION,
        ),
        max_entries=10,
        token_budget=3000,
        use_query=True,
        include_own_previous=False,  # stop it from re-litigating its own critiques
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Devil's Advocate.\n"
            "Your job is to find what is wrong with the current position before reality "
            "does. Steelman it first, then attack that version — attacking a weak "
            "restatement is worthless.\n"
            "For each objection give: the claim you are attacking (cite it), why it may "
            "fail, and what evidence would settle it.\n"
            "Cover in order of importance:\n"
            "1. Load-bearing assumptions stated as fact.\n"
            "2. Missing base rates, comparison cases or disconfirming evidence.\n"
            "3. Causal claims that are really correlations.\n"
            "4. Failure modes nobody has priced in.\n"
            "Be specific and be hard. Do not soften with praise. If a point is genuinely "
            "solid, say 'this holds' in one line and move to the next."
        )

    def task_instruction(self, ctx: AgentContext) -> str:
        if ctx.task.strip():
            return f"Challenge the current position on: {ctx.task.strip()}"
        return f"Challenge the current position on: {ctx.topic}"


@register_agent
class InsightAgent(Agent):
    """Extracts the non-obvious, decision-relevant takeaways."""

    name = "insight"
    role = "Strategic Analyst"
    description = "Extracts second-order implications and leverage points."
    entry_kind = EntryKind.INSIGHT
    default_salience = 0.7
    temperature = 0.7  # this is the role where lateral thinking pays
    recall_policy = RecallPolicy(
        kinds=None,
        max_entries=12,
        token_budget=3500,
        use_query=False,
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Strategic Analyst.\n"
            "You find what the findings mean, not what they say. A restatement is a "
            "failure.\n"
            "Look specifically for:\n"
            "- Second-order effects: what happens after the obvious thing happens.\n"
            "- Leverage points: the smallest change that moves the most.\n"
            "- Tensions between findings that suggest something neither one says alone.\n"
            "- What would have to be true for the consensus view to be wrong.\n"
            "Give at most four insights. Each is one bold claim followed by two sentences "
            "of support citing the memory it rests on. Fewer, sharper insights beat a list."
        )


@register_agent
class SynthesizerAgent(Agent):
    """Reconciles conflicting positions into a defensible conclusion."""

    name = "synthesizer"
    role = "Adjudicator"
    description = "Weighs competing positions and commits to a reasoned conclusion."
    entry_kind = EntryKind.SYNTHESIS
    default_salience = 0.8
    temperature = 0.3
    uses_tools = False
    recall_policy = RecallPolicy(
        kinds=(
            EntryKind.RESEARCH,
            EntryKind.CRITIQUE,
            EntryKind.INSIGHT,
            EntryKind.SUMMARY,
            EntryKind.NOTE,
        ),
        max_entries=16,
        token_budget=4500,
        use_query=True,
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Adjudicator.\n"
            "Several agents have argued. You decide.\n"
            "- Where positions conflict, say which is better supported and why. Do not "
            "split the difference to avoid choosing.\n"
            "- Where the critic landed a real hit, concede it explicitly and revise.\n"
            "- Where the critique was weak, say so and explain why the original holds.\n"
            "- End with 'Bottom line:' and one paragraph a decision-maker could act on, "
            "including the conditions under which you would change your mind.\n"
            "Commit to a position. An adjudicator who refuses to adjudicate is useless."
        )

    def build_task_prompt(self, ctx: AgentContext, recall: RecallResult) -> str:
        prompt = super().build_task_prompt(ctx, recall)
        return (
            f"{prompt}\n\nWeigh the positions above against each other and reach a "
            "conclusion. Name the disagreements before you resolve them."
        )


@register_agent
class PlannerAgent(Agent):
    """Turns conclusions into a concrete, sequenced plan."""

    name = "planner"
    role = "Planner"
    description = "Converts conclusions into sequenced, checkable next steps."
    entry_kind = EntryKind.PLAN
    default_salience = 0.6
    temperature = 0.4
    recall_policy = RecallPolicy(
        kinds=(
            EntryKind.SYNTHESIS,
            EntryKind.DECISION,
            EntryKind.INSIGHT,
            EntryKind.SUMMARY,
            EntryKind.NOTE,
        ),
        max_entries=10,
        token_budget=3000,
        use_query=True,
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Planner.\n"
            "You convert conclusions into action.\n"
            "- Produce at most six steps, ordered by dependency, not by importance.\n"
            "- Each step: the action, the artefact it produces, and how you would know "
            "it succeeded.\n"
            "- Mark any step that depends on an unresolved question with OPEN: and name "
            "the question.\n"
            "- Flag the step most likely to fail and say what you would do instead.\n"
            "No motivational language. A plan nobody can check is not a plan."
        )


@register_agent
class FactCheckAgent(Agent):
    """Audits stored claims for internal consistency and unsupported specifics."""

    name = "factcheck"
    role = "Fact Checker"
    description = "Audits claims in memory for contradictions and unsupported specifics."
    entry_kind = EntryKind.CRITIQUE
    default_salience = 0.6
    temperature = 0.1  # auditing wants determinism
    recall_policy = RecallPolicy(
        kinds=(EntryKind.RESEARCH, EntryKind.SYNTHESIS, EntryKind.INSIGHT),
        max_entries=14,
        token_budget=4000,
        use_query=True,
    )

    @property
    def system_prompt(self) -> str:
        return (
            "Fact Checker.\n"
            "You audit what is in memory. You do not add new analysis.\n"
            "For each issue emit one line:\n"
            "  VERDICT | citation | the claim | why\n"
            "where VERDICT is one of CONTRADICTION, UNSUPPORTED, IMPRECISE or STALE.\n"
            "- CONTRADICTION: two entries cannot both be true.\n"
            "- UNSUPPORTED: a specific number, name or date appears with no source.\n"
            "- IMPRECISE: a quantitative claim stated without magnitude or timeframe.\n"
            "- STALE: a claim that depends on a date or condition that may have moved.\n"
            "If memory is clean, say 'No issues found' and stop. Do not manufacture "
            "problems to look useful."
        )

    def task_instruction(self, ctx: AgentContext) -> str:
        return f"Audit the memory above for issues relating to: {ctx.task.strip() or ctx.topic}"
