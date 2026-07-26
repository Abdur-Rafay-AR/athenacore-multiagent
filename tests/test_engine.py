"""Config, providers, tools, agents, graphs and orchestration."""

from __future__ import annotations

import json

import pytest

from crucible.agents.base import AgentContext
from crucible.agents.registry import AgentFactory, available_agents, get_agent_class
from crucible.config import Settings, load_dotenv, split_model_spec
from crucible.errors import ConfigurationError, GraphError, ProviderError
from crucible.llm.base import Message, estimate_cost
from crucible.llm.providers import EchoProvider
from crucible.llm.registry import available_providers, build_provider
from crucible.memory.models import EntryKind
from crucible.memory.store import EntryFilter
from crucible.orchestration.debate import DebateOrchestrator
from crucible.orchestration.events import CancellationToken, EventBus, EventQueue, EventType
from crucible.orchestration.graph import AgentGraph, GraphNode
from crucible.orchestration.orchestrator import Orchestrator
from crucible.orchestration.presets import PRESETS, build_preset
from crucible.tools.base import ToolRegistry, parse_tool_calls, strip_tool_calls
from crucible.tools.builtin import CalculatorTool, ClockTool, MemorySearchTool

# -- config ------------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("ollama:llama3.1", ("ollama", "llama3.1")),
            ("llama3.1", ("ollama", "llama3.1")),
            ("openai:gpt-4o-mini", ("openai", "gpt-4o-mini")),
            ("  Anthropic:claude-x  ", ("anthropic", "claude-x")),
        ],
    )
    def test_split_model_spec(self, spec, expected):
        assert split_model_spec(spec) == expected

    @pytest.mark.parametrize("spec", ["", "   ", "openai:"])
    def test_split_model_spec_rejects_junk(self, spec):
        with pytest.raises(ConfigurationError):
            split_model_spec(spec)

    def test_defaults_survive_slots(self):
        """Slotted dataclasses hide class-level defaults behind descriptors."""
        settings = Settings.from_env(dotenv=None)
        assert isinstance(settings.log_level, str)
        assert isinstance(settings.model, str)

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("CRUCIBLE_MODEL", "openai:gpt-4o-mini")
        monkeypatch.setenv("CRUCIBLE_TEMPERATURE", "0.9")
        monkeypatch.setenv("CRUCIBLE_RECALL_MAX_ENTRIES", "3")
        settings = Settings.from_env(dotenv=None)
        assert settings.model == "openai:gpt-4o-mini"
        assert settings.temperature == 0.9
        assert settings.retrieval.max_entries == 3

    def test_invalid_env_is_reported(self, monkeypatch):
        monkeypatch.setenv("CRUCIBLE_MAX_OUTPUT_TOKENS", "not-a-number")
        with pytest.raises(ConfigurationError):
            Settings.from_env(dotenv=None)

    def test_validation_rejects_bad_values(self):
        with pytest.raises(ConfigurationError):
            Settings(temperature=9.0).validate()
        with pytest.raises(ConfigurationError):
            Settings(max_parallel_agents=0).validate()

    def test_unknown_override_is_rejected(self):
        with pytest.raises(ConfigurationError):
            Settings().with_overrides(not_a_setting=1)

    def test_secrets_are_redacted(self):
        redacted = Settings(openai_api_key="sk-secret").redacted()
        assert redacted["openai_api_key"] == "set"
        assert "sk-secret" not in json.dumps(redacted)

    def test_dotenv_parsing(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(
            '# comment\nexport CRUCIBLE_MODEL="echo:test"\nCRUCIBLE_TEMPERATURE=0.1\nbroken line\n',
            encoding="utf-8",
        )
        monkeypatch.delenv("CRUCIBLE_MODEL", raising=False)
        parsed = load_dotenv(env, override=True)
        assert parsed["CRUCIBLE_MODEL"] == "echo:test"
        assert parsed["CRUCIBLE_TEMPERATURE"] == "0.1"


# -- providers ---------------------------------------------------------------


class TestProviders:
    def test_echo_is_deterministic(self, provider):
        messages = [Message.system("Research Analyst"), Message.user("lithium refining")]
        assert provider.complete(messages).text == provider.complete(messages).text

    def test_usage_is_populated(self, provider):
        completion = provider.complete([Message.user("hello there")])
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0
        assert completion.usage.calls == 1

    def test_streaming_reassembles(self, provider):
        messages = [Message.user("stream this please")]
        assert "".join(provider.stream(messages)).strip() == provider.complete(messages).text

    def test_non_streaming_provider_still_streams(self, scripted):
        provider = scripted("only answer")
        assert "".join(provider.stream([Message.user("x")])) == "only answer"

    def test_registry_resolves_known_providers(self):
        assert "ollama" in available_providers()
        assert build_provider("echo:test", Settings()).name == "echo"

    def test_alias_uses_its_own_base_url(self):
        provider = build_provider("groq:llama-3.3-70b", Settings())
        assert "groq.com" in provider.base_url

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ConfigurationError):
            build_provider("nosuch:model", Settings())

    def test_retries_stop_at_the_limit(self):
        class Flaky(EchoProvider):
            attempts = 0

            def _complete(self, messages, **options):
                Flaky.attempts += 1
                raise ProviderError("boom", provider="flaky", retryable=True)

        provider = Flaky("x", max_retries=2, backoff_s=0.001)
        with pytest.raises(ProviderError):
            provider.complete([Message.user("hi")])
        assert Flaky.attempts == 3  # initial attempt plus two retries

    def test_non_retryable_errors_fail_fast(self):
        class Hard(EchoProvider):
            attempts = 0

            def _complete(self, messages, **options):
                Hard.attempts += 1
                raise ProviderError("bad key", provider="hard", retryable=False)

        with pytest.raises(ProviderError):
            Hard("x", max_retries=3, backoff_s=0.001).complete([Message.user("hi")])
        assert Hard.attempts == 1

    def test_cost_estimation(self):
        assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
        assert estimate_cost("some-local-model", 10_000, 10_000) == 0.0


# -- tools -------------------------------------------------------------------


class TestTools:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2 + 3 * 4", "14"),
            ("1.4e6 * 0.47 / 12", "54833"),
            ("sqrt(144)", "12"),
            ("round(3.14159, 2)", "3.14"),
            ("max(3, 7, 2)", "7"),
            ("-2 ** 3", "-8"),
        ],
    )
    def test_calculator_evaluates(self, expression, expected):
        assert expected in CalculatorTool().run(expression=expression)

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo hi')",  # code execution
            "open('/etc/passwd').read()",  # file access
            "9 ** 9 ** 9",  # resource exhaustion
            "1 / 0",
            "undefined_name + 1",
            "[1, 2, 3]",
        ],
    )
    def test_calculator_refuses_unsafe_input(self, expression):
        from crucible.errors import ToolError

        with pytest.raises(ToolError):
            CalculatorTool().run(expression=expression)

    def test_parse_tool_calls(self):
        text = (
            "Thinking about it.\n"
            'TOOL: calculator {"expression": "2+2"}\n'
            "TOOL: now\n"
            "Some trailing prose."
        )
        calls = parse_tool_calls(text)
        assert [c.name for c in calls] == ["calculator", "now"]
        assert calls[0].arguments == {"expression": "2+2"}

    @pytest.mark.parametrize(
        "line",
        [
            "TOOL: calculator {'expression': '2+2'}",  # single quotes
            'TOOL: calculator {"expression": "2+2",}',  # trailing comma
            'TOOL:calculator{"expression": "2+2"}',  # no spaces
        ],
    )
    def test_parse_tolerates_sloppy_syntax(self, line):
        calls = parse_tool_calls(line)
        assert len(calls) == 1
        assert calls[0].arguments["expression"] == "2+2"

    def test_strip_removes_calls(self):
        assert "TOOL:" not in strip_tool_calls("answer\nTOOL: now\nmore")

    def test_registry_reports_unknown_tools_as_results(self):
        registry = ToolRegistry([ClockTool()])
        results = registry.execute_all(parse_tool_calls("TOOL: nonexistent {}"))
        assert results[0].ok is False
        assert "unknown tool" in results[0].error

    def test_unsafe_tools_are_gated(self):
        class Dangerous(ClockTool):
            name = "dangerous"
            safe = False

        registry = ToolRegistry([Dangerous()], allow_unsafe=False)
        result = registry.execute_all(parse_tool_calls("TOOL: dangerous {}"))[0]
        assert result.ok is False
        assert "disabled by policy" in result.error

    def test_a_raising_tool_does_not_propagate(self):
        class Broken(ClockTool):
            name = "broken"

            def run(self, **kwargs):
                raise RuntimeError("kaboom")

        result = ToolRegistry([Broken()]).execute_all(parse_tool_calls("TOOL: broken {}"))[0]
        assert result.ok is False
        assert "kaboom" in result.error

    def test_memory_search_tool(self, seeded):
        output = MemorySearchTool(seeded, default_topic="lithium").run(query="water")
        assert "match" in output.lower()

    def test_prompt_section_documents_tools(self):
        section = ToolRegistry([CalculatorTool(), ClockTool()]).prompt_section()
        assert "calculator" in section and "TOOL:" in section


# -- graphs ------------------------------------------------------------------


class TestGraph:
    def test_levels_group_independent_nodes(self):
        graph = AgentGraph(
            [
                GraphNode(name="a", agent="research"),
                GraphNode(name="b", agent="critic", depends_on=("a",)),
                GraphNode(name="c", agent="insight", depends_on=("a",)),
                GraphNode(name="d", agent="synthesizer", depends_on=("b", "c")),
            ]
        )
        assert graph.levels() == [["a"], ["b", "c"], ["d"]]
        assert graph.max_width == 2

    def test_cycles_are_rejected(self):
        graph = AgentGraph(
            [
                GraphNode(name="a", agent="research", depends_on=("b",)),
                GraphNode(name="b", agent="critic", depends_on=("a",)),
            ]
        )
        with pytest.raises(GraphError):
            graph.validate()

    def test_unknown_dependencies_are_rejected(self):
        graph = AgentGraph([GraphNode(name="a", agent="research", depends_on=("ghost",))])
        with pytest.raises(GraphError):
            graph.validate()

    def test_self_dependency_is_rejected(self):
        graph = AgentGraph([GraphNode(name="a", agent="research", depends_on=("a",))])
        with pytest.raises(GraphError):
            graph.validate()

    def test_empty_graph_is_rejected(self):
        with pytest.raises(GraphError):
            AgentGraph().validate()

    def test_duplicate_names_are_rejected(self):
        graph = AgentGraph([GraphNode(name="a", agent="research")])
        with pytest.raises(GraphError):
            graph.add(GraphNode(name="a", agent="critic"))

    def test_descendants(self):
        graph = AgentGraph(
            [
                GraphNode(name="a", agent="research"),
                GraphNode(name="b", agent="critic", depends_on=("a",)),
                GraphNode(name="c", agent="insight", depends_on=("b",)),
                GraphNode(name="d", agent="planner"),
            ]
        )
        assert graph.descendants("a") == {"b", "c"}
        assert graph.descendants("d") == set()

    def test_builder_shorthand(self):
        graph = AgentGraph().then("research").then("critic").then("synthesizer")
        graph.validate()
        assert graph.order() == ["research", "critic", "synthesizer"]

    def test_json_round_trip(self):
        graph = build_preset("deep-dive")
        restored = AgentGraph.from_json(graph.to_json())
        assert restored.order() == graph.order()

    def test_mermaid_and_ascii_render(self):
        graph = build_preset("red-team")
        mermaid = graph.to_mermaid(states={"research": "succeeded"})
        assert "flowchart" in mermaid and "style" in mermaid
        assert "research" in graph.to_ascii()

    @pytest.mark.parametrize("key", sorted(PRESETS))
    def test_every_preset_is_valid(self, key):
        build_preset(key).validate()

    def test_solo_preset(self):
        graph = build_preset("solo:critic")
        assert graph.order() == ["critic"]

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(ConfigurationError):
            build_preset("does-not-exist")


# -- events ------------------------------------------------------------------


class TestEvents:
    def test_subscribe_and_unsubscribe(self):
        bus = EventBus()
        seen = []
        unsubscribe = bus.subscribe(seen.append)
        bus.publish(EventType.LOG, message="one")
        unsubscribe()
        bus.publish(EventType.LOG, message="two")
        assert [e.message for e in seen] == ["one"]

    def test_a_raising_subscriber_does_not_break_the_bus(self):
        bus = EventBus()
        seen = []

        def broken(event):
            raise RuntimeError("subscriber exploded")

        bus.subscribe(broken)
        bus.subscribe(seen.append)
        bus.publish(EventType.LOG, message="still delivered")
        assert len(seen) == 1

    def test_queue_drains(self):
        bus = EventBus()
        with EventQueue(bus) as events:
            bus.publish(EventType.LOG, message="a")
            bus.publish(EventType.LOG, message="b")
            assert [e.message for e in events.drain()] == ["a", "b"]
            assert events.empty

    def test_cancellation_token(self):
        from crucible.errors import RunCancelled

        token = CancellationToken()
        token.raise_if_cancelled()
        token.cancel()
        assert token.cancelled
        with pytest.raises(RunCancelled):
            token.raise_if_cancelled()


# -- agents ------------------------------------------------------------------


class TestAgents:
    def test_all_builtin_agents_are_registered(self):
        for name in [
            "research",
            "critic",
            "insight",
            "summarizer",
            "synthesizer",
            "planner",
            "factcheck",
        ]:
            assert name in available_agents()

    def test_unknown_agent_is_rejected(self):
        with pytest.raises(ConfigurationError):
            get_agent_class("nonexistent")

    def test_agent_writes_a_typed_entry(self, store, provider, settings):
        factory = AgentFactory(provider, store, settings=settings)
        result = factory.create("research").run(
            AgentContext(topic="t", task="What limits lithium refining?")
        )
        assert result.ok
        assert result.entry is not None
        assert result.entry.kind is EntryKind.RESEARCH
        assert store.get_entry(result.entry.id) is not None

    def test_agent_requiring_a_task_refuses_without_one(self, store, provider, settings):
        factory = AgentFactory(provider, store, settings=settings)
        result = factory.create("research").run(AgentContext(topic="t", task=""))
        assert not result.ok
        assert "requires a task" in result.error

    def test_provider_failure_becomes_a_result(self, store, settings):
        class Broken(EchoProvider):
            def _complete(self, messages, **options):
                raise ProviderError("model is down", provider="broken")

        factory = AgentFactory(Broken("x", max_retries=0), store, settings=settings)
        result = factory.create("critic").run(AgentContext(topic="t", task="anything"))
        assert not result.ok
        assert "model is down" in result.error

    def test_preamble_is_stripped(self, store, settings, scripted):
        provider = scripted("Sure, here is the answer:\nThe actual content.")
        factory = AgentFactory(provider, store, settings=settings)
        result = factory.create("insight").run(AgentContext(topic="t", task="x"))
        assert result.content.startswith("The actual content")

    def test_citations_are_resolved(self, seeded, settings, scripted):
        provider = scripted("Building on [1], the constraint is refining.")
        factory = AgentFactory(provider, seeded, settings=settings)
        result = factory.create("insight").run(
            AgentContext(topic="lithium", task="What is the constraint?")
        )
        assert result.citations, "expected [1] to resolve to a recalled entry id"
        assert seeded.get_entry(result.citations[0]) is not None

    def test_tool_loop_feeds_observations_back(self, store, settings):
        class ToolUser(EchoProvider):
            turn = 0

            def _complete(self, messages, **options):
                ToolUser.turn += 1
                if ToolUser.turn == 1:
                    return super()._complete(
                        [Message.user('TOOL: calculator {"expression": "6*7"}')]
                    )
                # The observation must have reached the model.
                assert any("OBSERVATION" in m.content for m in messages)
                return super()._complete([Message.user("final answer after tool use")])

        provider = ToolUser("x")
        provider._render = lambda messages: messages[-1].content  # echo the prompt verbatim
        factory = AgentFactory(
            provider, store, settings=settings, tools=ToolRegistry([CalculatorTool()])
        )
        result = factory.create("research").run(AgentContext(topic="t", task="what is 6*7?"))
        assert result.tool_results and result.tool_results[0].ok
        assert "42" in result.tool_results[0].output


# -- orchestration -----------------------------------------------------------


class TestOrchestrator:
    def test_runs_a_graph_and_persists_everything(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        report = orchestrator.run(
            build_preset("research-critique-synthesis", settings=settings),
            topic="energy",
            query="Is refining the real constraint?",
        )
        assert report.ok
        assert set(report.results) == {"research", "factcheck", "critic", "synthesizer"}
        assert all(r.ok for r in report.results.values())

        stored = store.get_run(report.run.id)
        assert stored is not None and stored.status.value == "succeeded"
        # The question plus one entry per agent.
        assert len(store.query_entries(EntryFilter(topic="energy", limit=50))) == 5

    def test_upstream_output_reaches_dependents(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        orchestrator.run(AgentGraph.linear("research", "critic"), topic="t", query="a question")
        critic_prompt = orchestrator.provider.calls[-1][-1].content
        assert "THIS ROUND" in critic_prompt

    def test_failure_skips_only_the_downstream_branch(self, store, settings):
        class FailingResearch(EchoProvider):
            def _complete(self, messages, **options):
                system = next((m.content for m in messages if m.role == "system"), "")
                if system.startswith("Research Analyst"):
                    raise ProviderError("research is down", provider="x")
                return super()._complete(messages, **options)

        orchestrator = Orchestrator(
            store, settings=settings, provider=FailingResearch("x", max_retries=0)
        )
        graph = AgentGraph(
            [
                GraphNode(name="research", agent="research"),
                GraphNode(name="critic", agent="critic", depends_on=("research",)),
                GraphNode(name="independent", agent="insight"),
            ]
        )
        report = orchestrator.run(graph, topic="t", query="q")

        assert report.run.status.value == "partial"
        assert "critic" in report.skipped  # downstream of the failure
        assert report.results["independent"].ok  # unrelated branch survived

    def test_optional_node_failure_does_not_skip_dependents(self, store, settings):
        class FailingCheck(EchoProvider):
            def _complete(self, messages, **options):
                system = next((m.content for m in messages if m.role == "system"), "")
                if system.startswith("Fact Checker"):
                    raise ProviderError("checker down", provider="x")
                return super()._complete(messages, **options)

        orchestrator = Orchestrator(
            store, settings=settings, provider=FailingCheck("x", max_retries=0)
        )
        report = orchestrator.run(
            build_preset("research-critique-synthesis", settings=settings), topic="t", query="q"
        )
        assert not report.results["factcheck"].ok
        assert report.results["synthesizer"].ok, "optional failures must not block the graph"

    def test_events_are_emitted_in_order(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        seen = []
        orchestrator.bus.subscribe(lambda e: seen.append(e.type))
        orchestrator.run(build_preset("brief", settings=settings), topic="t", query="q")
        assert seen[0] is EventType.RUN_STARTED
        assert seen[-1] is EventType.RUN_FINISHED
        assert EventType.NODE_FINISHED in seen
        assert EventType.MEMORY_WRITTEN in seen

    def test_cancellation_stops_the_run(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        token = CancellationToken()
        token.cancel()
        report = orchestrator.run(
            build_preset("deep-dive", settings=settings), topic="t", query="q", cancel=token
        )
        assert report.run.status.value == "cancelled"

    def test_memory_accumulates_across_runs(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        graph = build_preset("brief", settings=settings)
        orchestrator.run(graph, topic="ongoing", query="first question")
        first = store.get_topic("ongoing").entry_count

        orchestrator.run(graph, topic="ongoing", query="second question")
        assert store.get_topic("ongoing").entry_count > first

        # The second run must actually see the first run's output.
        last_prompt = orchestrator.provider.calls[-1][-1].content
        assert "PRIOR MEMORY" in last_prompt

    def test_run_agent_shortcut(self, store, settings):
        report = Orchestrator(store, settings=settings).run_agent(
            "critic", topic="t", query="challenge this"
        )
        assert report.ok
        assert "critic" in report.results

    def test_conditions_can_skip_nodes(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        orchestrator.register_condition("never", lambda report: False)
        graph = AgentGraph(
            [
                GraphNode(name="research", agent="research"),
                GraphNode(
                    name="planner", agent="planner", depends_on=("research",), condition="never"
                ),
            ]
        )
        report = orchestrator.run(graph, topic="t", query="q")
        assert "planner" in report.skipped


class TestDebate:
    def test_runs_rounds_and_adjudicates(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        debate = DebateOrchestrator(orchestrator, participants=["research", "critic"])
        report = debate.run(topic="t", query="Is X true?", rounds=2)
        assert 1 <= len(report.rounds) <= 2
        assert report.verdict is not None and report.verdict.ok
        assert report.output()

    def test_converges_early_when_positions_stop_moving(self, store, settings, scripted):
        # An unchanging provider means round two is identical to round one.
        provider = scripted("The position is unchanged and fully stated here.")
        provider.responses = ["The position is unchanged and fully stated here."]
        orchestrator = Orchestrator(store, settings=settings, provider=provider)
        debate = DebateOrchestrator(orchestrator, participants=["research", "critic"], judge=None)
        report = debate.run(topic="t", query="q", rounds=5)
        assert report.converged
        assert len(report.rounds) < 5

    def test_requires_two_participants(self, store, settings):
        with pytest.raises(ValueError):
            DebateOrchestrator(Orchestrator(store, settings=settings), participants=["research"])

    def test_later_rounds_demand_movement(self, store, settings):
        orchestrator = Orchestrator(store, settings=settings)
        debate = DebateOrchestrator(orchestrator, participants=["research", "critic"], judge=None)
        debate.run(topic="t", query="q", rounds=2)
        prompts = [call[-1].content for call in orchestrator.provider.calls]
        assert any("CONCEDE" in p and "SHARPEN" in p and "HOLD" in p for p in prompts)
