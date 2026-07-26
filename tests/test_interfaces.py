"""CLI and HTTP API.

These are the surfaces users actually touch, so they are tested end to end
against a real temporary database rather than mocks.
"""

from __future__ import annotations

import json

import pytest

from athenacore.cli import main


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    """Point every command at an isolated database and the offline provider."""
    monkeypatch.setenv("ATHENA_MODEL", "echo:test")
    monkeypatch.setenv("ATHENA_DATABASE_PATH", str(tmp_path / "cli.sqlite3"))
    monkeypatch.setenv("ATHENA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ATHENA_LOG_LEVEL", "ERROR")
    return tmp_path


# -- CLI ---------------------------------------------------------------------


class TestCLI:
    def test_run_produces_output(self, capsys):
        assert main(["run", "Why is refining constrained?", "--topic", "t", "-p", "brief"]) == 0
        out = capsys.readouterr().out
        assert "research" in out
        assert "succeeded" in out

    def test_run_json_is_parseable(self, capsys):
        assert main(["run", "q", "--topic", "t", "-p", "solo:critic", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["run"]["status"] == "succeeded"
        assert "critic" in payload["results"]

    def test_topics_and_timeline(self, capsys):
        main(["run", "q", "--topic", "energy", "-p", "brief", "--json"])
        capsys.readouterr()

        assert main(["topics"]) == 0
        assert "energy" in capsys.readouterr().out

        assert main(["timeline", "--topic", "energy"]) == 0
        assert "Timeline" in capsys.readouterr().out

    def test_recall_explains_scores(self, capsys):
        main(["run", "lithium refining capacity", "--topic", "t", "-p", "brief", "--json"])
        capsys.readouterr()
        assert main(["recall", "refining", "--topic", "t", "--explain"]) == 0
        out = capsys.readouterr().out
        assert "score=" in out and "kw=" in out

    def test_stats_and_runs(self, capsys):
        main(["run", "q", "--topic", "t", "-p", "brief", "--json"])
        capsys.readouterr()
        assert main(["stats", "--json"]) == 0
        stats = json.loads(capsys.readouterr().out)
        assert stats["entries"] > 0

        assert main(["runs", "--json"]) == 0
        assert len(json.loads(capsys.readouterr().out)) == 1

    def test_export_markdown(self, capsys, tmp_path):
        main(["run", "q", "--topic", "t", "-p", "brief", "--json"])
        capsys.readouterr()
        target = tmp_path / "report.md"
        assert main(["export", "--topic", "t", "--format", "md", "-o", str(target)]) == 0
        text = target.read_text(encoding="utf-8")
        assert text.startswith("# t")
        assert "research" in text

    def test_export_of_unknown_topic_fails(self, capsys):
        assert main(["export", "--topic", "ghost"]) == 1

    def test_forget_deletes(self, capsys):
        main(["run", "q", "--topic", "doomed", "-p", "brief", "--json"])
        capsys.readouterr()
        assert main(["forget", "--topic", "doomed", "--yes"]) == 0
        capsys.readouterr()
        assert main(["topics", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == []

    def test_agents_and_presets(self, capsys):
        assert main(["agents", "--json"]) == 0
        agents = json.loads(capsys.readouterr().out)
        assert {a["name"] for a in agents} >= {"research", "critic", "synthesizer"}

        assert main(["presets", "--json"]) == 0
        presets = json.loads(capsys.readouterr().out)
        assert any(p["key"] == "red-team" for p in presets)

    def test_graph_rendering(self, capsys):
        assert main(["graph", "deep-dive", "--format", "mermaid"]) == 0
        assert "flowchart" in capsys.readouterr().out

    def test_debate(self, capsys):
        assert main(["debate", "Is X true?", "--topic", "t", "--rounds", "2", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["rounds"]
        assert payload["output"]

    def test_reindex(self, capsys):
        main(["run", "q", "--topic", "t", "-p", "brief", "--json"])
        capsys.readouterr()
        assert main(["reindex"]) == 0
        assert "Indexed" in capsys.readouterr().out

    def test_doctor(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "diagnostics" in out and "provider" in out

    def test_unknown_preset_exits_with_an_error(self, capsys):
        assert main(["run", "q", "--topic", "t", "-p", "nope"]) == 2
        assert "unknown preset" in capsys.readouterr().err

    def test_bad_model_spec_exits_with_an_error(self, capsys):
        assert main(["run", "q", "--topic", "t", "-m", "nosuch:model"]) == 2
        assert "unknown provider" in capsys.readouterr().err


# -- API ---------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="requires the [api] extra")


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from athenacore.api.server import create_app
    from athenacore.config import Settings

    app = create_app(
        Settings(
            model="echo:test",
            database_path=tmp_path / "api.sqlite3",
            data_dir=tmp_path,
            embedding_dims=64,
        )
    )
    with TestClient(app) as client:
        yield client


class TestAPI:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model"] == "echo:test"

    def test_config_redacts_secrets(self, client):
        assert "sk-" not in json.dumps(client.get("/config").json())

    def test_agents_and_presets(self, client):
        assert {a["name"] for a in client.get("/agents").json()} >= {"research", "critic"}
        assert any(p["key"] == "brief" for p in client.get("/presets").json())

    def test_run_creates_memory(self, client):
        response = client.post("/runs", json={"topic": "t", "query": "Why?", "preset": "brief"})
        assert response.status_code == 200
        body = response.json()
        assert body["run"]["status"] == "succeeded"
        assert body["output"]

        timeline = client.get("/topics/t/timeline").json()
        assert len(timeline) >= 2

    def test_unknown_preset_is_a_400(self, client):
        response = client.post("/runs", json={"topic": "t", "query": "q", "preset": "ghost"})
        assert response.status_code == 400

    def test_missing_topic_is_a_404(self, client):
        assert client.get("/topics/ghost").status_code == 404

    def test_recall_returns_signal_breakdown(self, client):
        client.post("/runs", json={"topic": "t", "query": "lithium refining", "preset": "brief"})
        body = client.get("/memory/recall", params={"topic": "t", "query": "refining"}).json()
        assert body["entries"]
        assert set(body["entries"][0]["signals"]) == {"keyword", "semantic", "recency", "salience"}

    def test_search(self, client):
        client.post("/runs", json={"topic": "t", "query": "lithium refining", "preset": "brief"})
        hits = client.get("/memory/search", params={"query": "refining"}).json()
        assert hits and "relevance" in hits[0]

    def test_patch_and_archive_entry(self, client):
        client.post("/runs", json={"topic": "t", "query": "q", "preset": "brief"})
        entry_id = client.get("/topics/t/timeline").json()[0]["id"]

        patched = client.patch(f"/memory/entries/{entry_id}", json={"salience": 0.9}).json()
        assert patched["salience"] == pytest.approx(0.9)

        assert client.post(f"/memory/entries/{entry_id}/archive").json()["archived"] == 1
        assert client.patch("/memory/entries/ghost", json={"salience": 0.5}).status_code == 404

    def test_topic_lifecycle(self, client):
        assert client.post("/topics", json={"name": "fresh", "description": "d"}).status_code == 201
        assert client.get("/topics/fresh").json()["topic"]["description"] == "d"
        assert client.delete("/topics/fresh").status_code == 200
        assert client.get("/topics/fresh").status_code == 404

    def test_stream_emits_lifecycle_events(self, client):
        with client.stream(
            "POST", "/runs/stream", json={"topic": "t", "query": "q", "preset": "solo:critic"}
        ) as response:
            events = [
                line.split(": ", 1)[1]
                for line in response.iter_lines()
                if line.startswith("event:")
            ]
        assert events[0] == "run.started"
        assert "node.finished" in events
        assert events[-1] == "report"

    def test_debate_endpoint(self, client):
        body = client.post(
            "/debates", json={"topic": "t", "query": "Is X true?", "rounds": 2}
        ).json()
        assert body["rounds"]
        assert body["output"]

    def test_stats_includes_activity(self, client):
        client.post("/runs", json={"topic": "t", "query": "q", "preset": "brief"})
        stats = client.get("/stats").json()
        assert stats["entries"] > 0
        assert stats["activity"]
