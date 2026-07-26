"""UI smoke tests.

Streamlit's ``AppTest`` executes the real app script and reports any exception
raised during a rerun, which catches the failure mode that matters most here: a
view that crashes on empty state, or after a schema/API change elsewhere in the
codebase. Skipped when the ``ui`` extra is not installed.
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit", reason="requires the [ui] extra")

from streamlit.testing.v1 import AppTest  # noqa: E402

from athenacore.ui import theme  # noqa: E402

APP = "src/athenacore/ui/app.py"
VIEWS = ["Console", "Memory", "Timeline", "Runs", "Analytics"]


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENA_MODEL", "echo:test")
    monkeypatch.setenv("ATHENA_DATABASE_PATH", str(tmp_path / "ui.sqlite3"))
    monkeypatch.setenv("ATHENA_DATA_DIR", str(tmp_path))


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders_on_an_empty_database(view):
    """A brand-new install must not crash on any view."""
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception, [e.message for e in app.exception]
    app.sidebar.radio[0].set_value(view).run()
    assert not app.exception, [e.message for e in app.exception]


def test_a_full_run_completes_through_the_ui(tmp_path):
    # The Console shows an empty state until a topic exists, so seed one.
    from athenacore.memory.sqlite_store import SqliteMemoryStore

    store = SqliteMemoryStore(tmp_path / "ui.sqlite3")
    store.ensure_topic("scaling")
    store.close()

    app = AppTest.from_file(APP, default_timeout=180).run()
    app.text_area[0].set_value("Does compute scaling predict capability?").run()
    app.button[0].click().run()

    assert not app.exception, [e.message for e in app.exception]
    body = " ".join(m.value for m in app.markdown if m.value)
    assert "ac-tile" in body, "expected the result summary tiles"
    assert "flowchart" in body, "expected the live graph"


class TestTheme:
    def test_entry_card_escapes_untrusted_content(self):
        """Model output lands in an HTML string, so escaping is load-bearing."""
        html = theme.entry_card(
            {"agent": "a", "kind": "note", "content": "<script>alert(1)</script>"}
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_every_entry_kind_has_a_colour(self):
        from athenacore.memory.models import EntryKind

        for kind in EntryKind:
            assert kind.value in theme.KIND_COLOURS

    def test_activity_chart_handles_empty_input(self):
        assert "No activity" in theme.activity_chart([])

    def test_activity_chart_scales_to_the_peak(self):
        html = theme.activity_chart([("2026-01-01", 1), ("2026-01-02", 10)])
        assert "peak 10/day" in html
