"""UI styling and shared render helpers.

Kept apart from ``app.py`` so the page logic stays readable. Colours are defined
once, per entry kind, and reused by the entry cards, the timeline and the
charts — so an agent's colour means the same thing everywhere in the interface.
"""

from __future__ import annotations

import html
from typing import Any

# One hue per entry kind. Chosen for contrast against both Streamlit themes and
# to stay distinguishable in the most common form of colour blindness
# (deuteranopia): the palette varies lightness as well as hue, so the categories
# remain separable even when the reds and greens collapse together.
KIND_COLOURS: dict[str, str] = {
    "note": "#8b95a5",
    "question": "#6b7fd7",
    "research": "#2f7fd1",
    "summary": "#7b5ea7",
    "critique": "#d1603d",
    "insight": "#c9a227",
    "synthesis": "#2a9d8f",
    "plan": "#4a7c59",
    "decision": "#1f8a70",
    "tool": "#7d8597",
    "error": "#c1121f",
}

STATE_COLOURS = {
    "pending": "#8b95a5",
    "running": "#2f7fd1",
    "succeeded": "#2a9d8f",
    "failed": "#c1121f",
    "skipped": "#8b95a5",
}


def kind_colour(kind: str) -> str:
    return KIND_COLOURS.get(kind, "#8b95a5")


CSS = """
<style>
:root {
  --ac-radius: 10px;
  --ac-border: rgba(128, 138, 155, 0.28);
}

/* Entry card ------------------------------------------------------------- */
.ac-card {
  border: 1px solid var(--ac-border);
  border-left: 4px solid var(--ac-accent, #8b95a5);
  border-radius: var(--ac-radius);
  padding: 0.85rem 1rem;
  margin-bottom: 0.7rem;
  background: rgba(128, 138, 155, 0.05);
}
.ac-card.ac-archived { opacity: 0.55; }
.ac-card-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
  font-size: 0.82rem;
}
.ac-agent { font-weight: 700; }
.ac-chip {
  display: inline-block;
  padding: 0.08rem 0.5rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  color: #fff;
  background: var(--ac-accent, #8b95a5);
  white-space: nowrap;
}
.ac-meta { color: #8b95a5; font-size: 0.75rem; }
.ac-body { white-space: pre-wrap; line-height: 1.55; font-size: 0.93rem; }

/* Score bar -------------------------------------------------------------- */
.ac-signals { display: flex; gap: 0.35rem; margin-top: 0.5rem; flex-wrap: wrap; }
.ac-signal {
  font-size: 0.68rem;
  padding: 0.05rem 0.4rem;
  border-radius: 4px;
  border: 1px solid var(--ac-border);
  color: #8b95a5;
}

/* Stat tiles ------------------------------------------------------------- */
.ac-tiles { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
.ac-tile {
  flex: 1 1 120px;
  border: 1px solid var(--ac-border);
  border-radius: var(--ac-radius);
  padding: 0.65rem 0.8rem;
}
.ac-tile-value { font-size: 1.5rem; font-weight: 700; line-height: 1.15; }
.ac-tile-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #8b95a5;
}

/* Live trace ------------------------------------------------------------- */
.ac-trace {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.78rem;
  line-height: 1.65;
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid var(--ac-border);
  border-radius: var(--ac-radius);
  padding: 0.6rem 0.8rem;
}
.ac-trace-row { display: flex; gap: 0.5rem; }
.ac-trace-node { color: #2f7fd1; min-width: 8rem; }
.ac-trace-time { color: #8b95a5; min-width: 5rem; }

/* Misc ------------------------------------------------------------------- */
.ac-empty {
  text-align: center;
  padding: 2.5rem 1rem;
  color: #8b95a5;
  border: 1px dashed var(--ac-border);
  border-radius: var(--ac-radius);
}
.ac-bar-track {
  height: 8px;
  background: rgba(128, 138, 155, 0.2);
  border-radius: 999px;
  overflow: hidden;
}
.ac-bar-fill { height: 100%; border-radius: 999px; }
</style>
"""


def entry_card(
    entry: dict[str, Any], *, signals: dict[str, float] | None = None, score: float | None = None
) -> str:
    """Render one memory entry as HTML."""
    kind = entry.get("kind", "note")
    colour = kind_colour(kind)
    stamp = str(entry.get("created_at", ""))[:16].replace("T", " ")
    archived = " ac-archived" if entry.get("archived") else ""
    badge = " · archived" if entry.get("archived") else ""

    meta_bits = [stamp]
    if entry.get("model"):
        meta_bits.append(str(entry["model"]))
    if entry.get("tokens"):
        meta_bits.append(f"{entry['tokens']} tok")
    if score is not None:
        meta_bits.append(f"score {score:.3f}")

    signal_html = ""
    if signals:
        chips = "".join(
            f'<span class="ac-signal">{name} {value:.2f}</span>' for name, value in signals.items()
        )
        signal_html = f'<div class="ac-signals">{chips}</div>'

    return (
        f'<div class="ac-card{archived}" style="--ac-accent:{colour}">'
        f'<div class="ac-card-head">'
        f'<span class="ac-agent">{html.escape(str(entry.get("agent", "?")))}</span>'
        f'<span class="ac-chip">{html.escape(kind)}</span>'
        f'<span class="ac-meta">{html.escape(" · ".join(meta_bits))}{badge}</span>'
        f"</div>"
        f'<div class="ac-body">{html.escape(str(entry.get("content", "")))}</div>'
        f"{signal_html}"
        f"</div>"
    )


def stat_tiles(tiles: list[tuple[str, Any]]) -> str:
    """A row of headline numbers."""
    cells = "".join(
        f'<div class="ac-tile"><div class="ac-tile-value">{html.escape(str(value))}</div>'
        f'<div class="ac-tile-label">{html.escape(label)}</div></div>'
        for label, value in tiles
    )
    return f'<div class="ac-tiles">{cells}</div>'


def bar_row(label: str, value: int, maximum: int, colour: str) -> str:
    pct = 0 if maximum <= 0 else max(2, int(100 * value / maximum))
    return (
        f'<div style="margin-bottom:0.5rem">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.8rem">'
        f"<span>{html.escape(label)}</span><span>{value}</span></div>"
        f'<div class="ac-bar-track"><div class="ac-bar-fill" '
        f'style="width:{pct}%;background:{colour}"></div></div></div>'
    )


def trace_row(event: dict[str, Any]) -> str:
    symbols = {
        "run.started": "▶",
        "run.finished": "■",
        "run.failed": "✖",
        "node.started": "·",
        "node.finished": "✓",
        "node.failed": "✖",
        "node.skipped": "→",
        "memory.recalled": "🧠",
        "memory.written": "💾",
        "tool.called": "🔧",
        "round.started": "◆",
        "debate.converged": "≈",
        "memory.compacted": "▽",
    }
    symbol = symbols.get(event.get("type", ""), "•")
    node = event.get("node") or ""
    stamp = str(event.get("timestamp", ""))[11:19]
    return (
        f'<div class="ac-trace-row"><span class="ac-trace-time">{html.escape(stamp)}</span>'
        f"<span>{symbol}</span>"
        f'<span class="ac-trace-node">{html.escape(node)}</span>'
        f"<span>{html.escape(str(event.get('message', '')))}</span></div>"
    )


def activity_chart(activity: list[tuple[str, int]], *, height: int = 90) -> str:
    """A column chart of daily activity, rendered as plain HTML.

    Deliberately not ``st.bar_chart``: that pulls in pandas, which is a heavy
    dependency for one chart and — as seen on locked-down Windows hosts where
    pandas' compiled extensions are blocked — a single point of failure for the
    whole page. Divs always render.
    """
    if not activity:
        return empty_state("No activity recorded yet.")
    top = max(count for _, count in activity) or 1
    columns = []
    for day, count in activity:
        pct = max(3, int(100 * count / top))
        columns.append(
            f'<div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;'
            f'align-items:center;gap:2px" title="{html.escape(day)}: {count} entries">'
            f'<div style="width:100%;height:{pct}%;background:{KIND_COLOURS["research"]};'
            f'border-radius:3px 3px 0 0;min-height:3px"></div></div>'
        )
    first, last = activity[0][0], activity[-1][0]
    return (
        f'<div style="display:flex;gap:3px;align-items:flex-end;height:{height}px;'
        f'padding:0.4rem 0">{"".join(columns)}</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#8b95a5">'
        f"<span>{html.escape(first)}</span><span>peak {top}/day</span>"
        f"<span>{html.escape(last)}</span></div>"
    )


def empty_state(message: str, hint: str = "") -> str:
    hint_html = (
        f'<div style="font-size:0.85rem;margin-top:0.4rem">{html.escape(hint)}</div>'
        if hint
        else ""
    )
    return f'<div class="ac-empty"><div>{html.escape(message)}</div>{hint_html}</div>'
