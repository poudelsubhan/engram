"""Rich terminal observability.

This is a window onto the engine, not the product. Left panel: the live event
feed. Right panel: the memory population. On `contamination_traced` the screen
gives way to a red infection tree — the one frame worth pausing on.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from . import events, store
from . import trust as T

console = Console()

STATUS_STYLE = {
    T.PROVISIONAL: "yellow",
    T.TRUSTED: "bold green",
    T.QUARANTINED: "bold red",
    T.DEAD: "dim",
}

EVENT_STYLE = {
    events.MEMORY_WRITTEN: "green",
    events.MEMORY_MERGED: "cyan",
    events.MEMORY_RETRIEVED: "blue",
    events.MEMORY_CITED: "bold blue",
    events.TRUST_UPDATED: "magenta",
    events.STATUS_CHANGED: "bold yellow",
    events.EPISODE_START: "bold white",
    events.EPISODE_END: "bold white",
    events.CONTAMINATION_TRACED: "bold red",
    events.CONTRADICTION_DETECTED: "bold magenta",
    events.CONTRADICTION_SUPPRESSED: "bold magenta",
    events.DECAY_TICK: "dim",
    events.NOTE: "dim",
}


def trust_bar(trust: float, width: int = 10) -> Text:
    filled = max(0, min(width, round(float(trust) * width)))
    colour = "green" if trust >= 0.6 else "yellow" if trust >= 0.15 else "red"
    bar = Text("█" * filled, style=colour)
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {float(trust):.2f}", style="white")
    return bar


def truncate(text: str, width: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def memory_table(docs: list[dict[str, Any]] | None = None, width: int = 58) -> Table:
    docs = store.all_memories() if docs is None else docs
    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("mid", style="bold", width=7)
    table.add_column("claim", width=width, overflow="ellipsis")
    table.add_column("status", width=12)
    table.add_column("trust", width=16)
    table.add_column("W/L", width=6, justify="right")

    for doc in docs:
        status = doc.get("status", T.PROVISIONAL)
        label = Text(status, style=STATUS_STYLE.get(status, "white"))
        if doc.get("contaminated_by"):
            label.append(" ☣", style="bold red")
        if doc.get("contradicts"):
            label.append(" ⚔", style="bold magenta")
        table.add_row(
            doc.get("mid", "?"),
            truncate(doc.get("text", ""), width),
            label,
            trust_bar(doc.get("trust", 0.0)),
            f"{doc.get('wins', 0)}/{doc.get('losses', 0)}",
        )
    if not docs:
        table.add_row("—", Text("no memories yet", style="dim"), "", "", "")
    return table


def format_event(event: dict[str, Any]) -> Text:
    kind = event.get("kind", "?")
    line = Text()
    line.append(f"{kind:<24}", style=EVENT_STYLE.get(kind, "white"))

    if kind in (events.MEMORY_WRITTEN, events.MEMORY_MERGED):
        line.append(f"{event.get('mid','')} ", style="bold")
        parents = event.get("parents") or event.get("parents_added") or []
        if parents:
            line.append(f"←{','.join(parents)} ", style="dim")
        line.append(truncate(event.get("text", ""), 52), style="dim")
    elif kind == events.MEMORY_RETRIEVED:
        line.append(", ".join(event.get("mids", [])), style="bold")
    elif kind == events.MEMORY_CITED:
        line.append(", ".join(event.get("mids", [])), style="bold blue")
    elif kind == events.TRUST_UPDATED:
        arrow = "↑" if event.get("trust_after", 0) >= event.get("trust_before", 0) else "↓"
        line.append(f"{event.get('mid','')} ", style="bold")
        line.append(
            f"{event.get('trust_before')} {arrow} {event.get('trust_after')} "
            f"({event.get('reason','')})"
        )
    elif kind == events.STATUS_CHANGED:
        line.append(f"{event.get('mid','')} ", style="bold")
        line.append(f"{event.get('before')} → ", style="dim")
        line.append(str(event.get("after")),
                    style=STATUS_STYLE.get(event.get("after"), "white"))
        if event.get("cause"):
            line.append(f"  [{event['cause']}]", style="red")
    elif kind == events.EPISODE_START:
        line.append(str(event.get("task_id", "")), style="bold")
    elif kind == events.EPISODE_END:
        outcome = event.get("outcome", "")
        line.append(f"{event.get('task_id','')} ", style="bold")
        line.append(outcome.upper(), style="bold green" if outcome == "pass" else "bold red")
        line.append(f"  {event.get('latency_ms','?')}ms", style="dim")
        if event.get("cited"):
            line.append(f"  cited {','.join(event['cited'])}", style="blue")
    elif kind == events.CONTAMINATION_TRACED:
        line.append(f"{event.get('root','')} infected {event.get('count',0)} descendant(s)",
                    style="bold red")
    elif kind == events.CONTRADICTION_SUPPRESSED:
        line.append(f"kept {event.get('kept')} over {event.get('suppressed')}",
                    style="magenta")
    elif kind == events.CONTRADICTION_DETECTED:
        line.append(f"{event.get('mid')} ⚔ {','.join(event.get('against', []))}",
                    style="magenta")
    else:
        line.append(truncate(str(event.get("note", "")), 60), style="dim")
    return line


def contamination_tree(event: dict[str, Any]) -> Tree:
    """The money frame: the lie at the root, everything it taught underneath."""
    root_label = Text("☣ ", style="bold red")
    root_label.append(f"{event.get('root','')} ", style="bold red")
    root_label.append("QUARANTINED", style="bold white on red")
    root_label.append(f"\n   {truncate(event.get('root_text',''), 72)}", style="red")
    tree = Tree(root_label, guide_style="red")

    nodes: dict[str, Tree] = {event.get("root", ""): tree}
    for node in sorted(event.get("subtree", []), key=lambda n: int(n.get("depth", 0))):
        label = Text(f"{node['mid']} ", style="bold red")
        label.append(f"trust {node['trust_before']} → {node['trust_after']}  ", style="yellow")
        label.append(f"{node['status_before']} → {node['status_after']}", style="red")
        label.append(f"\n{truncate(node.get('text',''), 72)}", style="dim red")
        parent = next(
            (nodes[p] for p in (node.get("parents") or []) if p in nodes), tree
        )
        nodes[node["mid"]] = parent.add(label)
    return tree


def show_contamination(event: dict[str, Any], hold: float = 3.0) -> None:
    console.print()
    console.print(
        Panel(
            contamination_tree(event),
            title="[bold white on red] CONTAMINATION TRACED — $graphLookup [/]",
            subtitle=f"[red]{event.get('count', 0)} memory(s) learned under a lie[/]",
            border_style="red",
        )
    )
    time.sleep(hold)


class Dashboard:
    """Live two-panel view, refreshed on every event the engine emits."""

    def __init__(self, title: str, feed_lines: int = 22) -> None:
        self.title = title
        self.feed_lines = feed_lines
        self.lines: list[Text] = []
        self.pending_contamination: list[dict[str, Any]] = []
        self.live: Live | None = None

    def _layout(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(
                Panel(
                    Group(*self.lines[-self.feed_lines:]) if self.lines
                    else Text("waiting…", style="dim"),
                    title="[bold]event feed[/]",
                    border_style="blue",
                ),
                name="feed",
                ratio=1,
            ),
            Layout(
                Panel(
                    memory_table(),
                    title="[bold]memories[/]",
                    subtitle="[dim]green trusted · yellow provisional · red quarantined[/]",
                    border_style="green",
                ),
                name="memories",
                ratio=1,
            ),
        )
        root = Layout()
        root.split_column(
            Layout(Panel(Text(self.title, style="bold white"), border_style="white"),
                   size=3),
            Layout(layout),
        )
        return root

    def on_event(self, event: dict[str, Any]) -> None:
        self.lines.append(format_event(event))
        if event.get("kind") == events.CONTAMINATION_TRACED:
            self.pending_contamination.append(event)
        if self.live:
            self.live.update(self._layout())

    def __enter__(self) -> "Dashboard":
        events.bus().subscribe(self.on_event)
        self.live = Live(self._layout(), console=console, refresh_per_second=6,
                         screen=False)
        self.live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.live:
            self.live.__exit__(*exc)
            self.live = None
        for event in self.pending_contamination:
            show_contamination(event)
        self.pending_contamination.clear()
