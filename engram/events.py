"""Append-only event log.

Every state change in Engram is emitted here: to `runs/{ts}.jsonl` on disk
(demo-replay insurance) and to an in-memory deque the TUI drains. Nothing in
the engine mutates a memory without saying so out loud.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque

# Canonical event names — the TUI and the JSONL replay both key off these.
MEMORY_WRITTEN = "memory_written"
MEMORY_MERGED = "memory_merged"
MEMORY_RETRIEVED = "memory_retrieved"
MEMORY_CITED = "memory_cited"
TRUST_UPDATED = "trust_updated"
STATUS_CHANGED = "status_changed"
EPISODE_START = "episode_start"
EPISODE_END = "episode_end"
CONTAMINATION_TRACED = "contamination_traced"
CONTRADICTION_DETECTED = "contradiction_detected"
CONTRADICTION_SUPPRESSED = "contradiction_suppressed"
DECAY_TICK = "decay_tick"
NOTE = "note"

RUNS_DIR = Path(os.environ.get("ENGRAM_RUNS_DIR", "runs"))


class EventBus:
    """Thread-safe fan-out to disk, memory, and any live subscribers."""

    def __init__(self, run_id: str | None = None, maxlen: int = 500) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.recent: Deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            self._path = RUNS_DIR / f"{self.run_id}.jsonl"
        return self._path

    def subscribe(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "kind": kind,
            **fields,
        }
        with self._lock:
            self.recent.append(event)
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            except OSError:
                pass  # never let observability break the engine
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception:
                pass
        return event

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.recent)


_BUS: EventBus | None = None


def bus() -> EventBus:
    """Process-wide default bus."""
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS


def set_bus(new_bus: EventBus) -> EventBus:
    global _BUS
    _BUS = new_bus
    return _BUS


def emit(kind: str, **fields: Any) -> dict[str, Any]:
    return bus().emit(kind, **fields)
