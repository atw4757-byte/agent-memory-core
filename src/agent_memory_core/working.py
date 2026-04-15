"""
agent_memory_core.working — Short-term working memory buffer.

Extracted from archon-working-memory. Maintains a JSON buffer with:
  - current_goal: the active task
  - active_context: up to max_context_slots items (FIFO, oldest drops)
  - blockers: impediments to the current goal
  - next_actions: queued follow-up steps

The buffer persists to disk atomically. It is designed to be read at
session start so working state survives process restarts.

Typical use:
    wm = WorkingMemory()
    wm.set_goal("Build the divergence dataset")
    wm.add_context("RTX 4090 ordered, ETA April 15")
    wm.add_action("Run preflight check before first data run")
    print(wm.get())
    wm.flush(store)   # serialize to long-term MemoryStore, then clear
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .types import WorkingMemoryBuffer

if TYPE_CHECKING:
    from .store import MemoryStore


class WorkingMemory:
    """Short-term scratchpad that survives session restarts.

    Parameters
    ----------
    buffer_path:
        Path to the JSON buffer file. Defaults to
        ``~/.agent-memory-core/working-memory.json``.
    max_context_slots:
        Maximum items in ``active_context``. Oldest item drops when exceeded.
        Mirrors the Miller's Law 7±2 heuristic. Default: 7.
    max_list_items:
        Maximum items in ``blockers`` and ``next_actions``. Default: 20.
    """

    def __init__(
        self,
        buffer_path: Optional[str | Path] = None,
        max_context_slots: int = 7,
        max_list_items: int = 20,
    ) -> None:
        self._path = (
            Path(buffer_path)
            if buffer_path
            else Path.home() / ".agent-memory-core" / "working-memory.json"
        )
        self._max_context = max_context_slots
        self._max_list = max_list_items

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load buffer from disk. Returns empty buffer on missing file or parse error."""
        if not self._path.exists():
            return self._empty()
        try:
            data = json.loads(self._path.read_text())
            # Forward-compat: ensure all expected keys present
            empty = self._empty()
            for key, default in empty.items():
                if key not in data:
                    data[key] = copy.deepcopy(default)
            return data
        except (json.JSONDecodeError, OSError):
            return self._empty()

    def _save(self, buf: dict) -> None:
        """Write buffer atomically with updated timestamp."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        buf["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(buf, indent=2))
        tmp.replace(self._path)

    @staticmethod
    def _empty() -> dict:
        return {
            "current_goal": "",
            "active_context": [],
            "blockers": [],
            "next_actions": [],
            "updated_at": "",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self) -> WorkingMemoryBuffer:
        """Return the current buffer as a ``WorkingMemoryBuffer`` dataclass."""
        return WorkingMemoryBuffer.from_dict(self._load())

    def set_goal(self, goal: str) -> None:
        """Set the current working goal."""
        buf = self._load()
        buf["current_goal"] = goal
        self._save(buf)

    def add_context(self, item: str) -> Optional[str]:
        """Add a context item. Returns the dropped item if the buffer was full, else None."""
        buf = self._load()
        buf["active_context"].append(item)
        dropped = None
        if len(buf["active_context"]) > self._max_context:
            dropped = buf["active_context"].pop(0)
        self._save(buf)
        return dropped

    def add_blocker(self, item: str) -> Optional[str]:
        """Add a blocker. Returns the dropped item if at capacity, else None."""
        buf = self._load()
        buf["blockers"].append(item)
        dropped = None
        if len(buf["blockers"]) > self._max_list:
            dropped = buf["blockers"].pop(0)
        self._save(buf)
        return dropped

    def add_action(self, item: str) -> Optional[str]:
        """Add a next action. Returns the dropped item if at capacity, else None."""
        buf = self._load()
        buf["next_actions"].append(item)
        dropped = None
        if len(buf["next_actions"]) > self._max_list:
            dropped = buf["next_actions"].pop(0)
        self._save(buf)
        return dropped

    def clear(self) -> None:
        """Reset the buffer to an empty state."""
        self._save(self._empty())

    def flush(self, store: "MemoryStore") -> Optional[str]:
        """Serialize the buffer contents to long-term memory, then clear.

        All fields are joined into a single session memory chunk and stored
        with type ``"session"`` in the provided ``MemoryStore``.

        Returns the chunk ID if anything was flushed, else None.
        """
        buf = self._load()

        if not buf["current_goal"] and not buf["active_context"] and not buf["next_actions"]:
            return None

        lines = []
        if buf["current_goal"]:
            lines.append(f"Goal: {buf['current_goal']}")
        if buf["active_context"]:
            lines.append("Context: " + " | ".join(buf["active_context"]))
        if buf["blockers"]:
            lines.append("Blockers: " + " | ".join(buf["blockers"]))
        if buf["next_actions"]:
            lines.append("Next actions: " + " | ".join(buf["next_actions"]))

        timestamp = buf.get("updated_at") or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        text = f"[working-memory flush {timestamp}] " + " | ".join(lines)

        chunk_id = store.add(text, type="session", source="working-memory")
        self.clear()
        return chunk_id

    def as_query_context(self) -> str:
        """Return a compact string suitable for appending to search queries."""
        buf = self._load()
        parts = []
        if buf.get("current_goal"):
            parts.append(buf["current_goal"])
        context = buf.get("active_context", [])
        if context:
            parts.extend(context[:3])  # first 3 context items
        return " ".join(parts)
