"""agent-memory-core adapter — wraps MemoryStore + WorkingMemory.

Stock mode: consolidate() is a no-op.
Tuned mode: consolidate() runs the v1 consolidator nightly.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from benchmark.amb_v2.adapters.base import Mode
from benchmark.amb_v2.chunks import Chunk

_log = logging.getLogger(__name__)

try:
    from agent_memory_core import MemoryStore, WorkingMemory
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

# AMB v2 scenarios use chunk types the core library doesn't recognize in
# VALID_TYPES. Without this map, add() raises ValueError and chunks vanish
# (the "Python"/"blue" retrieval gap diagnosed in v2.0.1).
_TYPE_MAP = {
    "preference": "personal",
    "update": "fact",
    "noise": "session",
}


class AgentMemoryCoreAdapter:
    def __init__(self, mode: Mode = "stock") -> None:
        if not _AVAILABLE:
            raise ImportError("agent-memory-core is not installed; run pip install -e .")
        self.mode: Mode = mode
        self._tmp = tempfile.mkdtemp(prefix="amb_v2_amc_")
        self._store = MemoryStore(db_path=self._tmp)
        wm_path = Path(self._tmp) / "working-memory.json"
        try:
            self._working = WorkingMemory(buffer_path=str(wm_path))
        except Exception as e:
            _log.debug("WorkingMemory init failed: %s", e)
            self._working = None

        # Fail loud if tuned mode is requested without a real consolidate().
        # Previously this silently no-oped, making tuned == stock a hidden bug.
        if mode == "tuned":
            consolidator = getattr(self._store, "consolidate", None)
            if not callable(consolidator):
                raise NotImplementedError(
                    "AgentMemoryCoreAdapter(mode='tuned') requires "
                    "MemoryStore.consolidate(); none was found on "
                    f"{type(self._store).__name__}. Upgrade agent-memory-core "
                    "or run with mode='stock'."
                )

    def __del__(self) -> None:
        if hasattr(self, "_tmp"):
            shutil.rmtree(self._tmp, ignore_errors=True)

    def ingest(self, day: int, chunks: list[Chunk]) -> None:
        for c in chunks:
            mapped_type = _TYPE_MAP.get(c.type, c.type)
            try:
                self._store.add(
                    c.text,
                    type=mapped_type,
                    source=f"d{day:03d}_{c.scenario_id}",
                )
            except ValueError as e:
                # Unmapped type — surface loudly so the gap is visible.
                _log.warning("ingest dropped chunk %s (type=%s→%s): %s",
                             c.id, c.type, mapped_type, e)
            except Exception as e:
                _log.warning("ingest chunk %s failed: %s", c.id, e)

    def consolidate(self, day: int) -> None:
        if self.mode == "stock":
            return
        # Tuned: existence of consolidate() was asserted at __init__.
        # Any failure here is a real bug — propagate.
        self._store.consolidate()

    def query(self, question: str, scenario_id: str) -> str:
        try:
            results = self._store.search(question, n=8)
        except Exception as e:
            _log.debug("query failed: %s", e)
            return ""
        if not results:
            return ""
        chunks = [r.text for r in results[:5]]
        return " | ".join(chunks)

    @property
    def metadata(self) -> dict:
        try:
            from agent_memory_core import __version__ as v
        except ImportError:
            v = "unknown"
        return {
            "name": "agent-memory-core",
            "version": v,
            "implements_consolidation": True,
            "mode": self.mode,
        }
