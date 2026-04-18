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

    def __del__(self) -> None:
        if hasattr(self, "_tmp"):
            shutil.rmtree(self._tmp, ignore_errors=True)

    def ingest(self, day: int, chunks: list[Chunk]) -> None:
        for c in chunks:
            try:
                self._store.add(
                    c.text,
                    type=c.type if c.type != "noise" else "session",
                    source=f"d{day:03d}_{c.scenario_id}",
                )
            except Exception as e:
                _log.debug("ingest chunk %s failed: %s", c.id, e)

    def consolidate(self, day: int) -> None:
        if self.mode == "stock":
            return
        # Tuned: trigger consolidation if available
        try:
            consolidator = getattr(self._store, "consolidate", None)
            if callable(consolidator):
                consolidator()
        except Exception as e:
            _log.debug("consolidate day=%d failed: %s", day, e)

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
