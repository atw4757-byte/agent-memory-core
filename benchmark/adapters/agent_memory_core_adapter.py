"""
benchmark/adapters/agent_memory_core_adapter.py — Full agent-memory-core system.

Uses all available features:
  - Salience scoring (type prior + access count + graph connectivity)
  - Adaptive retrieval (query-intent detection adjusts weights)
  - Cross-encoder reranking (if sentence-transformers is installed)
  - MMR diversity
  - Working memory for active session context

Dependencies: agent-memory-core (required), chromadb (required)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class AgentMemoryCoreFullAdapter:
    """agent-memory-core with all features enabled.

    Ingestion: user turns go in as 'session' type; assistant turns as 'session'
    too (capturing full conversation context). Working memory tracks the most
    recent active facts.

    Query: search top-8, reranked if available, MMR-diversified, top-5 returned.
    """

    def __init__(self) -> None:
        try:
            from agent_memory_core import MemoryStore, WorkingMemory
        except ImportError as e:
            raise ImportError(
                "agent-memory-core is required. Run: pip install -e . from project root."
            ) from e

        self._MemoryStore = MemoryStore
        self._WorkingMemory = WorkingMemory
        self._tmp: str | None = None
        self._store = None
        self._working = None
        self._init_store()

    def _init_store(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="amb_amc_")
        self._store = self._MemoryStore(db_path=self._tmp)
        # WorkingMemory uses a JSON file path, not a db_path
        wm_path = Path(self._tmp) / "working-memory.json"
        self._working = self._WorkingMemory(buffer_path=str(wm_path))

    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        if role == "user":
            self._store.add(
                content,
                type="session",
                source=f"session_{session_id}",
            )
            # Push into working memory's active_context for short-term recency
            try:
                self._working.add_context(content)
            except Exception:
                pass
        elif role == "assistant":
            # Assistant turns capture stated facts — useful for contradiction detection
            self._store.add(
                content,
                type="session",
                source=f"session_{session_id}_assistant",
            )

    def query(self, question: str) -> str:
        # Pull working memory context string for recent grounding
        working_ctx = ""
        try:
            working_ctx = self._working.as_query_context()
        except Exception:
            pass

        # Main retrieval: n=8, adaptive weights via query type detection
        results = self._store.search(question, n=8)
        if not results:
            return working_ctx if working_ctx else ""

        # Build answer from top-5 retrieved chunks + working memory grounding
        chunks = [r.text for r in results[:5]]
        answer = " | ".join(chunks)
        if working_ctx:
            answer = working_ctx + " | " + answer
        return answer

    def reset(self) -> None:
        if self._tmp:
            try:
                shutil.rmtree(self._tmp, ignore_errors=True)
            except Exception:
                pass
        self._init_store()
