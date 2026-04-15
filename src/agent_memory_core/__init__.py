"""
agent-memory-core — Agent memory that actually works.

Drop into any Python agent in 3 lines:

    from agent_memory_core import MemoryStore
    store = MemoryStore()
    store.add("The API key is in the keychain", type="credential")
    results = store.search("Where is the API key?")

Public API
----------
MemoryStore     — ChromaDB-backed store with salience scoring and adaptive retrieval
WorkingMemory   — Short-term scratchpad (4-7 slots) that survives session restarts
Consolidator    — Nightly lossy compression: many episodes -> stable facts via LLM
MemoryGraph     — Entity relationship graph with 2-hop neighbor expansion
ForgettingPolicy — Stale detection, soft/hard delete, health scoring
MemoryEval      — 30-query eval harness with recall/precision/answer metrics
"""

from .store import MemoryStore
from .working import WorkingMemory
from .consolidation import Consolidator
from .graph import MemoryGraph
from .forgetting import ForgettingPolicy
from .eval import MemoryEval

# Convenience re-exports from types
from .types import (
    MemoryResult,
    WorkingMemoryBuffer,
    GraphNode,
    EvalResult,
    VALID_TYPES,
    TYPE_TO_BANK,
)

__version__ = "0.1.0"
__all__ = [
    "MemoryStore",
    "WorkingMemory",
    "Consolidator",
    "MemoryGraph",
    "ForgettingPolicy",
    "MemoryEval",
    "MemoryResult",
    "WorkingMemoryBuffer",
    "GraphNode",
    "EvalResult",
    "VALID_TYPES",
    "TYPE_TO_BANK",
]
