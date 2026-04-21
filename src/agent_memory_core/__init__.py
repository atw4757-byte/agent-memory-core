"""ChromaDB-backed agent memory with salience scoring and adaptive retrieval."""

from .store import MemoryStore, CE_THRESHOLDS, detect_query_type
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

__version__ = "0.1.2"
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
    "CE_THRESHOLDS",
    "detect_query_type",
]
