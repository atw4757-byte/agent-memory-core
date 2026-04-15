"""
agent_memory_core.types — Shared dataclasses and type aliases.

All modules import from here to avoid circular dependencies and keep
the public API surface clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Valid chunk types (mirrors archon-memory's TYPE_TO_BANK mapping)
# ---------------------------------------------------------------------------

VALID_TYPES = frozenset([
    "fact",
    "personal",
    "professional",
    "credential",
    "financial",
    "goal",
    "project_status",
    "technical",
    "session",
    "task",
    "observation",
    "dream",
    "lesson",
])

# Map chunk types to Hindsight bank IDs
TYPE_TO_BANK: dict[str, str] = {
    "fact": "core",
    "personal": "core",
    "professional": "core",
    "credential": "core",
    "financial": "core",
    "goal": "core",
    "project_status": "projects",
    "technical": "projects",
    "session": "sessions",
    "task": "sessions",
    "observation": "health",
    "dream": "dreams",
    "lesson": "lessons",
}

# Type-based salience priors (0-1). Higher = more inherently important.
TYPE_SALIENCE_PRIORS: dict[str, float] = {
    "credential":     0.8,
    "lesson":         0.7,
    "financial":      0.7,
    "goal":           0.6,
    "personal":       0.6,
    "professional":   0.5,
    "project_status": 0.5,
    "technical":      0.5,
    "fact":           0.5,
    "task":           0.4,
    "session":        0.3,
    "observation":    0.3,
    "dream":          0.2,
}

# Temporal decay rates per type. 0.0 = never decays, 0.025 = ~10 day half-life.
DECAY_RATES: dict[str, float] = {
    "credential":     0.0,    # never decays
    "lesson":         0.0,    # never decays
    "goal":           0.001,  # very slow (years)
    "personal":       0.002,
    "professional":   0.002,
    "financial":      0.002,
    "dream":          0.01,
    "session":        0.015,
    "observation":    0.01,
    "project_status": 0.02,
    "technical":      0.02,
    "task":           0.025,
    "fact":           0.005,
}


# ---------------------------------------------------------------------------
# MemoryResult — returned by MemoryStore.search()
# ---------------------------------------------------------------------------

@dataclass
class MemoryResult:
    """A single search result from the memory store."""

    id: str
    text: str
    type: str
    source: str
    date: str
    distance: float
    recency_score: float
    salience: float
    combined_score: float
    age_days: int
    metadata: dict[str, Any] = field(default_factory=dict)
    ce_score: Optional[float] = None   # cross-encoder score, if reranker ran
    has_fact: bool = False             # True when an atomic fact was merged in

    def __repr__(self) -> str:  # noqa: D105
        preview = self.text[:80].replace("\n", " ")
        return (
            f"MemoryResult(type={self.type!r}, score={self.combined_score:.3f}, "
            f"age={self.age_days}d, text={preview!r})"
        )


# ---------------------------------------------------------------------------
# WorkingMemoryBuffer — holds the in-memory state of the working buffer
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemoryBuffer:
    """Snapshot of the working memory buffer."""

    current_goal: str = ""
    active_context: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "current_goal": self.current_goal,
            "active_context": self.active_context,
            "blockers": self.blockers,
            "next_actions": self.next_actions,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkingMemoryBuffer:
        return cls(
            current_goal=data.get("current_goal", ""),
            active_context=data.get("active_context", []),
            blockers=data.get("blockers", []),
            next_actions=data.get("next_actions", []),
            updated_at=data.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# GraphNode — one node in the MemoryGraph
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A node in the entity relationship graph."""

    id: str
    source_file: str
    title: str
    summary: str
    node_type: str        # EXTRACTED | INFERRED
    domain: str
    confidence: float
    last_modified: str
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EvalResult — one query's result in the eval harness
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Scoring result for a single eval query."""

    query: str
    query_type: str
    recall: bool
    precision: float
    answer: bool
    results_count: int

    @property
    def composite(self) -> float:
        """Per-query composite (for aggregation convenience)."""
        return 0.4 * (1.0 if self.recall else 0.0) + 0.3 * self.precision + 0.3 * (1.0 if self.answer else 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_recency(date_str: str, chunk_type: str) -> float:
    """Return a 0-1 recency score. 1.0 = today, decays per type."""
    decay_rate = DECAY_RATES.get(chunk_type, 0.01)
    if not date_str:
        return math.exp(-decay_rate * 30)  # assume 30 days if unknown
    try:
        mem_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
        age_days = max((datetime.now() - mem_date).days, 0)
        return math.exp(-decay_rate * age_days)
    except (ValueError, TypeError):
        return math.exp(-decay_rate * 30)


def compute_salience(chunk_type: str, metadata: dict, graph_connectivity: dict | None = None) -> float:
    """Compute a 0-1 salience score.

    Components:
      - type prior                 (base weight from TYPE_SALIENCE_PRIORS)
      - access count boost         +min(access_count / 10, 0.2)
      - graph connectivity boost   +min(connections / 10, 0.2)

    Final value capped at 1.0.
    """
    score = TYPE_SALIENCE_PRIORS.get(chunk_type, 0.5)

    access_count = int(metadata.get("access_count", 0))
    score += min(access_count / 10.0, 0.2)

    if graph_connectivity:
        source_file = metadata.get("source_file", "")
        connections = graph_connectivity.get(source_file, 0)
        score += min(connections / 10.0, 0.2)

    return min(score, 1.0)
