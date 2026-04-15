"""
benchmark/adapters/full_context.py — Upper-bound "oracle" adapter.

Stores ALL conversation turns and returns ALL of them as context on every
query. This is the theoretical ceiling — perfect recall at O(n) cost.

Useful for:
  - Establishing the maximum achievable score on the benchmark
  - Measuring how much a real system loses to retrieval errors
  - Sanity-checking that the benchmark scores make sense

No dependencies beyond the standard library.
"""

from __future__ import annotations


class FullContextAdapter:
    """Store every turn; return everything on every query.

    Perfect recall, terrible efficiency. O(n) memory, O(n) query latency.
    This is the upper bound — no real system should use this approach.
    """

    def __init__(self) -> None:
        self._turns: list[str] = []

    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        # Store all turns, all roles — nothing is filtered
        self._turns.append(content)

    def query(self, question: str) -> str:
        if not self._turns:
            return ""
        # Return everything. Separator chosen to match other adapters for fair scoring.
        return " | ".join(self._turns)

    def reset(self) -> None:
        self._turns = []
