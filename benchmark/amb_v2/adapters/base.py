"""DecayAdapter Protocol — the single extension point for v2.

Stock vs Tuned mode (D10):
  - mode="stock"  : consolidate() MUST be a no-op
  - mode="tuned"  : consolidate() MAY do work (e.g. nightly cleanup)

metadata.implements_consolidation MUST be truthful — lying disqualifies
leaderboard runs.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from benchmark.amb_v2.chunks import Chunk

Mode = Literal["stock", "tuned"]


@runtime_checkable
class DecayAdapter(Protocol):
    mode: Mode

    def ingest(self, day: int, chunks: list[Chunk]) -> None: ...
    def consolidate(self, day: int) -> None: ...
    def query(self, question: str, scenario_id: str) -> str: ...

    @property
    def metadata(self) -> dict: ...


def split_returned_chunks(answer: str, sep: str = " | ") -> tuple[str, ...]:
    """Heuristic split of an adapter's concatenated-context answer back into
    its constituent chunks. Used by the harness when the adapter doesn't
    expose an ordered-chunks accessor. Safe even if the adapter returns a
    single blob — the result is just a 1-tuple.
    """
    if not answer:
        return ()
    return tuple(p for p in answer.split(sep) if p)


METADATA_REQUIRED_KEYS = frozenset({"name", "version", "implements_consolidation"})


def validate_metadata(meta: dict) -> None:
    """Raise if metadata is missing required keys or has wrong types."""
    missing = METADATA_REQUIRED_KEYS - meta.keys()
    if missing:
        raise ValueError(f"adapter metadata missing keys: {sorted(missing)}")
    if not isinstance(meta["name"], str) or not meta["name"]:
        raise ValueError("metadata.name must be a non-empty string")
    if not isinstance(meta["version"], str) or not meta["version"]:
        raise ValueError("metadata.version must be a non-empty string")
    if not isinstance(meta["implements_consolidation"], bool):
        raise ValueError("metadata.implements_consolidation must be bool")
