"""Chunk — the smallest unit of memory ingested by an adapter (D data model)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChunkType = Literal["fact", "update", "noise", "credential", "preference", "session"]


@dataclass(frozen=True)
class Chunk:
    id: str
    scenario_id: str
    day: int
    text: str
    type: ChunkType
    supersedes: str | None = None
