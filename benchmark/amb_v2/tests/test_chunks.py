"""T-01 — Chunk dataclass."""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from benchmark.amb_v2.chunks import Chunk


def test_chunk_is_frozen():
    c = Chunk(id="x-d000-0", scenario_id="x", day=0, text="t", type="fact")
    with pytest.raises(FrozenInstanceError):
        c.text = "mutated"  # type: ignore[misc]


def test_chunk_supersedes_defaults_none():
    c = Chunk(id="x-d000-0", scenario_id="x", day=0, text="t", type="fact")
    assert c.supersedes is None


def test_chunk_supersedes_can_be_set():
    c = Chunk(
        id="x-d001-0", scenario_id="x", day=1, text="t2", type="update",
        supersedes="x-d000-0",
    )
    assert c.supersedes == "x-d000-0"


def test_chunk_id_required():
    with pytest.raises(TypeError):
        Chunk(scenario_id="x", day=0, text="t", type="fact")  # type: ignore[call-arg]


def test_chunk_all_valid_types():
    for t in ("fact", "update", "noise", "credential", "preference", "session"):
        Chunk(id=f"x-d000-{t}", scenario_id="x", day=0, text="t", type=t)
