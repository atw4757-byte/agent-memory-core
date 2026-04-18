"""T-02 — Query dataclass + JSON loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.amb_v2.queries import Query, load_queries


@pytest.fixture
def query_dict():
    return {
        "query_id": "q01",
        "scenario_id": "x",
        "question": "What is X?",
        "expected_answer": "Y",
        "reasoning_type": "factual",
        "difficulty": "easy",
        "trap": None,
        "checkpoint_eligibility": [0, 7, 30, 90],
        "resolution_type": "stable",
    }


def test_query_round_trip_json(tmp_path: Path, query_dict):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([query_dict]))
    qs = load_queries(p)
    assert len(qs) == 1
    q = qs[0]
    assert q.query_id == "q01"
    assert q.checkpoint_eligibility == frozenset({0, 7, 30, 90})
    assert q.resolution_type == "stable"


def test_checkpoint_eligibility_parses_set(tmp_path: Path, query_dict):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([query_dict]))
    q = load_queries(p)[0]
    assert isinstance(q.checkpoint_eligibility, frozenset)
    assert 7 in q.checkpoint_eligibility
    assert 14 not in q.checkpoint_eligibility


def test_load_queries_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_queries(tmp_path / "nope.json")


def test_load_queries_malformed_json_raises_clear_error(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError, match="malformed"):
        load_queries(p)


def test_load_queries_missing_required_field_raises(tmp_path: Path, query_dict):
    del query_dict["expected_answer"]
    p = tmp_path / "q.json"
    p.write_text(json.dumps([query_dict]))
    with pytest.raises(ValueError, match="expected_answer"):
        load_queries(p)
