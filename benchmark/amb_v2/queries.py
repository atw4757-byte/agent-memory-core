"""Query — what we ask the adapter at each checkpoint."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ResolutionType = Literal["stable", "contradiction", "aggregation", "trajectory"]
Difficulty = Literal["easy", "medium", "hard"]

REQUIRED_FIELDS = (
    "query_id", "scenario_id", "question", "expected_answer", "reasoning_type",
    "difficulty", "checkpoint_eligibility", "resolution_type",
)


@dataclass(frozen=True)
class Query:
    query_id: str
    scenario_id: str
    question: str
    expected_answer: str
    reasoning_type: str
    difficulty: Difficulty
    trap: str | None
    checkpoint_eligibility: frozenset[int]
    resolution_type: ResolutionType


def load_queries(path: Path | str) -> list[Query]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"queries file not found: {p}")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON in {p}: {e}") from e
    out: list[Query] = []
    for i, item in enumerate(raw):
        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"queries[{i}] missing required field: {field}")
        out.append(Query(
            query_id=item["query_id"],
            scenario_id=item["scenario_id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            reasoning_type=item["reasoning_type"],
            difficulty=item["difficulty"],
            trap=item.get("trap"),
            checkpoint_eligibility=frozenset(item["checkpoint_eligibility"]),
            resolution_type=item["resolution_type"],
        ))
    return out
