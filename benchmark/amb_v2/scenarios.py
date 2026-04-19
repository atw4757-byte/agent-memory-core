"""Scenario loaders — public JSON + held-out age-encrypted JSON."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.amb_v2.queries import Query


@dataclass(frozen=True)
class ScenarioBundle:
    scenario_id: str
    name: str
    timeline: list[dict[str, Any]]   # list of timeline events: {day, type, id, text, supersedes?}
    queries: list[Query]
    is_held_out: bool
    # v2.3: per-query confuser chunks. Shape: {query_id: [{"day": int, "text": str}, ...]}
    # Confusers share vocabulary with the query but contain wrong/no answer.
    # Injected into timeline by the simulator at the specified days.
    confusers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _bundle_from_obj(obj: dict[str, Any], *, is_held_out: bool) -> ScenarioBundle:
    scenario_id = obj["scenario_id"]
    queries = [
        Query(
            query_id=q["query_id"],
            scenario_id=q.get("scenario_id", scenario_id),
            question=q["question"],
            expected_answer=q["expected_answer"],
            reasoning_type=q.get("reasoning_type", "factual"),
            difficulty=q.get("difficulty", "medium"),
            trap=q.get("trap"),
            checkpoint_eligibility=frozenset(q["checkpoint_eligibility"]),
            resolution_type=q["resolution_type"],
        )
        for q in obj["queries"]
    ]
    confusers: dict[str, list[dict[str, Any]]] = {}
    for qid, items in obj.get("confusers", {}).items():
        confusers[qid] = [dict(x) for x in items]
    return ScenarioBundle(
        scenario_id=obj["scenario_id"],
        name=obj["name"],
        timeline=list(obj["timeline"]),
        queries=queries,
        is_held_out=is_held_out,
        confusers=confusers,
    )


def load_public_scenarios(directory: Path | str) -> list[ScenarioBundle]:
    d = Path(directory)
    if not d.exists():
        return []
    out: list[ScenarioBundle] = []
    for path in sorted(d.glob("*.json")):
        obj = json.loads(path.read_text())
        out.append(_bundle_from_obj(obj, is_held_out=False))
    return out


def load_held_out(
    directory: Path | str,
    *,
    key_path: Path | str,
) -> list[ScenarioBundle]:
    """Decrypt held-out scenarios from `directory/*.json.age` using the `age` CLI.

    Silently returns [] if the key file is missing. This lets external
    contributors run the public benchmark without ever touching held-out data.
    """
    d = Path(directory)
    k = Path(key_path)
    if not d.exists() or not k.exists():
        return []
    if shutil.which("age") is None:
        raise RuntimeError("age binary not on PATH; install with `brew install age`")
    out: list[ScenarioBundle] = []
    for cipher in sorted(d.glob("*.json.age")):
        proc = subprocess.run(
            ["age", "-d", "-i", str(k), str(cipher)],
            check=True, capture_output=True,
        )
        obj = json.loads(proc.stdout)
        out.append(_bundle_from_obj(obj, is_held_out=True))
    return out
