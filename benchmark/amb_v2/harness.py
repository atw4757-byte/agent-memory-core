"""Harness — main loop. Orchestration only; no business logic.

Loop:
  for day in range(days):
    chunks = simulator.next_day_chunks
    if chunks: adapter.ingest(day, chunks)
    if day > 0: adapter.consolidate(day)
    if day in checkpoints: results[day] = run_checkpoint_queries(adapter)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmark.amb_v2.adapters.base import Mode, split_returned_chunks, validate_metadata
from benchmark.amb_v2.metrics import (
    QueryResult,
    answer_accuracy,
    auc_quality,
    confuser_resistance,
    contradiction_resolution,
    quality_at,
    quality_at_v2_3,
    salience_preservation,
    stale_fact_rate,
    temporal_improvement,
    top1_answer_accuracy,
)
from benchmark.amb_v2.scenarios import ScenarioBundle
from benchmark.amb_v2.simulator import simulate

DEFAULT_CHECKPOINTS = (0, 7, 14, 30, 60, 90)
SPEC_VERSION = "v2.1.0"

RESULTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "adapter", "version", "mode", "seed", "noise_rate",
        "spec_version", "checkpoints", "auc_quality", "temporal_improvement",
        "metadata", "completed_at",
    ],
    "properties": {
        "adapter": {"type": "string"},
        "version": {"type": "string"},
        "mode": {"enum": ["stock", "tuned"]},
        "seed": {"type": "integer"},
        "noise_rate": {"type": "number"},
        "spec_version": {"type": "string"},
        "metadata": {"type": "object"},
        "completed_at": {"type": "string"},
        "auc_quality": {"type": "number"},
        "temporal_improvement": {"type": "number"},
        "checkpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["day", "answer_accuracy", "contradiction_resolution",
                             "stale_fact_rate", "salience_preservation", "quality"],
                "properties": {
                    "day": {"type": "integer"},
                    "answer_accuracy": {"type": "number"},
                    "contradiction_resolution": {"type": "number"},
                    "stale_fact_rate": {"type": "number"},
                    "salience_preservation": {"type": "number"},
                    "quality": {"type": "number"},
                },
            },
        },
    },
}


def _build_supersede_index(scenarios: list[ScenarioBundle]) -> dict[str, str]:
    """Map (scenario_id, query_id) → superseded value text, derived from timeline."""
    index: dict[str, str] = {}
    for b in scenarios:
        # For each query that's a contradiction, find the chunk with `supersedes`
        # and record the OLD value's text.
        chunk_by_id = {ev["id"]: ev for ev in b.timeline}
        for q in b.queries:
            if q.resolution_type != "contradiction":
                continue
            for ev in b.timeline:
                if ev.get("supersedes") and ev.get("supersedes") in chunk_by_id:
                    old = chunk_by_id[ev["supersedes"]]
                    # Naive heuristic: superseded value is the OLD chunk's text.
                    # Better: extract specific phrase. For benchmark scoring,
                    # the full old text suffices (substring containment).
                    index[f"{b.scenario_id}::{q.query_id}"] = old["text"]
                    break
    return index


def _build_chunk_type_index(scenarios: list[ScenarioBundle]) -> dict[str, str]:
    """Map (scenario_id, query_id) → chunk type of the answer source."""
    index: dict[str, str] = {}
    for b in scenarios:
        # Heuristic: a query's chunk_type matches the type of the chunk it asks
        # about. We approximate via expected_answer substring match against
        # timeline events. If multiple match, prefer the latest (highest day).
        for q in b.queries:
            best: tuple[int, str] | None = None
            for ev in b.timeline:
                if q.expected_answer.lower() in ev["text"].lower():
                    if best is None or ev["day"] > best[0]:
                        best = (ev["day"], ev["type"])
            index[f"{b.scenario_id}::{q.query_id}"] = best[1] if best else "fact"
    return index


def _build_confuser_index(scenarios: list[ScenarioBundle]) -> dict[str, tuple[str, ...]]:
    """Flatten per-bundle confusers into a query_id → texts map for metric use."""
    out: dict[str, tuple[str, ...]] = {}
    for b in scenarios:
        for qid, items in b.confusers.items():
            out[qid] = tuple(item["text"] for item in items)
    return out


def _run_checkpoint(
    adapter: Any,
    scenarios: list[ScenarioBundle],
    day: int,
    supersede_idx: dict[str, str],
    chunk_type_idx: dict[str, str],
    confuser_idx: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    qrs: list[QueryResult] = []
    for b in scenarios:
        for q in b.queries:
            if day not in q.checkpoint_eligibility:
                continue
            try:
                ans = adapter.query(q.question, b.scenario_id) or ""
            except Exception:
                ans = ""
            returned = split_returned_chunks(ans)
            top1 = returned[0] if returned else None
            key = f"{b.scenario_id}::{q.query_id}"
            qrs.append(QueryResult(
                query_id=q.query_id,
                scenario_id=b.scenario_id,
                actual_answer=ans,
                expected_answer=q.expected_answer,
                aliases=tuple(),
                resolution_type=q.resolution_type,
                chunk_type=chunk_type_idx.get(key, "fact"),
                superseded_value=supersede_idx.get(key),
                top1_chunk=top1,
                returned_chunks=returned,
            ))

    aa = answer_accuracy(qrs)
    cr = contradiction_resolution(qrs)
    sfr = stale_fact_rate(qrs)
    sp = salience_preservation(qrs)
    t1a = top1_answer_accuracy(qrs)
    cres = confuser_resistance(qrs, confuser_idx or {})
    q = quality_at(answer=aa, contradiction=cr, stale=sfr, salience=sp)
    q23 = quality_at_v2_3(
        top1_answer=t1a, any_answer=aa, contradiction=cr,
        stale=sfr, salience=sp, confuser_resist=cres,
    )

    return {
        "day": day,
        "answer_accuracy": round(aa, 4),
        "top1_answer_accuracy": round(t1a, 4),
        "confuser_resistance": round(cres, 4),
        "contradiction_resolution": round(cr, 4),
        "stale_fact_rate": round(sfr, 4),
        "salience_preservation": round(sp, 4),
        "quality": round(q, 4),
        "quality_v2_3": round(q23, 4),
    }


def run_one(
    *,
    adapter: Any,
    scenarios: list[ScenarioBundle],
    seed: int,
    noise_rate: float,
    mode: Mode,
    checkpoints: tuple[int, ...] | list[int] = DEFAULT_CHECKPOINTS,
    days: int | None = None,
    filler_facts_per_day: int = 0,
) -> dict[str, Any]:
    """Run one full benchmark pass for one (adapter, mode, seed, noise_rate)."""
    validate_metadata(adapter.metadata)
    if adapter.metadata["mode"] != mode:
        raise ValueError(
            f"adapter instantiated in mode={adapter.metadata['mode']!r} "
            f"but harness asked to run mode={mode!r}"
        )

    cps = sorted(set(checkpoints))
    if days is None:
        days = max(cps) + 1

    supersede_idx = _build_supersede_index(scenarios)
    chunk_type_idx = _build_chunk_type_index(scenarios)
    confuser_idx = _build_confuser_index(scenarios)

    checkpoint_results: list[dict[str, Any]] = []
    for day, chunks in simulate(scenarios, seed=seed, noise_rate=noise_rate,
                                 days=days,
                                 filler_facts_per_day=filler_facts_per_day):
        if chunks:
            adapter.ingest(day, chunks)
        if day > 0:
            adapter.consolidate(day)
        if day in cps:
            checkpoint_results.append(_run_checkpoint(
                adapter, scenarios, day, supersede_idx, chunk_type_idx,
                confuser_idx,
            ))

    pts = [(c["day"], c["quality"]) for c in checkpoint_results]
    auc = auc_quality(pts)
    temporal = temporal_improvement(pts)

    return {
        "adapter": adapter.metadata["name"],
        "version": adapter.metadata["version"],
        "mode": mode,
        "seed": seed,
        "noise_rate": noise_rate,
        "spec_version": SPEC_VERSION,
        "metadata": dict(adapter.metadata),
        "checkpoints": checkpoint_results,
        "auc_quality": round(auc, 4),
        "temporal_improvement": round(temporal, 4),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
