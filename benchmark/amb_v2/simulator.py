"""Simulator — pure (scenarios, seed, noise_rate) → (day, chunks) event stream.

Determinism guarantee: identical inputs → byte-identical outputs.
Noise distribution is calibrated to match the requested rate within statistical
noise across the full corpus.
"""
from __future__ import annotations

import random
import warnings
from collections.abc import Iterator

from benchmark.amb_v2.chunks import Chunk
from benchmark.amb_v2.scenarios import ScenarioBundle

SOFT_CAP_PER_DAY = 200

NOISE_TEMPLATES = (
    "Casual chat about the weather and weekend plans.",
    "Asked the assistant for a joke; received and laughed.",
    "Brainstorm session about household projects.",
    "Discussed news headlines from this morning.",
    "Vented about a frustrating commute.",
    "Asked for restaurant recommendations nearby.",
    "Requested a music playlist for the afternoon.",
    "Talked about a recent movie or show.",
    "Casual reflection on the day so far.",
    "Asked the assistant to summarize a podcast episode.",
)


def _scenario_short(sid: str) -> str:
    return sid.split("_", 1)[0] if "_" in sid else sid[:6]


def simulate(
    scenarios: list[ScenarioBundle],
    seed: int,
    *,
    noise_rate: float = 0.45,
    days: int = 90,
) -> Iterator[tuple[int, list[Chunk]]]:
    """Yield (day, chunks) tuples for `days` simulated days.

    Pure: same inputs → byte-identical output sequence.

    Parameters
    ----------
    scenarios : list of ScenarioBundle to interleave.
    seed : RNG seed.
    noise_rate : target fraction of total chunks that should be noise.
        Calibrated at ±5% across full 90-day runs.
    days : number of simulated days to emit (0..days-1).
    """
    rng = random.Random(seed)

    # Materialize scenario timeline events into Chunks per day.
    by_day: dict[int, list[Chunk]] = {d: [] for d in range(days)}
    for bundle in scenarios:
        for event in bundle.timeline:
            d = event["day"]
            if d >= days:
                continue
            by_day[d].append(Chunk(
                id=event["id"],
                scenario_id=bundle.scenario_id,
                day=d,
                text=event["text"],
                type=event["type"],
                supersedes=event.get("supersedes"),
            ))

    # Compute noise injection. Some scenario events are themselves noise; we need
    # the total noise fraction of the final corpus to equal `noise_rate`.
    # Solve: (existing_noise + K) / (N_scenario + K) = noise_rate
    n_scenario = sum(len(v) for v in by_day.values())
    existing_noise = sum(
        1 for v in by_day.values() for c in v if c.type == "noise"
    )
    if noise_rate >= 1.0:
        n_noise = n_scenario * 99
    elif noise_rate <= 0.0:
        n_noise = max(0, -existing_noise)  # can't have negative; just 0
    else:
        target = (noise_rate * n_scenario - existing_noise) / (1.0 - noise_rate)
        n_noise = max(0, round(target))

    # Distribute noise across days using a deterministic shuffle.
    day_pool = list(range(days))
    noise_seq: list[int] = []
    for _ in range(n_noise):
        noise_seq.append(rng.choice(day_pool))
    noise_seq.sort()  # determinism + ascending day

    noise_counter = 0
    for d in noise_seq:
        # Pick scenario for noise attribution
        if scenarios:
            sid = scenarios[rng.randrange(len(scenarios))].scenario_id
        else:
            sid = "noise"
        text = rng.choice(NOISE_TEMPLATES)
        cid = f"{_scenario_short(sid)}-d{d:03d}-n{noise_counter:04d}"
        by_day[d].append(Chunk(
            id=cid, scenario_id=sid, day=d, text=text, type="noise",
        ))
        noise_counter += 1

    # Emit per day; warn if any day exceeds soft cap.
    for d in range(days):
        chunks = by_day[d]
        if len(chunks) > SOFT_CAP_PER_DAY:
            warnings.warn(
                f"day {d} emitted {len(chunks)} chunks, exceeding soft cap "
                f"of {SOFT_CAP_PER_DAY}",
                stacklevel=2,
            )
        # Sort within-day for determinism: by id (which itself encodes day + seq)
        chunks.sort(key=lambda c: c.id)
        yield d, chunks
