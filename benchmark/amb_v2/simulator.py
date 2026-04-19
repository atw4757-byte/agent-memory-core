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

FILLER_FACT_TEMPLATES = (
    # v2.2 scale knob: well-formed "facts" about unrelated entities that share
    # embedding neighborhood with scenario queries without containing any
    # expected_answer substring. These stress retrieval pool density — the
    # regime where a real memory primitive (filter/rerank) should win.
    # Entities (people, projects, places, colors, drinks) are intentionally
    # disjoint from public + known held-out scenario vocab.
    "Marcus takes his espresso with a twist of lemon peel.",
    "The Jupiter project finished its closed beta in mid-October.",
    "Lena's bicycle is a teal Bianchi with chrome handlebars.",
    "The Chronos team relocated to the third floor last quarter.",
    "Hiroshi's favorite pastime is weekend woodworking in the garage.",
    "The Meridian initiative picked Wednesday mornings for status syncs.",
    "Priya keeps a fiddle-leaf fig on the left side of her desk.",
    "The Atlas workstream runs a quarterly show-and-tell.",
    "Diego's rescue parrot is named Oreo and speaks four words.",
    "The Nebula program is sponsored by the infrastructure org.",
    "Sana drives a navy crossover with roof rails and a bike rack.",
    "The Kepler pilot covered three regions before the re-scoping.",
    "Ravi's bookshelf is mostly biographies and nautical maps.",
    "The Orion channel was muted after the reorg announcement.",
    "Camila's houseplant collection includes a snake plant by the window.",
    "The Vesta dashboard was retired in favor of the unified portal.",
    "Theo brews cold drip coffee on Sunday afternoons.",
    "The Calypso team renamed itself to reduce confusion with Vega.",
    "Mei's laptop case has stickers from every conference she attended.",
    "The Titan ritual is a Friday hand-off meeting at four PM.",
    "Sven bakes sourdough on a schedule keyed to moon phases.",
    "The Pegasus budget line was consolidated with Orion for Q4.",
    "Aarav's dog is a border collie called Scout.",
    "The Helios squad owns the public-facing reporting layer.",
    "Ines keeps a sextant on her desk as a conversation piece.",
    "The Draco working group meets the first Tuesday of the month.",
    "Omar's cycling shoes are white with blue accents on the heel.",
    "The Persephone migration was scoped to six sprint cycles.",
    "Yuki hand-letters birthday cards for the team every year.",
    "The Icarus review board skipped its November session for holidays.",
    "Rafa's bonsai is a five-year-old juniper with a curved trunk.",
    "The Aurora contract renewal negotiation dragged into February.",
    "Noa's keyboard has blue switches and a woven braid cable.",
    "The Sagitta playbook was updated after the last incident retro.",
    "Pablo trains for marathons on a route through Druid Hills.",
    "The Lyra roadmap has three milestones tied to the annual summit.",
    "Fatima paints watercolors of coastal lighthouses on weekends.",
    "The Carina service returned to green after the failover drill.",
    "Lukas keeps a notebook of overheard sentences from train stations.",
    "The Phoenix working doc lives in the shared drive's archive folder.",
)

NOISE_TEMPLATES = (
    # Lexically-loaded: share domain vocab with scenario queries (projects,
    # languages, colors, teams, schedules, credentials) without containing
    # any actual answer substring. These compete in embedding space, which
    # is what separates a retrieval system that reasons from one that does
    # lexical matching on top-k.
    "The project kickoff meeting was rescheduled to next Thursday afternoon.",
    "A client asked whether our roadmap covers the legacy reporting module.",
    "The team debated whether a different framework would simplify tooling.",
    "Marketing shared a mood board full of muted earth tones for review.",
    "Someone on the design team proposed rotating the color palette quarterly.",
    "The infrastructure channel discussed migrating the staging cluster.",
    "A new hire asked about onboarding docs for the authentication service.",
    "The backlog grooming session moved five tickets into next sprint.",
    "HR sent a reminder about the upcoming benefits enrollment window.",
    "The CI pipeline flagged a flaky test that no one has claimed yet.",
    "A stakeholder questioned whether the current vendor meets compliance.",
    "The architecture review was postponed until the lead returns from leave.",
    "Finance confirmed the hardware budget for the next quarter is locked.",
    "A team member shared an article about developer productivity metrics.",
    "The weekly engineering sync covered three unrelated status updates.",
    "Someone suggested using a polling approach instead of webhooks.",
    "A product manager pushed back on the proposed release cadence.",
    "Legal wants another review pass on the vendor data-sharing clauses.",
    "The documentation team requested examples for the public API reference.",
    "An operations lead raised concerns about on-call rotation fairness.",
)


def _scenario_short(sid: str) -> str:
    return sid.split("_", 1)[0] if "_" in sid else sid[:6]


def simulate(
    scenarios: list[ScenarioBundle],
    seed: int,
    *,
    noise_rate: float = 0.45,
    days: int = 90,
    filler_facts_per_day: int = 0,
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
    filler_facts_per_day : v2.2 scale knob. Inject K additional well-formed
        "facts" per simulated day drawn from FILLER_FACT_TEMPLATES. These are
        domain-neighbor facts about unrelated entities — they compete in the
        retrieval pool without containing any query's expected_answer, so
        they stress embedding-space density without creating false positives.
        Default 0 preserves v2.1 output byte-for-byte.
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

    # Inject filler facts (v2.2 scale knob). Deterministic: RNG state after
    # noise injection is a pure function of (seed, scenarios, noise_rate, days).
    if filler_facts_per_day > 0:
        filler_counter = 0
        for d in range(days):
            for _ in range(filler_facts_per_day):
                sid = (scenarios[rng.randrange(len(scenarios))].scenario_id
                       if scenarios else "filler")
                text = rng.choice(FILLER_FACT_TEMPLATES)
                cid = f"filler-{_scenario_short(sid)}-d{d:03d}-f{filler_counter:05d}"
                by_day[d].append(Chunk(
                    id=cid, scenario_id=sid, day=d, text=text, type="fact",
                ))
                filler_counter += 1

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
