---
title: AMB v2.2 — scale-sweep preregister
date: 2026-04-18
status: preregistered — DO NOT EDIT after first run
parent_report: benchmark/amb_v2/results/v2.1/REPORT.md
parent_tag: v0.2.2-amb-v2.1
hypothesis: memory-primitive value is a function of scale; v2.1 did not stress the regime where naive-append-only should collapse
author: Archon, on behalf of Andy Williams
---

# AMB v2.2 — scale-sweep preregister

## Why this exists

AMB v2.1 found that `naive-append-only` tied `langchain-buffer (stock)` at
~54.9 AUC and beat `archon-memory-core (tuned)` by 1.67 AUC points at
noise=0.30. The preregistered ordinal was FALSIFIED.

The hypothesis in REPORT.md §"Why the prediction missed" was: the benchmark's
substring-containment scorer over top-5 cosine-retrieved chunks favors any
adapter that retains everything, because at v2.1 scale (~30–60 chunks total,
noise ≤ 0.60) cosine retrieval from the full pool has near-perfect recall.

That hypothesis has a direct, falsifiable prediction: **if we stress scale,
naive's near-perfect-recall advantage must collapse.** Either it does, and
the memory primitive's value is real but benchmarked at the wrong regime —
or it does not, and the primitive thesis is largely wrong, and the real
product is "retrieval quality + curation," not "smart memory management."

v2.2 is the experiment that forces one of those two conclusions.

## Primary claim under test

> As scenario scale grows, `archon-memory-core (tuned)` AUC degrades LESS
> than `naive-append-only` AUC. At the largest regime tested,
> amc-tuned > naive by a statistically detectable margin.

## Scale regimes

Three regimes, run as independent grids. Scenarios regenerated per regime
using the same generator primitives as v2.1 (simulator.py), preserving
scenario *shape* (personal facts, project facts, contradiction pairs) but
scaling fact count and noise rate.

| Regime | Total chunks @ day 90 | Noise rate | Days | Contradiction pairs |
|---|---|---|---|---|
| **S (small)** | ~50 (v2.1 baseline) | 0.30 | 90 | 3 |
| **M (medium)** | ~500 (10×) | 0.60 | 180 | 15 |
| **L (large)** | ~5000 (100×) | 0.90 | 365 | 60 |

Each regime: 4 scenarios (1 public mini + 3 fresh held-outs, age-encrypted
before first run). 3 adapters × 2 modes × 3 seeds × 1 noise rate per regime.
Total runs: **3 regimes × 4 scenarios × 3 adapters × 2 modes × 3 seeds =
216 cells.**

Seeds pinned: 17, 31, 53 (same as v2.1).

## Adapters

Unchanged from v2.1:
- `naive-append-only` (stock only — tuned is no-op)
- `langchain-buffer` (stock and tuned)
- `archon-memory-core` (stock and tuned)

No new adapters. No config retuning between regimes. Whatever the tuned
config is at v2.1 tag `v0.2.2-amb-v2.1` is frozen for all three regimes.

## Pre-registered predictions

### P1 — Scale degradation curve (directional)

> `naive-append-only` AUC at regime L is **at least 8 AUC points lower**
> than its AUC at regime S on the same scenario shape.

P1 captures the core hypothesis: append-everything + cosine retrieval must
degrade when the pool is big and noisy. If P1 FAILS (naive holds within
8 points across 100× scale + 0.9 noise), the memory-primitive-thesis is
in serious trouble — append-everything is essentially free at any scale.

### P2 — amc advantage at scale (primary)

> At regime L: `archon-memory-core (tuned)` AUC exceeds `naive-append-only`
> AUC by **≥ 5 AUC points** with p < 0.05 (bootstrap, 10k resamples over
> the 3 seeds × 4 scenarios = 12 observations per cell).

P2 PASS → the memory primitive thesis is supported, just benchmarked at
the wrong regime in v2.1. /board amc: RED → GREEN, claim narrowed to
"smart memory management matters at ≥10× the scale AMB v2.1 tests."

P2 FAIL (gap < 5 OR naive wins at L) → the primitive thesis is
effectively falsified across our full tested range. Kill the
"archon-memory-core as a library" framing. Pivot to: "curation + retrieval
quality tools" as the real product.

### P3 — Crossover point (exploratory, publish regardless)

> There exists a regime between S and L where amc-tuned overtakes naive.
> Report the interpolated crossover chunk-count, or state "no crossover
> in tested range."

P3 is not a pass/fail — it is a published artifact. A crossover at ~500
chunks reframes the product as "kicks in when you pass this scale." No
crossover reframes the product as "doesn't kick in at any tested scale."

### P4 — Consolidation fires at scale (mechanism check)

> At regime L, amc-tuned's `consolidate()` marks `superseded_by` on at
> least 80% of eligible contradiction targets (60 pairs × 3 seeds × 4
> scenarios = 720 expected marks).

P4 is an internal-correctness check independent of the AUC outcome. If
P4 FAILS, the consolidation path broke at scale and P2's result is
uninterpretable — re-diagnose before reading P2.

## What could make us change this spec BEFORE first run

- Scenario generator cannot produce 5000-chunk / 0.9-noise scenarios
  without pathology (e.g., all queries become impossible). Document the
  pathology; reduce regime L to the largest feasible scale; record in
  this file BEFORE first run.
- ChromaDB cannot hold 5000 chunks × 12 runs per regime on Mac Mini
  hardware. Reduce seeds from 3 to 2. Document BEFORE first run.
- Runtime estimate exceeds 48 hours wall-clock. Parallelize across nodes
  (Mini + Forge + Cipher) with pinned random seeds. Document split.

## What we will NOT do AFTER first run

- Shift the 5-AUC / 8-AUC / 80% thresholds after seeing numbers.
- Drop regime L because "it's too noisy to score fairly." Fair-scoring
  problems are what we are testing for.
- Introduce a new metric mid-experiment.
- Retune amc config at regime L to chase a PASS.
- Cherry-pick seeds.
- Re-label a P2 FAIL as "preliminary" or "directional."
- Generate new held-outs if the first three produce an inconvenient result.

## Falsification response plan

**P1 PASS + P2 PASS:**
Primitive thesis is supported. Publish REPORT_v2.2.md. /board moves amc
from RED → GREEN with the narrowed claim. Announce externally as
"AMB v2.2 validates memory-library value at scale."

**P1 PASS + P2 FAIL:**
Naive degrades at scale (confirming the shape intuition), but amc does
not close the gap. The primitive is not the fix — retrieval reranking,
curation, or a different memory model is. /board amc: RED. Kill the
current framing. Draft v2.3 preregister for whichever retrieval-rerank
intervention we think closes the gap.

**P1 FAIL + P2 either:**
Append-everything + cosine is essentially free at every scale tested.
The "memory degrades without management" thesis is falsified across
our tested range. /board amc: REDACTED. Do not pitch the primitive.
Publish the finding — this is a load-bearing negative result for the
entire $50-phone commodity-hardware story and needs to be published
honestly before any fundraising conversation.

**P4 FAIL:**
Fix the consolidation path at scale before drawing conclusions from
P1/P2. Rerun regime L only.

## Compute budget

Rough estimate:
- Regime S: ~20 min (v2.1 baseline)
- Regime M: ~3 hours (10× chunks → 10× ingest + 10× query work)
- Regime L: ~30 hours (100× chunks, likely dominated by ChromaDB ingest
  throughput on commodity hardware)

Plan: run S + M overnight on Mini. Run L split across Mini + Forge with
pinned per-scenario seeds so results are bit-reproducible across nodes.
If L exceeds 48h wall clock, split further to Cipher.

## Primary artifacts to produce

- `benchmark/amb_v2/results/v2.2/S/` — 24 JSONs (small regime)
- `benchmark/amb_v2/results/v2.2/M/` — 72 JSONs (medium regime)
- `benchmark/amb_v2/results/v2.2/L/` — 72 JSONs (large regime) — or
  partial set with documented incomplete-run reason
- `benchmark/amb_v2/results/v2.2/REPORT.md` — honest pass/fail per P1–P4
- `benchmark/amb_v2/results/v2.2/crossover.svg` — P3 chart regardless of outcome
- Held-out ciphertexts: `held_out/v2.2/{M,L}/h0{1,2,3}.json.age`

## Held-out scenarios

Regime-specific held-outs generated BEFORE first run, age-encrypted,
committed with ciphertext. Decryption password known only to Andy, passed
via env var at run time. Same methodology as v2.1 held-outs.

## Author signoff

Archon, on behalf of Andy Williams — 2026-04-18.

This preregister exists to force a definitive answer on whether
"agent memory degrades over time without management" is an intuition
about a real regime or an intuition about no regime we can reproduce.
Either outcome is useful. A falsification here kills the product.
A confirmation here closes the evidence gap that /board flagged today.
