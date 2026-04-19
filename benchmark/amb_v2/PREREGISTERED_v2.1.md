---
title: AMB v2.1 — Pre-registration
status: FROZEN
spec_version: v2.1.0
preregistered_at: 2026-04-18
author: Archon (on behalf of Andy Williams)
commit_pin: f3a1570
---

# AMB v2.1 Pre-registration

v2.0.1 shipped a falsification: `agent-memory-core` (stock + tuned)
under-performed the `naive-append-only` baseline, and `langchain-tuned`
silently no-op'd because its consolidate() path was never exercised.
The v2.0.1 REPORT.md documented this honestly instead of quietly
retuning. See `results/v2.0.1/REPORT.md` for the autopsy.

v2.1 is a deliberate response to three methodology findings and one
core-library gap uncovered during v2.0.1:

1. **Core gap** — `MemoryStore` had no `consolidate()` method, so the
   adapter's `mode="tuned"` path could not exercise any supersedes-aware
   logic. FIX: added `MemoryStore.consolidate()` (deterministic,
   metadata-driven) and `search(..., include_superseded=False)` filter.
2. **Adapter gap** — the adapter silently dropped chunks whose scenario
   type was not in core's `VALID_TYPES` (`preference`, `update`, `noise`),
   and did not propagate scenario-level chunk ids or supersedes metadata
   into the core store. FIX: type-map + `extra_metadata={scenario_chunk_id,
   supersedes}` pass-through in `ingest()`.
3. **Noise invariance** — v2.0.1 showed essentially flat Quality@T as
   `noise_rate` swept from 0.20 to 0.60 for every adapter, indicating
   that the noise templates never competed in embedding space. FIX:
   `NOISE_TEMPLATES` replaced with 20 lexically-loaded business-domain
   distractors that share vocab (projects, colors, teams, roadmaps,
   credentials) with the scenario query space.
4. **Missing temporal signal** — AUC aggregates quality across time but
   cannot distinguish a system that learns from one that plateaus. FIX:
   new `temporal_improvement(checkpoints)` metric (late-half mean minus
   early-half mean) emitted alongside AUC in every result file.

This document is the contract. It is written **before any v2.1 grid is
computed**.

If the published v2.1 numbers diverge from the predictions below, the
divergence is a finding — not a bug, not an excuse to revise the formula.
v2.0.1 already established that we publish falsification.

## 0. Revision log (pre-publish)

- **2026-04-18 v2.1** — initial registration at commit `f3a1570` following
  the v2.0.1 falsification release (`v0.2.1-amb-v2-preview`).

## 1. Methodology summary (delta vs v2.0.0)

- **Spec version**: `v2.1.0` (bumped in `harness.py` SPEC_VERSION).
- **Composite formula**: UNCHANGED. The AST-pinned weights `0.40,
  0.30, 0.15, 0.15` remain frozen; the v2.1 changes do not touch any
  metric formula.
- **New metric** (report-only, NOT part of composite):
  `temporal_improvement = mean(late-half checkpoints) −
   mean(early-half checkpoints)`. Emitted in every result JSON as
   `"temporal_improvement": <float>`. A system that genuinely learns
   from accumulated context should score >0 here; a stateless retrieval
   loop scores ~0.
- **Noise templates**: replaced 10 generic templates with 20 domain-
  loaded templates. See diff in `simulator.py`. Determinism and noise-
  rate calibration guarantees are unchanged — only the template
  content string set changed.
- **Core library changes** are out-of-tree from the benchmark, but the
  adapter hash (`adapters/agent_memory_core.py`) captures the adapter-
  side wiring (type-map, supersedes pass-through, fail-loud tuned).
- **Adapters shipped**: naive, agent-memory-core, langchain-buffer
  (same as v2.0.1). llama-index and mem0 remain deferred.
- **Modes**, **sensitivity sweep** (4 noise rates × 3 seeds), and
  **checkpoints** (0, 7, 14, 30, 60, 90) are UNCHANGED.

## 2. Composite formula — FROZEN (unchanged from v2.0)

```
quality = 0.40 · answer_accuracy
        + 0.30 · contradiction_resolution
        + 0.15 · (1 − stale_fact_rate)
        + 0.15 · salience_preservation
```

Weights are pinned by an AST-level unit test in
`tests/test_metrics.py::test_quality_formula_weights_frozen_in_source`.

The new `temporal_improvement` metric is **reported but not composited**.
Composite tuning is forbidden until a hypothetical v3 release with a
60-day grace period.

## 3. Implementation file hashes (SHA-256) — v2.1 pin

Computed at pre-registration time on the tip of `v1.1-dev` after all
v2.1 source changes have landed. Any change to these files after the
commit pin invalidates the pre-registration and requires a new
`PREREGISTERED_v2.1.1.md`.

| File | SHA-256 |
|---|---|
| `chunks.py` | `e9269752dbc99088dc92e64552fe56653116b47dfb2932da7d62b8f8a9c8a991` |
| `queries.py` | `a8fb6bef2f3cde5a3ec08cedc6a51f980ad59f59ebaff851f4d762204f332c7b` |
| `scenarios.py` | `1448e9932d383552589d2c34abc691a2a1849be75fe233daa8220064fb1c3615` |
| `simulator.py` | `5ee8471dacf59c2c59556144122a36d4aaf7fbf8fb688824b289a95ba04b0aac` |
| `metrics.py` | `fd884362a77611c1332ec5fa089df8275617c84120d86b60a28b53dbbfd334c2` |
| `harness.py` | `e7e5db54a3be2c9ae4169a5603219c7a30d595f54828b3c2b2dcd7c168c74515` |
| `run.py` | `05e7813ab26e0ede01de2029fac3b89d2818f1a49ac9bf7a8ad4315bb0f72a7d` |
| `run_all.py` | `a2c4af4d631aa39ac4571a1b14ffe5816119992b574587bb0918ef257a317392` |
| `chart.py` | `32240c9f83541f4ad8cce3d808d4f1ede60bd4ce274c547b2ea818d377f21541` |
| `adapters/base.py` | `85952e7c5d2d03271e26a4cceb9f080ed0d1686b3a4c8dd6530804369b33d3a7` |
| `adapters/naive.py` | `62d973fc595bf9f6451fd4b47c40714a87927db2f332b1dade837245bf2c08ea` |
| `adapters/agent_memory_core.py` | `3da96f59ec716fb8b52af7f7bdc7fff1a49b261f66c7da6f5ef2116a660ef024` |
| `adapters/langchain_adapter.py` | `4ceadc31fc98d4050236e2907b8b517434d7221e094172265d6d98bdc7a4d970` |
| `scripts/generate_held_out.py` | `4bcbcb6c4a7025df51147a8377376149918214e4a385fb68259e463b9b0bd6dd` |
| `tests/fixtures/mini_scenario.json` | `4a750e8b58b206074455c8b9e78af8d2694b48e6821a514452b298532d454c7f` |

Regenerate with:
```bash
cd benchmark/amb_v2 && shasum -a 256 chunks.py queries.py scenarios.py \
  simulator.py metrics.py harness.py run.py run_all.py chart.py \
  adapters/base.py adapters/naive.py adapters/agent_memory_core.py \
  adapters/langchain_adapter.py scripts/generate_held_out.py \
  tests/fixtures/mini_scenario.json
```

## 4. Held-out generation prompt

UNCHANGED from v2.0.0. The `h01`/`h02`/`h03` scenarios used in v2.1 are
the same ciphertext bundles used in v2.0.1 — no re-generation, no
re-seeding. This keeps the held-out set genuinely held out across
releases.

## 5. Pre-registered Quality@90 predictions (v2.1)

These predictions are authored by Archon **before any v2.1 grid is
run**. They reflect the expectation that the four v2.1 changes will
move the ordinal prediction from "reversed" (v2.0.1) toward "confirmed"
(v2.1). Point estimates remain best-guess under meaningful noise.

All predictions at `noise_rate=0.30`, averaged across seeds `{42, 43,
44}`, across public `mini_scenario` plus held-out `h01` + `h02` + `h03`.

| # | Adapter | Mode | Predicted Quality@90 | Rationale |
|---|---|---|---|---|
| 1 | `naive-append-only` | stock | **0.35** | unchanged from v2.0 |
| 2 | `naive-append-only` | tuned | **0.35** | still a no-op for naive |
| 3 | `agent-memory-core` | stock | **0.55** | type-map fix + metadata pass-through restores chunks that were being dropped; same level as v2.0 prediction, now defensible |
| 4 | `agent-memory-core` | tuned | **0.70** | supersedes-aware `consolidate()` + search filter actively resolves contradictions; this is the bet |
| 5 | `langchain-buffer` | stock | **0.40** | unchanged |
| 6 | `langchain-buffer` | tuned | **0.50** | unchanged |

**Ordinal @ AUC (v2.1, primary bet):**

> `agent-memory-core (tuned) > agent-memory-core (stock) > langchain
> (tuned) > langchain (stock) ≈ naive`

To count as confirmed, this ordering must hold across **≥3 of the 4
noise rates** `{0.20, 0.30, 0.45, 0.60}` AND the gap between
`agent-memory-core (tuned)` and the next-best adapter must be ≥5 AUC
points at `noise_rate=0.30`.

**Temporal improvement prediction (secondary bet):**

> `agent-memory-core (tuned)` scores `temporal_improvement > 0.10`
> under `noise_rate=0.30`, averaged across seeds, while
> `naive-append-only (stock)` scores `temporal_improvement ∈ [-0.05,
> 0.05]`.

In plain terms: a real memory system should learn as more context
arrives; a stateless append-only loop should not. If the tuned amc
trajectory is flat (`temporal_improvement ≤ 0.05`), the consolidate
path is not contributing signal, regardless of AUC.

## 6. Falsification plan — explicit

If the v2.1 grid does NOT confirm the ordinal above, we do **one of
two things** — no third option:

- **If the gap is small** (amc-tuned within 3 AUC points of the next
  adapter): publish a v2.1 REPORT.md styled the same as v2.0.1, ship
  the falsification honestly, and the next change targets the CORE
  library (retrieval scoring, salience weighting, consolidation
  cadence), NOT the benchmark.
- **If the gap is inverted** (amc-tuned worse than naive): emergency
  autopsy, pause all v2.x grid work, open an issue against
  `agent-memory-core` main. Do NOT adjust the benchmark.

Under no circumstances do we retune composite weights, drop scenarios,
or re-generate held-out sets in response to v2.1 numbers.

## 7. Non-goals for v2.1

- No claim of absolute scale. Still synthetic + held-out, no real
  conversational dataset (that's v2.2 real-data track).
- No absolute leaderboard claim across labs.
- No new adapter integrations (llama-index, mem0 remain deferred).

## 8. Author signoff

Archon, on behalf of Andy Williams — 2026-04-18, commit pin `TBD`.
This PREREGISTERED_v2.1.md file will be committed immediately AFTER
the source changes it describes, and the commit_pin field will be
updated in the same commit. No v2.1 grid is run until both have
landed on `v1.1-dev`.
