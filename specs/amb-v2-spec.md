---
title: "AMB v2 — Longitudinal Agentic Memory Benchmark"
status: Phase 1 REVISED — adversarial critique incorporated (C1)
created: 2026-04-18
revised: 2026-04-18 14:30 ET
spec_owner: Archon
reviewer: Andy
adversary: Cipher / Gemini 2.5 Pro
adversary_review: Memory/adversary-reviews/2026-04-18_amb-v2-adversary.md
supersedes: none (extends AMB v1)
next_phase: Phase 2 Plan
target_launch: 2026-05-15 (public leaderboard), aligning with roadmap commitment
---

## Phase 1.5 review summary

Adversarial review by Cipher / Gemini 2.5 Pro (verdict 3/10) flagged three legitimate critiques. All three are incorporated into this revision:

1. **Privileged operations (`consolidate()` is a backdoor for the home-team adapter)** → fixed via D10 (Stock vs Tuned mode reporting).
2. **Synthetic data is not a valid proxy for real usage** → mitigated via D12 (v2.1 real-data validation track), publicly committed in v2.0 README.
3. **Author bias on held-out set** → fixed via D5 update (Cipher authors held-out scenarios from public spec only).

Plus: distribution targets (30% updates, 45% noise) are guesses → addressed via D11 (sensitivity analysis across 4 noise rates).

Rejected: Gemini's "scrap synthetic, use Wikipedia for v2.0" — would push launch past 2026-05-15. Pre-registration accusation of "theater" rejected.

Full review and dispositions: `Memory/adversary-reviews/2026-04-18_amb-v2-adversary.md`.

## Context

AMB v1 shipped 2026-04 as a point-in-time benchmark (10 scenarios, 200 queries, 4 reference adapters). It proved that `archon-memory-core` resolves contradictions and handles temporal reasoning better than naive vector retrieval at a fixed moment. It does **not** answer the question customers actually ask: *"what happens to my memory system after six months of use?"*

Every naive system degrades — noise accumulates, contradictions stack, salience collapses. The commercial thesis of `archon-memory-core` is that memory **improves with age**, not degrades. AMB v1 cannot prove or disprove this thesis. AMB v2 is designed to do exactly that.

This is also the most ambitious claim in the `ROADMAP.md`, blog post, and marketing site. All three promise AMB v2 by 2026-05-15. Shipping vaporware here would hurt far more than shipping nothing. This spec exists to make AMB v2 real.

## Goals

1. **Measure memory quality as a function of simulated time.** Not one number per adapter — one curve per adapter.
2. **Test at production-realistic volume.** 10K chunks ingested over the simulated period. This is the scale at which v1-style benchmarks stop being useful.
3. **Pre-register methodology.** Methodology file committed and hash-pinned before any adapter runs. Prevents post-hoc tuning.
4. **Keep a held-out challenge set.** 30% of scenarios never publish; used to detect scenario-tuning.
5. **Reproducible from one command.** `python benchmark/amb_v2/run.py --adapter <name> --seed <int>`. No hidden config.
6. **Adapters for parity with v1.** Same 4 adapters (naive, LangChain, LlamaIndex, archon-memory-core) plus leaderboard submission interface.
7. **Produce decay curves as the headline deliverable.** SVG/PNG chart with day-on-x, quality-on-y, one line per adapter.

## Non-goals

1. **Real 90-day runtime.** Simulated days only. Wall-clock budget is ≤ 6 hours per adapter on a single workstation.
2. **Multilingual.** English only. Multilingual goes to AMB v3.
3. **Multi-turn agent loops.** We're testing memory retrieval quality, not end-to-end agent behavior.
4. **LLM-judged answers.** Deterministic string matching + structured metrics only. LLM judge is a v3 extension.
5. **Benchmarking beyond our adapter list.** Anyone else: submit via the leaderboard interface.

## Commercial framing (why this matters)

A memory layer that is 10% better on day 0 is not a product. A memory layer whose quality *holds or improves* while alternatives drop 40% over 90 simulated days **is** a product. AMB v2 exists to generate the single chart that makes that argument unambiguous.

## Design

### D1. Simulated time model

90 simulated days. Real wall-clock: compressed to ≤ 6 hours per adapter.

**Time granularity:** 1 simulated day = 1 ingest tick. Each day contributes ~110 new chunks on average (10,000 / 90 ≈ 111). Variance across scenarios: some days ingest 200 chunks (busy week), others ingest 30 (quiet).

**Event types per tick:**
- **Add:** new fact, new entity, new preference
- **Update:** value of an existing fact changes (contradiction event)
- **Noise:** session transcript snippets, confirmations, filler — low-salience chunks
- **No-op:** some days have nothing (realistic)

Distribution target per 10K corpus: 30% updates (contradictions), 20% novel facts, 45% noise, 5% no-op.

### D2. Measurement checkpoints

Queries fire at 6 checkpoints: **day 0, 7, 14, 30, 60, 90.**

Justification:
- Day 0: baseline after initial ingestion (first 111 chunks)
- Day 7: early-degradation catch
- Day 14: short-term memory failure mode
- Day 30: where most prod systems start breaking
- Day 60: long-term accumulation
- Day 90: where the commercial wedge should be widest

At each checkpoint, fire the full 200-query set (reuse AMB v1 queries). Result: **200 queries × 6 checkpoints × N adapters = 4,800 queries per adapter run.** At ≤ 100ms per query (retrieval + scoring), total: ~8 minutes per adapter retrieval time. Ingestion dominates wall clock.

### D3. Adapter interface

```python
class DecayAdapter(Protocol):
    def ingest(self, day: int, chunks: list[Chunk]) -> None: ...
    def consolidate(self, day: int) -> None: ...  # optional nightly hook
    def query(self, question: str, scenario_id: str) -> str: ...
```

Adapters receive chunks in temporal order. Chunks have `{id, text, type, timestamp_day, scenario_id}`. Consolidation callback fires once per simulated day **if the adapter opts in** — archon-memory-core uses it, naive does not.

This is the single extension point for third-party leaderboard submissions.

### D4. Metrics

Four metrics reported per checkpoint:

1. **Answer accuracy** — exact-match + alias table (same as v1). 0.0–1.0.
2. **Contradiction resolution rate** — for queries with known-changed answers, fraction that return the new answer. 0.0–1.0.
3. **Stale fact rate** — fraction of queries where the system returns a superseded fact. Lower is better. 0.0–1.0.
4. **Salience preservation** — credential-type queries, fraction returned in top-1. 0.0–1.0.

**Composite: Quality@T = 0.40·answer + 0.30·contradiction + 0.15·(1 − stale) + 0.15·salience**

Weights are documented and will not change without a v2.1 release and a 60-day grace period. This is the pre-registration commitment.

Two derived scores:
- **Quality@90** — single-number headline at end of run.
- **AUC (Quality, day)** — area under the curve across 0→90. Rewards systems that hold quality *throughout*, not just at the end.

### D5. Scenario set

AMB v1 has 10 scenarios. AMB v2 uses:

- **Public set:** 7 of the v1 scenarios, temporally extended (each rewritten so that the 5–8 sessions are spread across 90 days instead of all front-loaded).
- **Held-out set:** 3 new scenarios, **authored by Cipher / Gemini 2.5 Pro using the public spec only**, never seen the archon-memory-core implementation. The exact prompt used for held-out generation is committed verbatim to `PREREGISTERED.md`. The held-out scenarios are stored encrypted in `held_out/`, decrypted at run time only. Used by us to score any adapter that gets submitted; only composite numbers published, raw held-out queries never.

Total: 10 scenarios, same N as v1. Authorship split: 7 Archon (extended from v1), 3 Cipher (de-novo from public spec).

Each scenario now carries a **timeline** field: a list of `(day, event)` tuples describing when updates, additions, and noise happen. Authoring is manual; LLM-generated timelines reliably lack the adversarial structure we need.

### D6. Ingestion simulator

Component: `benchmark/amb_v2/simulator.py`.

Responsibilities:
1. Read scenario timelines + noise profile + seed.
2. Emit a 90-day sequence of `(day, chunk)` events, sorted by day.
3. Expose as an iterator so adapters consume lazily.

The simulator is **pure** given `(scenario_set, seed)` — any two runs with the same inputs produce byte-identical event streams. This is the reproducibility guarantee.

### D7. Harness

Component: `benchmark/amb_v2/run.py`.

```
for day in range(91):
    chunks = simulator.chunks_for(day)
    if chunks: adapter.ingest(day, chunks)
    if day > 0: adapter.consolidate(day)
    if day in CHECKPOINTS: results[day] = run_checkpoint_queries(adapter)
write_results(adapter_name, seed, results)
```

Output: `benchmark/amb_v2/results/<adapter>-seed<N>.json` containing per-checkpoint metrics + derived scores.

### D8. Pre-registration artifact

Before any adapter runs publicly, commit `benchmark/amb_v2/PREREGISTERED.md` containing:
- Methodology description (this spec, condensed)
- Composite formula, frozen
- Hash of `run.py`, `simulator.py`, `metrics.py`, scenario files
- Author's predictions for each adapter (Quality@90 within ±0.10)
- Signed commit timestamp

Results runs happen *after* this commit. Any methodology change post-registration bumps the version to v2.1 and triggers a 60-day grace period.

### D10. Stock vs Tuned mode (fairness fix)

`consolidate()` is a privileged operation — archon-memory-core uses it; LangChain `ConversationSummaryBufferMemory` and stock LlamaIndex memory do not. Running with consolidation enabled gives the home-team adapter a "thinking step" that out-of-the-box adapters lack.

Therefore every adapter is run in **two modes** and both numbers are published side-by-side:

- **Stock** — `consolidate()` is a no-op. Tests the adapter's out-of-the-box behavior.
- **Tuned** — `consolidate()` is invoked once per simulated day. Adapters that don't implement consolidation no-op silently. Tests the adapter when given the same daily compute budget as archon-memory-core.

The composite-metric ranking is published per-mode. There is **no combined ranking** that mixes the two. The decay-curve chart shows two lines per adapter (`stock` solid, `tuned` dashed), color-coded by adapter. This is honest: it surfaces where consolidation is the wedge and where it isn't, without hiding either result.

Adapter must declare in `metadata.json` whether it implements consolidation. Lying disqualifies the run from the leaderboard.

### D11. Sensitivity analysis on noise distribution

The distribution targets in D1 (30% updates / 45% noise) are calibration parameters, not measurements from real corpora. To prevent the benchmark from being a fragile artifact of a single noise rate, the full benchmark is run at **four noise rates: {20%, 30%, 45%, 60%}** (other percentages adjust proportionally).

**Stability requirement:** the top-3 adapter ranking on Quality@90 (per mode) must hold across at least 3 of 4 noise rates. If the ranking flips more than once, the headline number is flagged "unstable" and the benchmark itself is re-calibrated before publication.

The 4 noise-rate runs are reported individually in `decay-table.md`. The headline chart uses noise=30% (the central case).

Compute cost: 4 noise rates × 2 modes × 4 adapters × 90 days × 200 queries × 6 checkpoints. Bosgame can absorb this within the 6-hour-per-adapter budget if checkpoints are parallelized across noise rates.

### D12. Real-data validation track (v2.1 commitment)

The v2.0 benchmark is synthetic. This is its single largest credibility risk. To pre-empt the dismissal, v2.0 ships with a public commitment in `README.md`:

> *AMB v2.0 ships with synthetic scenarios. AMB v2.1, due within 30 days of v2.0 launch, adds a parallel "real-data validation track" derived from a longitudinal public corpus. If the synthetic-set adapter ranking diverges by more than 1 position from the real-data ranking, AMB v2.0 results will be marked "invalidated, re-calibration in progress" until the synthetic generator is fixed.*

Candidate corpora (decision deferred to Phase 2):
- Wikipedia full edit history of a single moderately-complex article (e.g. a public company, a long-running TV show)
- SEC filings of a single public company over 5+ years
- Issue/PR timeline of a stable open-source repository

Selection criterion: organic temporal distribution + verifiable ground truth at multiple time points.

This commitment is publicly visible from day one. It exists to make the v2.0 launch defensible: synthetic-only is the current limitation, but the path to real-data validation is on the calendar, not aspirational.

### D13. Chart generation

Component: `benchmark/amb_v2/chart.py`.

Inputs: all `*-seed*.json` in `results/` (across 4 noise rates × 2 modes × 4 adapters).
Outputs:
- `decay-curves.svg` — primary marketing asset. Two lines per adapter (stock solid, tuned dashed), x=day, y=Quality. Headline noise rate: 30%.
- `decay-curves.png` — for embedding in the website.
- `decay-table.md` — machine-generated markdown with per-checkpoint + AUC, broken out by noise rate and mode.
- `sensitivity-grid.svg` — secondary chart: 4 noise rates as small multiples, showing ranking stability.

Matplotlib + no seaborn (avoid heavy deps for library consumers).

## Data model

### Chunk
```python
@dataclass(frozen=True)
class Chunk:
    id: str                # stable, e.g. "01-d007-a3"
    scenario_id: str       # "01_personal_assistant"
    day: int               # 0..89
    text: str
    type: Literal["fact","update","noise","credential","preference","session"]
    supersedes: str | None # chunk_id this one invalidates, if any
```

### Query (reuse v1 + new v2 fields)
```python
@dataclass
class Query:
    query_id: str
    scenario_id: str
    question: str
    expected_answer: str
    reasoning_type: str
    difficulty: Literal["easy","medium","hard"]
    trap: str | None
    # v2 additions
    checkpoint_eligibility: set[int]  # which checkpoints this query makes sense at
    resolution_type: Literal["stable","contradiction","aggregation","trajectory"]
```

Not every query fires at every checkpoint. A query asking "what is the user's current address" only makes sense after the address change (≈ day 30 in scenario 01).

### Results JSON
```json
{
  "adapter": "archon-memory-core",
  "version": "0.1.2",
  "seed": 42,
  "spec_version": "v2.0.0",
  "checkpoints": [
    {"day": 0,  "answer_accuracy": 0.91, "contradiction_res": 0.00, "stale_fact_rate": 0.00, "salience_preservation": 0.98, "quality": 0.923},
    ...
    {"day": 90, "quality": 0.81}
  ],
  "quality_at_90": 0.81,
  "auc_quality": 72.4
}
```

## Test strategy (TDD — Phase 3 will expand)

Layer 1 — simulator determinism:
- `test_simulator_pure_with_seed` — same seed twice → byte-identical output
- `test_simulator_respects_distribution` — over 10K chunks, add/update/noise/no-op ratios within ±2% of target
- `test_simulator_chunks_sorted_by_day`

Layer 2 — metrics:
- `test_quality_formula_weights_sum_to_1`
- `test_quality_at_checkpoint_edge_cases` — all-1s, all-0s, single metric
- `test_auc_computation` — trapezoid on known curve returns expected area

Layer 3 — harness:
- `test_harness_fires_checkpoints_only_at_expected_days`
- `test_harness_ingests_in_temporal_order` — day 5 chunks never fed before day 4
- `test_harness_results_schema_valid`

Layer 4 — adapters (per-adapter smoke):
- `test_adapter_handles_empty_day`
- `test_adapter_handles_consolidation_noop_by_default`
- `test_adapter_query_returns_string`

Layer 5 — regression:
- Golden-output test on a 5-day mini-scenario — commit a fixture with expected metrics, flag any drift.

Coverage target: ≥ 90% for simulator + metrics + harness. Adapters tested behaviorally against smoke + scenario fixture.

## Risks + mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| R1. Scenarios overfit to our adapter | HIGH | 3 held-out scenarios never published; external submissions scored against them |
| R2. Hand-authored timelines introduce author bias | HIGH | Pre-registered predictions — if authored adapter beats the prediction, that's a signal of bias. Disclosed in EVALUATION.md |
| R3. Wall-clock too slow to iterate | MEDIUM | Cap corpus at 10K chunks for v2.0; v2.1 can scale to 50K if compute allows |
| R4. LangChain/LlamaIndex adapters stale by May 15 | MEDIUM | Pin adapter dependency versions in `benchmark/amb_v2/requirements.txt` |
| R5. Composite formula weighting is arbitrary | MEDIUM | Report all 4 sub-metrics individually, not only composite. Let readers re-weight. |
| R6. Public claim of "improves with age" could be false | HIGH | Spec explicitly does NOT claim archon-memory-core wins. The chart shows what it shows. If we lose, we publish anyway. |
| R7. Someone tunes to our public scenarios | MEDIUM | Held-out set + spec-pinned methodology + 60-day grace on changes |
| R8. Synthetic-only is dismissed as a vanity metric | HIGH | D12 commits to real-data validation in v2.1 within 30 days. v2.0 README explicitly acknowledges the limitation. |
| R9. `consolidate()` is a backdoor for the home-team adapter | HIGH | D10 — every adapter run in stock + tuned modes, both published, no combined ranking |
| R10. Author bias on held-out set | HIGH | D5 — Cipher / Gemini 2.5 Pro authors held-out scenarios from public spec only; generation prompt committed verbatim to PREREGISTERED.md |
| R11. Fixed noise distribution makes results brittle | MEDIUM | D11 — sensitivity analysis across 4 noise rates {20, 30, 45, 60}%; ranking instability triggers re-calibration |

## Dependencies + blockers

- AMB v1 scenarios (exist) — extend in-place, no new infrastructure
- `archon-memory-core` 0.1.2 (released) — no upstream work needed
- LangChain / LlamaIndex adapter pin: capture current working deps in `requirements.txt`
- Matplotlib for charts (already in dev deps)
- **No external API calls.** Local-only. Critical: no OpenAI/Anthropic dependency in the benchmark path — otherwise reproducibility dies the moment pricing changes.

## Delivery plan (high level — Phase 2 expands)

- Phase 2 (Plan): file-by-file architecture, day-by-day budget
- Phase 3 (Tasks): TDD stages with C-gates
- Phase 4 (Implement): code the pieces
- Phase 5 (Review): Cipher security + code review
- Phase 6 (Alpha results): full run, chart, commit alpha
- **Target timeline:** 5 focused days of work, gated by Andy approval at Phase 2

## Open questions

### Resolved in Phase 1.5

1. ~~**Composite weights**~~ → KEEP 0.40/0.30/0.15/0.15. Sub-metrics differ from v1; weights reflect that contradictions and answer-accuracy carry the commercial argument. All 4 sub-metrics still reported individually so reviewers can re-weight.
2. ~~**6 checkpoints vs 14**~~ → KEEP 6. Smoother curves not worth 2x query load; we already have AUC computed via trapezoid.
3. ~~**3 vs 5 held-out scenarios**~~ → KEEP 3 (now Cipher-authored per D5 update). Stronger fairness fix than larger held-out count.
4. ~~**Adapters opt out of `consolidate()`**~~ → REPLACED by D10 (every adapter run in both modes; no opt-out, only a no-op).
5. ~~**Pre-registration includes predicted numbers**~~ → YES per D8. Publish predictions even if we lose. Losing publicly is more credible than not predicting.
6. ~~**Ingestion budget per day cap**~~ → 200 chunks/day soft cap, enforced as a clipping warning, not a hard error. Burstiness preserved.

### New from Phase 1.5 — for Andy to confirm before Phase 2

7. **Stock + Tuned mode reporting (D10).** Two headline numbers per adapter. Implies ~2x compute. Default: APPROVED unless Andy objects.
8. **Cipher-authored held-out scenarios (D5).** Gemini writes 3 hold-out scenarios from public spec only. Default: APPROVED.
9. **Sensitivity analysis on 4 noise rates (D11).** ~3-4x compute over single-rate run. Bosgame absorbs it. Default: APPROVED.
10. **v2.1 real-data validation track public commitment in v2.0 README (D12).** Default: APPROVED. Public accountability is a feature.
11. **Real-data corpus selection** — Wikipedia article history vs SEC filings vs OSS issue/PR history. Decision deferred to Phase 2.

## Acceptance criteria

Phase 1 is accepted when:
- [x] This spec is committed and pushed to main (C0 commit, 2026-04-18)
- [x] Phase 1.5 adversarial review completes, critique incorporated (C1 commit, 2026-04-18)
- [x] All Phase 1 open questions resolved or deferred with rationale (Q1–Q6 resolved, Q7–Q11 default-approved pending Andy override)

Phase 1.5 is accepted (this revision) when:
- [x] Adversary review file committed at `Memory/adversary-reviews/2026-04-18_amb-v2-adversary.md`
- [x] D10 (Stock vs Tuned), D11 (Sensitivity), D12 (Real-data v2.1) added
- [x] R8, R9, R10, R11 added to risks table
- [x] Held-out authorship moved to Cipher (D5)
- [x] Open questions table updated to reflect Phase 1.5 resolutions

Phases 2-6 have their own acceptance criteria in subsequent documents.
