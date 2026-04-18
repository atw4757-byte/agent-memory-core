---
title: "AMB v2 — Longitudinal Agentic Memory Benchmark"
status: Phase 1 DRAFT — awaiting Phase 1.5 adversarial review
created: 2026-04-18
spec_owner: Archon
reviewer: Andy
adversary: Cipher (via archon-adversary)
supersedes: none (extends AMB v1)
next_phase: Phase 1.5 Adversarial Review
target_launch: 2026-05-15 (public leaderboard), aligning with roadmap commitment
---

## Context

AMB v1 shipped 2026-04 as a point-in-time benchmark (10 scenarios, 200 queries, 4 reference adapters). It proved that `agent-memory-core` resolves contradictions and handles temporal reasoning better than naive vector retrieval at a fixed moment. It does **not** answer the question customers actually ask: *"what happens to my memory system after six months of use?"*

Every naive system degrades — noise accumulates, contradictions stack, salience collapses. The commercial thesis of `agent-memory-core` is that memory **improves with age**, not degrades. AMB v1 cannot prove or disprove this thesis. AMB v2 is designed to do exactly that.

This is also the most ambitious claim in the `ROADMAP.md`, blog post, and marketing site. All three promise AMB v2 by 2026-05-15. Shipping vaporware here would hurt far more than shipping nothing. This spec exists to make AMB v2 real.

## Goals

1. **Measure memory quality as a function of simulated time.** Not one number per adapter — one curve per adapter.
2. **Test at production-realistic volume.** 10K chunks ingested over the simulated period. This is the scale at which v1-style benchmarks stop being useful.
3. **Pre-register methodology.** Methodology file committed and hash-pinned before any adapter runs. Prevents post-hoc tuning.
4. **Keep a held-out challenge set.** 30% of scenarios never publish; used to detect scenario-tuning.
5. **Reproducible from one command.** `python benchmark/amb_v2/run.py --adapter <name> --seed <int>`. No hidden config.
6. **Adapters for parity with v1.** Same 4 adapters (naive, LangChain, LlamaIndex, agent-memory-core) plus leaderboard submission interface.
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

Adapters receive chunks in temporal order. Chunks have `{id, text, type, timestamp_day, scenario_id}`. Consolidation callback fires once per simulated day **if the adapter opts in** — agent-memory-core uses it, naive does not.

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
- **Held-out set:** 3 new scenarios, authored for v2, never published. Used by us to score any adapter that gets submitted; only composite numbers published, raw held-out queries never.

Total: 10 scenarios, same N as v1.

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

### D9. Chart generation

Component: `benchmark/amb_v2/chart.py`.

Inputs: all `*-seed*.json` in `results/`.
Outputs:
- `decay-curves.svg` — primary marketing asset. One line per adapter, x=day, y=Quality.
- `decay-curves.png` — for embedding in the website.
- `decay-table.md` — machine-generated markdown with per-checkpoint + AUC.

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
  "adapter": "agent-memory-core",
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
| R6. Public claim of "improves with age" could be false | HIGH | Spec explicitly does NOT claim agent-memory-core wins. The chart shows what it shows. If we lose, we publish anyway. |
| R7. Someone tunes to our public scenarios | MEDIUM | Held-out set + spec-pinned methodology + 60-day grace on changes |

## Dependencies + blockers

- AMB v1 scenarios (exist) — extend in-place, no new infrastructure
- `agent-memory-core` 0.1.2 (released) — no upstream work needed
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

## Open questions (to resolve in Phase 1.5)

1. **Should the composite weights be the same as v1 (0.25/0.20/0.25/0.15/0.15)?** Spec currently proposes 0.40/0.30/0.15/0.15 — different from v1 because v2 has different sub-metrics (no precision, no recall). Need reviewer check that this framing is principled, not opportunistic.
2. **Is 6 checkpoints enough, or should we sample at every 7 days (14 checkpoints)?** More checkpoints = smoother curves but 2x query load.
3. **Held-out scenario count: 3 of 10, or 5 of 10?** More held-out = more overfitting resistance but less public transparency.
4. **Do we allow adapters to opt out of consolidate()?** Currently yes. Some reviewers may argue this favors agent-memory-core (which opts in).
5. **Should the pre-registration include our *predicted* numbers for each adapter?** Yes per D8, but this requires us to publish a prediction we may lose on.
6. **Ingestion budget per simulated day — cap at 200 chunks/day?** Burstier distributions might be more realistic but hurt reproducibility.

## Acceptance criteria

Phase 1 is accepted when:
- [ ] This spec is committed and pushed to main
- [ ] Phase 1.5 adversarial review completes, critique is either incorporated or explicitly rejected
- [ ] All 6 open questions have resolution from Andy (or deferred with rationale)

Phases 2-6 have their own acceptance criteria in subsequent documents.
