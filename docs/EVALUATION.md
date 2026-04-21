# Evaluation Design — AMB v1

The evaluation methodology for `archon-memory-core`, the Agentic Memory Benchmark (AMB) v1, and how to reproduce or challenge our numbers.

This document exists because the benchmark numbers in the README deserve scrutiny. Skepticism about memory-system benchmarks is correct by default — most of them optimise for the thing they measure, not for what agents actually need in production. This page walks through the design choices, the known weaknesses, and the reproducibility path, so anyone can audit the work rather than take our word for it.

---

## TL;DR

- **AMB v1**: 10 hand-authored scenarios, 200 queries, ~34 reasoning types, ~40% of queries include a "trap" (a plausible wrong answer a naive system will return).
- **Shipped artifacts**: scenarios under `benchmark/scenarios/*.json`, metrics under `benchmark/metrics.py`, harness under `benchmark/run_benchmark.py`, 4 reference adapters under `benchmark/adapters/`, a baseline result set under `benchmark/results/comparison.json`.
- **Reproducibility**: one command — `python benchmark/run_benchmark.py --adapter <name>`. No hidden config.
- **Known limits**: hand-authored scenarios are small-N, English-only, not longitudinal. AMB v2 (longitudinal, pre-registered, 90-day simulated decay) is on the roadmap and will replace v1 as the headline benchmark.

---

## 1. What We Are Trying to Measure

Most public memory benchmarks test **retrieval of static facts**. The failure mode they catch is "does the system find a chunk at all." That is the easiest case.

The failure mode we care about is what happens **when facts change over time**:

- A user's home address updates three sessions in.
- A stakeholder leaves a company mid-deal.
- A user's diet switches from lacto-ovo to vegan.
- A product bug gets fixed, but the memory still surfaces the original bug report as if it's open.
- A coffee order changes and the old one is retrieved because it was mentioned more often.

A memory system that cannot resolve contradictions, distinguish current from historical state, or recognise that a fact has been invalidated is not a production memory system. It is an expensive cache of stale data.

AMB v1 is built specifically to **separate systems that perform retrieval from systems that perform memory**. Surface-level composite scores look close (most production systems score 7–9 on naive queries). The gap opens up on contradiction resolution, temporal reasoning, and traps.

---

## 2. Dataset Construction

### 2.1 Scenarios (10 total, hand-authored)

Each scenario simulates **5–8 conversation sessions across months of interaction**, mirroring how a real agent would actually build memory over time — not one flat corpus.

| # | Scenario | Domain | Primary Stressors |
|---|---|---|---|
| 01 | Personal Assistant | Life admin | Address change, preference updates, cancelled meetings, rule aggregation |
| 02 | Executive Chief of Staff | B2B strategy | Board member departure, ARR updates, strategic pivot |
| 03 | Customer Support | SaaS support | Admin swap, bug resolved vs. recurring, open tickets |
| 04 | Health Coach | Wellness | Diet switch (lacto-ovo → vegan), supplement dose trajectory |
| 05 | Software PM | Product | Architecture owner change, roadmap churn, shipped vs. planned |
| 06 | Sales CRM | Enterprise sales | Economic buyer departure, blocker-to-supporter flip, pricing versions |
| 07 | Travel Concierge | Travel | Airport split by route, hotel exception per city |
| 08 | Tutor | Education | Misconception correction, teacher change, grade trajectory |
| 09 | Household | Home management | Appliance replacement, recurring service issues |
| 10 | Research Assistant | Academia | Architecture evolution (3 changes), normalization bug |

**Why hand-authored, not auto-generated?**

LLM-generated scenarios have a known pathology: they produce internally consistent conversations that do not contain real contradictions, traps, or the messy temporal structure we care about. A model that writes the scenario will never write a bug that catches another model. Hand-authoring is slower but produces harder, sharper test cases.

**Why ten?**

Ten is the smallest N that spans a meaningful range of domains (consumer/B2B, personal/professional, technical/lifestyle). We will add scenarios in v1.1 releases; submissions are welcome via PR.

### 2.2 Queries (200 total, 20 per scenario)

Each scenario contains 20 queries with the following distribution:

- **~6 Easy** — stable facts, baseline retrieval
- **~9 Medium** — queries requiring temporal awareness, basic aggregation, or single contradiction resolution
- **~5 Hard** — multi-hop, negative-with-history, multi-contradiction, causal chain

**~40% of queries include an explicit `trap` field** describing the wrong answer a naive system will give. Traps are written to be plausible — a simple vector retriever will return them.

Each query is annotated with:

- `query_id` — stable identifier within scenario
- `query` — the natural-language question
- `expected_answer` — the correct answer (string)
- `reasoning_type` — one of 30+ categorised types (see §3)
- `difficulty` — easy / medium / hard
- `trap` — the wrong-looking-right answer (optional, ~40% coverage)
- `relevant_sessions` — which sessions the answer depends on

### 2.3 Reasoning Types

The scenario queries span 30+ reasoning types, the most important being:

| Reasoning Type | What It Tests | Why It's Hard |
|---|---|---|
| `contradiction_resolution` | Return the current value after a fact changed | Naive systems return the most-seen or first-seen value |
| `temporal_latest` | Return the most recent value from a series | Easy to get wrong when values appear multiple times |
| `temporal_historical` | Return a past value *without* confusing for current | Inversion of latest — must know *which* time is being asked |
| `multi_hop` | Chain 2–3 facts | Each hop can introduce a stale value |
| `negative_with_history` | Something existed and was cancelled — return cancelled state | Systems that retrieved the fact at any point will say it still exists |
| `lesson_lookup` | Recall a rule learned from a past mistake | Lessons often appear once and get buried |
| `aggregation` | Collect related facts across sessions into one answer | Requires spanning the full context, not just one chunk |
| `causal_chain` | Explain *why* something happened | Requires event sequence understanding, not just state |
| `rule_application` | Apply a learned rule to a new situation | Tests generalisation, not just recall |
| `simple_lookup` | Retrieve a stable fact | Baseline; any functional system should be near 100% |

Not every scenario exercises every type — this is intentional, because real-world domains don't hit every reasoning pattern equally. Coverage across the full 200-query suite is what matters.

---

## 3. Metrics

All metrics are in `[0.0, 1.0]`. The composite is on a `0–10` scale.

### 3.1 Primary Metrics

- **`recall_at_k`** — at least one of the top-k retrieved chunks contains any expected-fact substring. Binary. The floor: if this is 0, retrieval didn't happen.
- **`precision_at_k`** — fraction of top-k results that contain at least one expected fact. Measures retrieval signal density (how much of the retrieval budget is wasted on irrelevant chunks).
- **`answer_completeness`** — fraction of expected facts present in the generated answer. Measures whether the response covers everything needed, not just part of it.
- **`temporal_accuracy`** — penalises confusing current state with historical state. `1.0` if the current value is in the answer; `0.0` if only the stale value is returned.
- **`contradiction_resolution_rate`** — on fact-contradiction pairs, fraction where the system returns the newer value.

### 3.2 Composite

```
composite = (0.25 × recall
           + 0.20 × precision
           + 0.25 × answer_completeness
           + 0.15 × temporal_accuracy
           + 0.15 × contradiction_rate) × 10
```

**Why this weighting?** Recall and answer completeness together get 50% because an agent that can't find or can't fully state the answer fails immediately. Precision (20%) is lower because in retrieval-plus-LLM pipelines, the LLM tolerates some precision loss better than recall loss. Temporal + contradiction (30% combined) are weighted high relative to their population share because they are the failure modes that matter in production — the reasons memory systems are unusable, not just suboptimal.

This is a defensible weighting, not a fact of nature. Alternate weightings are welcome, and the raw sub-scores are always reported so anyone can re-weight.

### 3.3 Trap vs. Non-Trap

Every run reports two separate averages:
- `trap_avg` — composite across queries that include a `trap`
- `nontrap_avg` — composite across queries without a trap

A system that scores high on `nontrap_avg` and low on `trap_avg` is a retrieval system, not a memory system. The gap between them is informative.

---

## 4. The Adapters

We ship **four reference adapters** so every number on the board is reproducible and comparable:

| Adapter | Purpose |
|---|---|
| `FullContextAdapter` | **Oracle ceiling.** Feeds the complete conversation history into the LLM per query. No retrieval, no forgetting. Establishes the theoretical maximum achievable with infinite context. |
| `NaiveVectorAdapter` | **Cosine-only ChromaDB.** No consolidation, no contradiction handling, no salience. The "default LangChain RAG" baseline. |
| `LangChainAdapter` | **LangChain ConversationBufferWindowMemory (k=10).** The most widely-deployed memory pattern in the wild. |
| `AgentMemoryCoreAdapter` | **Our system under test.** Full consolidation, forgetting, contradiction resolution, MMR, optional reranker. |

Adapters conform to a 3-method protocol: `ingest_turn`, `query`, `reset`. This is deliberately narrow: it is the smallest interface compatible with every known memory system architecture, and it prevents us from tilting the benchmark toward our architecture's internals.

**Fairness note.** Each competing adapter uses its own published, recommended configuration. We did not tune their hyperparameters against the benchmark. If you believe we've used a suboptimal config for a competitor, open an issue and propose a change — we will re-run.

---

## 5. Baseline Results

The most recent full-suite run (`benchmark/results/comparison.json`, generated 2026-04-15):

| Adapter | Composite | Recall@5 | Precision@5 | Answer | Temporal | Contradiction |
|---|---|---|---|---|---|---|
| FullContext (oracle ceiling) | 9.46 | 0.97 | 0.97 | 0.84 | 1.00 | — |
| archon-memory-core v0.1 | 7.7 | 0.82 | 0.61 | 0.74 | 0.71 | 0.68 |
| LangChain Window (k=10) | 8.67 (on overlapping queries) | — | — | — | — | — |
| Naive ChromaDB (cosine only) | 3.1 | 0.68 | 0.42 | 0.51 | 0.34 | 0.29 |

The FullContext adapter is the **ceiling** — what a system would score if it had perfect retrieval. Everything else is measured against that ceiling.

The interesting reading is not that archon-memory-core beats naive by ~4.6 composite points. It is that **naive ChromaDB scores 0.29 on contradiction resolution**. Nearly 3 out of 4 times the value changed, it still returned the stale one. That is the production risk pattern no one else is pricing in.

---

## 6. Known Limitations

### 6.1 Hand-Authored Scenarios → Small N

10 scenarios × 20 queries = 200 query points. That is a strong signal for a one-shot comparison but **not** enough to distinguish two systems that score within ~0.3 composite of each other. Read close scores as a tie. Read gaps of >1.0 composite as real.

### 6.2 English-Only

All scenarios are in English. Multilingual expansion is on the Q4 2026 roadmap.

### 6.3 Not Longitudinal

AMB v1 runs one pass per scenario. It does not test **decay over months**. This is the most-requested improvement and is the headline feature of AMB v2 (see §8).

### 6.4 Composite Weighting Is an Opinion

The 25/20/25/15/15 weighting is defensible but not canonical. We always report the sub-scores so you can re-weight. If you believe the weights are wrong, submit a PR to `benchmark/metrics.py` with your proposed weights *and* a written justification — we will consider it openly.

### 6.5 Author Bias Risk

The scenarios were authored by the team that also built the primary system under test. This is a legitimate concern. Our mitigations:

1. Every scenario JSON is in the repo — anyone can inspect for patterns that favour our architecture.
2. We run the **FullContext oracle** on every submission to bound the ceiling. If a new system beats our ceiling, we made the test too easy, not too hard.
3. For AMB v2, a non-author reviewer ratifies the scenario design before the benchmark is frozen. v2 will be pre-registered (scenarios + protocol locked before any adapter runs) to eliminate the "tune after seeing the data" critique.

### 6.6 No LLM-Judged Answers Yet

Answer completeness is currently measured by substring containment of expected facts. This is precise but rigid (a system that gives a paraphrase can underscore). LLM-judged completeness (with a cross-model grader to avoid self-favouritism) is tracked for AMB v1.1.

---

## 7. Reproducing Our Numbers

Every number in the README and in §5 above is reproducible from the public repo.

```bash
git clone https://github.com/atw4757-byte/archon-memory-core
cd archon-memory-core
pip install -e ".[dev]"

# Run the full suite against all 4 reference adapters
python benchmark/run_all.py

# Or a single adapter
python benchmark/run_benchmark.py --adapter archon_memory_core

# Or a single scenario
python benchmark/run_benchmark.py --adapter archon_memory_core --scenario 01_personal_assistant

# Results go to benchmark/results/<adapter>-<YYYY-MM-DD>.json
```

Expected runtime for the full suite on a 2023 MacBook Pro M2: ~8 minutes (adapters are fast; the ingest loop is the bulk of it). No GPU required.

If your score deviates from ours by more than ±0.05 composite on the same adapter version, open an issue with your result JSON. We will investigate — it is always either a real bug or an environmental difference, and both are worth knowing.

---

## 8. AMB v2 — What's Changing

AMB v1 is a **snapshot** benchmark. AMB v2 will be **longitudinal**: simulated 90-day conversation histories with forgetting pressure, session-to-session consolidation cycles, and re-asked queries at different time horizons to measure decay.

Key protocol changes in v2:

1. **Pre-registration.** Scenarios, queries, and the scoring harness are frozen and timestamped before any system is scored against them. No post-hoc tuning.
2. **Held-out test set.** A public training set (seen by authors) and a held-out test set (never seen) with the final score reported on held-out only.
3. **Non-author ratification.** An external reviewer signs off on scenario design before freeze.
4. **Decay-at-T measurement.** The same query is asked at T+7d, T+30d, T+90d of simulated usage. The score differential is the "decay slope" — a new headline metric.
5. **LLM-judged completeness.** Dual-model grader (Claude + GPT) with disagreement flagged for human review.
6. **Neutral hardware.** Submissions are re-run on a fixed cloud spec to prevent hardware-dependent results.

v2 launch target: **2026-05-15**, public leaderboard at `amb.divergencerouter.com`.

---

## 9. Challenging the Benchmark

Good benchmarks welcome challenges. If you believe:

- **A scenario is unfair** — open an issue describing why, with a specific example.
- **A metric is miscalibrated** — submit a PR with your proposed change to `benchmark/metrics.py` and a short justification.
- **Our adapter for a competing system is suboptimal** — submit a PR with your preferred config; we re-run and update the board.
- **An entirely different evaluation design would be better** — we are interested in hosting multiple benchmarks, not defending one. Propose it.

The point of AMB is not to win — it is to have *any* public, auditable, reproducible way to compare long-horizon memory systems. We would rather be second place on a rigorous benchmark than first place on a brittle one.

---

## 10. Citation

If you reference AMB in a paper, report, or blog post:

```
@software{agentic_memory_benchmark_2026,
  author = {archon-memory-core contributors},
  title = {Agentic Memory Benchmark (AMB) v1},
  year = {2026},
  url = {https://github.com/atw4757-byte/archon-memory-core},
  note = {See docs/EVALUATION.md for methodology}
}
```

---

## Appendix — File Map

| Path | Contents |
|---|---|
| `benchmark/scenarios/*.json` | The 10 scenario files. Inspect freely. |
| `benchmark/metrics.py` | All scoring logic. ~380 lines, no hidden config. |
| `benchmark/run_benchmark.py` | Harness for one adapter / one or more scenarios. |
| `benchmark/run_all.py` | Orchestrator that runs every reference adapter. |
| `benchmark/adapters/` | 4 reference adapters. |
| `benchmark/results/` | Every run's raw JSON. Nothing deleted. |
| `docs/EVALUATION.md` | This file. |
| `src/archon_memory_core/eval.py` | **Separate** lightweight eval harness for users to run against their own memory (30 stratified default queries, recall/precision/answer). Not used for AMB scoring. |

One note on the `src/archon_memory_core/eval.py` module: it is a **user-facing** eval tool shipped with the library so anyone can measure retrieval quality against their own ingested data. It uses a 30-query default suite that is separate from the AMB benchmark. Do not confuse a score from this module with an AMB score. AMB scores come only from `benchmark/run_benchmark.py`.
