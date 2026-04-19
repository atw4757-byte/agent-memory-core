---
status: v2.3 muddiness smoke — primitive thesis CONFIRMED
date: 2026-04-19
scenario: mini-v23 (per-query confusers + top-1 metric)
---

# AMB v2.3 — muddiness smoke results (3 seeds, mini-v23)

## Setup

User challenge: v2.2 showed naive word-overlap ceiling-hitting 90.00 AUC,
but real production memory gets muddy. Benchmark wasn't capturing
muddiness because corpus had distinctive vocabulary and substring-match
gave full credit for any-of-top-5.

v2.3 changes:
- Added per-query **confusers** to `ScenarioBundle` (chunks that share
  vocabulary with the query but contain wrong/no answer)
- Added **top1_answer_accuracy** metric (penalizes LLM-misattribution)
- Added **confuser_resistance** metric (fraction with no confuser in top-k)
- Added **quality_v2_3** composite (0.25·top1 + 0.15·any + 0.20·contradiction
  + 0.10·(1−stale) + 0.10·salience + 0.20·confuser_resist)
- Authored `mini-v23.json` with 27 confusers across 4 queries

No adapter changes. Same naive word-overlap, amc embedding search,
langchain-dump FIFO budget.

## Results (day-90, mean across seeds 42/43/44)

| adapter | mode | legacy AUC | any-acc | top1-acc | quality_v2_3 |
|---|---|---|---|---|---|
| naive-append-only | stock | 25.48 | 0.25 | 0.00 | **0.19** |
| naive-append-only | tuned | 25.48 | 0.25 | 0.00 | **0.19** |
| langchain-dump | stock | 90.00 | 1.00 | 0.00 | **0.55** |
| langchain-dump | tuned | 90.00 | 1.00 | 0.00 | **0.55** |
| agent-memory-core | stock | 90.00 | 1.00 | 0.75 | **0.74** |
| **agent-memory-core** | **tuned** | **90.00** | **1.00** | **1.00** | **0.80** |

## Findings

**Primitive thesis CONFIRMED under muddy corpus.**

- `amc-tuned` beats `naive` by **+0.61** on quality_v2_3 (0.80 vs 0.19).
  This is not noise. Naive's word-overlap retrieval catastrophically
  fails when confusers share vocabulary with queries — top-5 fills with
  confusers, the real answer gets pushed out.

- `amc-tuned` beats `langchain-dump` by **+0.25** on quality_v2_3 (0.80
  vs 0.55). Dumping all chunks into context gives any-answer=1.00
  (answer is somewhere in the blob) but top1=0.00 (wrong chunk at rank 1).
  Real LLM generation biased toward the top chunk would misattribute.

- Legacy AUC (using old any-of-top-5 metric) still shows naive crater:
  25.48 vs 90.00. Naive's failure is large enough to hit even the
  lenient metric once confusers exist.

**What this captures that v2.2 couldn't:**

The muddiness Andy described — "memory gets muddy, many things are
forgotten over time" — manifests in v2.3 as: when vocabulary-overlapping
chunks compete for retrieval, word-overlap retrieval picks the wrong
ones. Embedding-based retrieval (amc) distinguishes semantic relevance
from surface overlap. The consolidator removes superseded entries
entirely, so the trap chunk is no longer retrievable at all — that's
why amc-tuned hits top-1 = 1.00 while amc-stock sits at 0.75.

## Product claim after v2.3

> **At retrieval-pool densities where vocabulary collisions exist,
> embedding-based memory with supersede-aware consolidation beats
> word-overlap-top-k by a large margin on top-1 accuracy — the metric
> that matches real LLM generation behavior.**

This is defensible. The experiment is reproducible with the public
`mini-v23.json` scenario; the adapter comparison uses no tuning beyond
mode flags.

## Limitations / next work

- Only 4 queries × 27 confusers. Needs M/L scale with hundreds of
  queries and thousands of confusers to be publishable.
- Confusers authored by hand. Generating programmatically from scenario
  vocabulary would scale better.
- top-1 is a proxy for LLM misattribution; real LLM-judged accuracy is
  the gold standard. Expensive but next-version work.
- amc-stock top-1 at 0.75 (not 1.00) means embedding retrieval catches
  3 of 4 queries but flubs one. Worth diagnosing which one before
  publishing.
