---
status: v2.3 medium grid — primitive thesis HOLDS at scale
date: 2026-04-19
scenario: medium-v23 (75 queries × 660 confusers, 120 timeline events)
seeds: 42, 43, 44
---

# AMB v2.3 — medium results (3 seeds)

## Corpus
- 15 distinct user identities × 5 archetypes (home address, wifi password,
  dog name, coffee preference, employer)
- 75 queries, each a contradiction (original fact day 0, update day 15)
- 660 confusers (~9 per query) at day 4 and day 21 — vocabulary-overlapping,
  answer-absent
- 120 timeline events + ~330 noise chunks at 0.45 noise_rate

## Results (day-90, mean across seeds 42/43/44)

| adapter | mode | legacy AUC | any@90 | top1@90 | conf_resist@90 | quality_v2_3 |
|---|---|---|---|---|---|---|
| naive-append-only | stock | 33.90 | 0.027 | 0.000 | 0.394 | **0.283** |
| naive-append-only | tuned | 33.90 | 0.027 | 0.000 | 0.394 | **0.283** |
| langchain-dump | stock | 78.10 | 0.000 | 0.000 | 0.000 | **0.200** |
| langchain-dump | tuned | 88.60 | 1.000 | 0.000 | 0.000 | **0.550** |
| agent-memory-core | stock | 80.36 | 0.893 | 0.493 | 0.240 | **0.670** |
| **agent-memory-core** | **tuned** | **88.60** | **1.000** | **1.000** | **0.093** | **0.819** |

## Lift (amc-tuned vs baselines)

| vs | Δ quality_v2_3 |
|---|---|
| naive (stock or tuned) | **+0.536** |
| langchain-dump stock (8k budget) | **+0.619** |
| langchain-dump tuned (32k budget) | **+0.269** |
| amc-stock (no consolidator) | **+0.149** |

## Findings

**Primitive thesis CONFIRMED at M scale.**

- `amc-tuned` holds at **top1=1.000** across all 75 queries × 3 seeds. The
  supersede-aware consolidator deletes the trap chunk, so retrieval can
  never return it. This is the expected behavior of a production memory
  primitive that reasons about contradiction.

- `amc-stock` at **top1=0.493** is the honest retrieval-only baseline —
  embedding search picks the right chunk about half the time, but without
  consolidation the trap chunk still exists and competes for top-1. This
  is the headroom that consolidation buys.

- `langchain-dump stock` **collapses to any=0.000** — the 8k token budget
  can't hold 120+ scenario events + 660 confusers + noise, so FIFO
  eviction purges answer chunks entirely. This is the adapter-fakery
  falsification that pre-v2.2 versions masked by unrealistic "unlimited"
  context.

- `langchain-dump tuned` (32k budget) holds any=1.000 but top1=0.000 —
  the answer is in the blob, at the wrong position. Real LLM generation
  biased toward the top chunk would misattribute 100% of the time.

- `naive-append-only` is catastrophic: **top1=0.000, any=0.027**. Word-
  overlap top-5 is saturated by vocabulary-overlapping confusers. The
  answer chunks are almost never in the returned set.

## Reproduction

```bash
# Generate corpus (deterministic)
python benchmark/amb_v2/scenarios-v23/_generate.py \
  --size medium --out-dir benchmark/amb_v2/scenarios-v23/medium --seed 42

# Run grid (3 seeds, all 3 adapters × both modes)
python -m benchmark.amb_v2.run_all \
  --scenarios benchmark/amb_v2/scenarios-v23/medium \
  --out-dir benchmark/amb_v2/results/v2.3/medium \
  --adapters naive,agent-memory-core,langchain-dump \
  --modes stock,tuned --seeds 42,43,44 --noise-rates 0.45
```

Runtime: ~9 minutes on Mac Mini M4 Pro.

## Product claim after v2.3 medium

> **At production-realistic corpus density (75 queries × 660 semantically
> confusable distractors), embedding-based memory with supersede-aware
> consolidation beats word-overlap retrieval by +0.54 on quality_v2_3 and
> top-k context-dump by +0.27. top-1 accuracy — the metric that matches
> real LLM generation behavior — is 1.000 for amc-tuned versus 0.000 for
> every non-retriever baseline.**

## Next: large-v23 grid running in background

250 queries × 2300 confusers × 3 seeds × 6 adapter-mode combos. Expected
~30 minutes.
