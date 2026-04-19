---
status: v2.3 small smoke
date: 2026-04-19
scenario: small-v23 (20 queries × 156 confusers, 32 timeline events)
seeds: 42
---

# AMB v2.3 — small smoke (seed 42)

## Corpus
- 4 distinct user identities × 5 archetypes
- 20 queries, each a day-0/day-15 contradiction
- 156 confusers (~8 per query)

## Results (day-90)

| adapter | mode | auc | any@90 | top1@90 | conf_resist@90 | quality_v2_3 |
|---|---|---|---|---|---|---|
| naive-append-only | stock | 36.98 | 0.00 | 0.00 | 0.39 | **0.28** |
| naive-append-only | tuned | 36.98 | 0.00 | 0.00 | 0.39 | **0.28** |
| langchain-dump | stock | 88.60 | 1.00 | 0.00 | 0.00 | **0.55** |
| langchain-dump | tuned | 88.60 | 1.00 | 0.00 | 0.00 | **0.55** |
| agent-memory-core | stock | 80.61 | 0.90 | 0.50 | 0.25 | **0.68** |
| **agent-memory-core** | **tuned** | **88.60** | **1.00** | **1.00** | **0.15** | **0.83** |

## Read

- 20-query scale is too small for langchain-dump stock budget to overflow
  (content fits in 8k). Medium scale exposes the budget collapse.
- Signal direction matches mini and medium: amc-tuned beats naive by
  +0.55 on quality_v2_3.
- Confuser resistance is inversely correlated with recall at this scale —
  naive resists confusers because it retrieves nothing useful, amc
  returns them alongside the real answer. This is expected and is why
  top-1 is the key metric, not confuser_resistance in isolation.
