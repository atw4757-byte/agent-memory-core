---
status: draft
date: 2026-04-19
supersedes_stress_metric: v2.2 substring-match-any-of-top-5
---

# AMB v2.3 — design for muddiness

## The gap v2.2 exposed

v2.2 regime L result: naive word-overlap top-5 ceiling-hits 90.00 AUC. amc
gets no lift over naive. Andy's lived experience: his production memory
system gets muddy, forgets things, confuses facts over time.

Both true. The benchmark corpus isn't muddy.

## Why v2.2 isn't muddy

1. **Scenario queries are distinctive.** "What is user's home address?"
   word-overlaps only with the two address chunks (trap + answer). Nothing
   else in the corpus contains "home address" as a phrase.

2. **Noise/filler use disjoint vocabulary.** NOISE_TEMPLATES and
   FILLER_FACT_TEMPLATES were authored to be lexically-loaded-but-distinct
   — Marcus, Lena, Chronos. The scenario entities (user, dog, wifi) never
   appear in noise. So noise never outranks scenario chunks in word-overlap
   scoring.

3. **Substring-match gives full credit for "answer appears in top-5."**
   If trap + answer are both retrieved, metric says correct because the
   answer substring is present. Real LLMs would misattribute.

4. **One fact per concept.** Mini scenario has ONE home address. Real
   memory has 10 addresses across time (home, work, prior, mailing, parents').

## v2.3 changes

### Corpus: add confusers

New field on `ScenarioBundle`: `confusers: list[ConfuserSpec]`.

Each confuser specifies:
- `query_id` — the query this confuser targets
- `texts` — 3-10 chunk texts that share vocabulary with the query but
  contain wrong or no answer
- `day_range` — when they can appear (typically span query's eligibility)

Example for the home-address query:
```
{
  "query_id": "mini-q1",
  "texts": [
    "User mentioned that home is where the heart is.",
    "User's home base for the conference was the Airbnb.",
    "User's home internet speed is slow on weekends.",
    "User's home state is originally California.",
    "User dreams about buying a second home someday."
  ]
}
```

All share {user, home} with the query but none contain the actual address.
They OUTNUMBER the one correct chunk, so top-5 will likely include them.

### Metric: top-1 retrieval accuracy

Current: `answer_accuracy` = substring match over ALL returned text.
Returns 1.0 if answer appears anywhere in the joined top-5.

New metric: `top1_answer_accuracy` = substring match over ONLY the
highest-ranked chunk. Simulates LLM misattribution — if top-1 is the trap
or a confuser, wrong.

Keep old metric for backwards-compat and cross-v2 comparison. New composite
quality weights:
```
quality_v2_3 = 0.25·top1_answer + 0.20·any_answer + 0.25·contradiction
             + 0.15·(1−stale) + 0.15·salience
```

### Metric: confuser-resistance rate

Fraction of queries where NO confuser chunk appears in top-5. Tests
whether the retriever can filter semantically-off chunks that share
surface vocabulary.

### Scenarios: add multi-address, multi-employer history

Redesign mini scenario (or add a new one) so the user has 3 home addresses
across 180 days with temporal precision needed. Queries like "where did
the user live when they worked at Exotec?" require joining time + employer
+ location across chunks.

### Queries: paraphrase variants

Each query gets 3 phrasings:
- Direct: "what is the user's home address?"
- Paraphrase: "where does the user live?"
- Colloquial: "user's place?"

Score on average across phrasings — tests robustness to surface variation.

## What this should produce

Hypothesis: with confusers + top-1 metric, naive word-overlap drops
substantially at L. amc (embedding-based retrieval with MemoryStore.search)
should hold better because embeddings pick semantic similarity, not
surface vocabulary.

If amc STILL ties naive under v2.3, that's a real falsification of the
primitive thesis and the product claim needs to change accordingly.

## Implementation order (TDD)

1. Add `confusers` to ScenarioBundle + loader tests
2. Add `top1_answer_accuracy` to metrics.py + tests
3. Add `confuser_resistance` to metrics.py + tests
4. Add quality_v2_3 composite + tests
5. Author mini-v23 scenario with confusers + paraphrases
6. Regenerate M/L corpora with v2.3 fixtures
7. Run full grid on v2.3 corpus, compare to v2.2 baseline

## Scope

v2.3 is a metric + corpus change, not an adapter change. Adapters stay
the same. What we're testing is whether the benchmark can **see** the
muddiness the user lives with. If it still can't, the benchmark is
structurally wrong and we rebuild in v3.

## Open question

LLM-judged accuracy (run a real small model over returned context,
judge the generation) is the gold standard but expensive. For v2.3 we
approximate with top-1. For v3 we bite the LLM cost and do it right.
