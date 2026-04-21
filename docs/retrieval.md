# Adaptive retrieval policy

`MemoryStore.search()` picks one of three scoring policies based on corpus size. The reason: salience, keyword-boosting, and cross-encoder re-ranking all need enough variance across stored items to be useful. On a tiny corpus they add noise; on a large one they add signal.

## Modes

**lightweight** — corpus <100 chunks

Pure normalized vector distance. No salience, no recency, no keyword boost, no cross-encoder. Consolidated summaries get a small penalty (+0.05) so raw source chunks win ties.

**standard** — corpus 100–4,999 chunks

Light salience + recency weighting: similarity 0.70, recency 0.15, salience 0.15. Keyword boost on. Cross-encoder enabled once the corpus crosses 200. No CE relevance gate.

**full** — corpus ≥5,000 chunks

Everything on. Similarity 0.50, recency 0.20, salience 0.30 baseline — weights then re-tuned per query by `detect_query_type()` (credential / lesson / personal / project_status / technical / session / default). Keyword boost, cross-encoder re-ranking, CE relevance gate.

## Why the cutoffs

100 and 5,000 are empirical. Below 100, we found that layered scoring hurt top-1 accuracy on the AMB v2.3 benchmark — nothing beats raw cosine when you have no statistical surface to bite into. Above 5,000, the cross-encoder's latency cost is amortized across enough candidates that the precision gain becomes clearly worth it.

If you're benchmarking and want deterministic scoring, pin one mode explicitly rather than letting `search()` pick.
