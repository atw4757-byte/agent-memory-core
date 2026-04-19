---
status: regime L preview complete — thesis partially falsified, partially confirmed
preregister_commit: c6bb499
simulator_commit: 8af76d4
date: 2026-04-19
---

## Pre-run corrections (per preregister §"What could make us change this spec BEFORE first run")

1. **Seeds typo in preregister:** preregister §"Scale regimes" claims seeds
   "17, 31, 53 (same as v2.1)." v2.1 actually used `42, 43, 44`. Regime M/L
   grids use the real v2.1 seeds for apples-to-apples.

2. **Regime M launched 2026-04-19 07:55 ET, completed 08:01 ET** — 18 cells,
   noise=0.60, filler=5, seeds 42/43/44.

3. **Adapter fakery caught mid-run (2026-04-19):** see below.

---

# AMB v2.2 — status board

## Regime M results (noise=0.60, filler=5, ~468 chunks/run)

| adapter | mode | AUC avg | Q@90 | AA | CR | stale |
|---|---|---|---|---|---|---|
| agent-memory-core | stock | 50.61 | 0.836 | 0.853 | 0.700 | 0.100 |
| **agent-memory-core** | **tuned** | **55.48** | **0.911** | 0.853 | 0.900 | 0.000 |
| langchain-buffer | stock | 54.74 | 0.940 | 0.912 | 1.000 | 0.000 |
| langchain-buffer | tuned | 35.46 | 0.341 | 0.176 | 0.400 | 0.000 |
| naive-append-only | stock | 54.74 | 0.940 | 0.912 | 1.000 | 0.000 |
| naive-append-only | tuned | 54.74 | 0.940 | 0.912 | 1.000 | 0.000 |

### Primitive thesis at M: amc-tuned +0.74 AUC over naive

Directional flip from v2.1. At M scale the supersedes-aware consolidator
pulls ahead of word-overlap-top-5. Not the landslide you'd want for a
product claim, but the sign is right.

## Adapter audit (caught 2026-04-19)

**User challenge:** *"if the test simulates more than 10 sessions it is
impossible langchain doesn't break."*

**Finding:** The user was right. Both `langchain-buffer` and
`naive-append-only` were doing byte-identical word-overlap-top-5
retrieval. Proof in the table above: stock AUCs per-seed are byte-equal
(54.94/54.86/54.41). The "langchain" label was cosmetic — `ChatMessageHistory`
was stored but never read back at query time. v2.1's 1.67-AUC gap between
them was noise between nearly-identical retrievers.

`langchain-buffer tuned` does real work (drops chunks >30 days old) and
the work is actively bad — -19 AUC. That's a meaningful negative result
(a dumb heuristic that evicts still-valid contradictions tanks accuracy)
but it's not "real LangChain buffer behavior" either.

### Fix: langchain-dump (new, 2026-04-19)

Honest `ConversationTokenBufferMemory` simulation:
- Stores every chunk, no smart retrieval
- At query time, concatenates newest-first up to a token budget
- Stock: 8k tokens (typical small-model context)
- Tuned: 32k tokens (generous frontier-ish context)
- FIFO eviction when over budget (the real failure mode)

Tests: 11/11 green. Committed via langchain_dump.py + test_langchain_dump.py.

### langchain-dump at regime M: 90.00 AUC (perfect)

**This is expected.** Regime M = 468 chunks × ~60 chars = ~7k tokens total.
Both 8k stock and 32k tuned budgets fit every chunk, so dump returns
everything, expected answer is always present → substring match always
succeeds → quality = 1.0.

**This is precisely why regime L matters.** At L (~4600 chunks, ~69k tokens),
stock budget holds ~4% of chunks and tuned holds ~17%. Now FIFO eviction
bites hard — if the answer was ingested day 5 and day 90 is the checkpoint,
the answer is gone. That's where a real memory primitive should beat
"dump into prompt." The M-dump result being trivially perfect is the
benchmark honestly reporting: at this scale, context-window-first
strategies work fine. Scale matters.

## Regime L preview results (seed=42, noise=0.90, filler=50, ~4600 chunks)

| adapter | mode | AUC | Q@90 | AA | CR | stale |
|---|---|---|---|---|---|---|
| agent-memory-core | stock | **90.00** | 1.000 | 1.000 | 1.000 | 0.000 |
| agent-memory-core | tuned | **90.00** | 1.000 | 1.000 | 1.000 | 0.000 |
| langchain-dump | stock | 22.43 | 0.150 | 0.000 | 0.000 | 0.000 |
| langchain-dump | tuned | 51.75 | 0.150 | 0.000 | 0.000 | 0.000 |
| naive-append-only | stock | **90.00** | 1.000 | 1.000 | 1.000 | 0.000 |
| naive-append-only | tuned | **90.00** | 1.000 | 1.000 | 1.000 | 0.000 |

### What this means

**Thesis CONFIRMED (user's intuition):** Context-dumping breaks at scale.
`langchain-dump` stock collapsed from 90.00 at M → 22.43 at L. Tuned
(32k budget) recovered only partially to 51.75. FIFO eviction kicked out
the chunks containing the answers. At L, the 8k budget holds ~4% of the
corpus and the 32k holds ~17% — not enough. **This is the real failure
mode of real ConversationBufferMemory at scale.**

**Thesis FALSIFIED (primitive claim):** Word-overlap top-5 retrieval did
NOT break at L. Naive and amc both ceiling-hit 90.00 AUC. At 4600 chunks,
selective top-k retrieval still pulls the right answer out. amc-tuned
provides zero lift over naive at this scale with this metric.

### P1-P4 accounting against preregister

- **P1** (naive must drop ≥8 AUC S→L): naive stock at M = 54.74, at L = 90.00.
  **P1 FAIL.** Naive *rose* 35 points, the opposite of the preregistered
  prediction. Per preregister §"What changes if this fails": the memory-as-a-
  library product framing needs redaction. Word-overlap top-5 scales fine
  at this corpus density. The benchmark's metric or corpus density can't
  distinguish strategies at L.

- **P2** (amc-tuned > naive by ≥5 AUC at L, p<0.05): both at 90.00 ceiling.
  **P2 FAIL.** Cannot claim primitive lift over naive at L.

- **P3** (publish crossover regardless): will do — the crossover is between
  `langchain-dump` (context-dumping) and *any* top-k retriever, not between
  naive and amc.

- **P4** (consolidate fires on ≥80% of contradiction targets at L):
  irrelevant — consolidation didn't move the needle because top-k already
  retrieves the right chunk.

### Honest product claim after L

- "Top-k retrieval > dumping into context at scale" — CONFIRMED, +38 to
  +68 AUC points depending on budget. This is the story worth telling.
- "Supersedes-aware memory primitive > baseline top-k retrieval" —
  NOT SHOWN at this benchmark's scale/metric. The ceiling effect blocks
  discrimination.

### Why the metric is ceilinged

Substring-match answer accuracy = "expected string appears in returned
text." Top-5 × 4600-chunk corpus = 5 chunks returned of ~50 chars each =
~250 chars. If the answer is in ANY of those 5 retrieved chunks,
accuracy = 1.0. At current corpus density, word-overlap retrieval pulls
the correct chunk >99% of the time.

To discriminate smart-vs-basic retrieval, future work needs either:
1. **Harder queries** — queries whose answers require multi-chunk reasoning,
   not single-chunk lookup.
2. **LLM-judged accuracy** — real attention failure on long contexts,
   not substring proxy.
3. **Tighter retrieval budget** — top-1 instead of top-5 would surface
   ranking quality differences.

### Next

- Extend L to seeds 43/44 only for `langchain-dump` (the adapter that
  actually varies). Naive/amc will hit the same ceiling, no new info.
- Write v2.2 REPORT documenting: thesis confirmed for context-dumping,
  falsified for "our primitive beats retrieval." Publish honestly.
- For v2.3: add top-1 metric + multi-chunk-reasoning queries to break
  the ceiling.

The preregister is binding. Whatever it says is what gets published,
on the thresholds that are already locked.
