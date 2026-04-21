---
status: staged
for: news.ycombinator.com (Show HN)
created: 2026-04-20
staged_on: 2026-04-21
submit_when: Tuesday 2026-04-21 evening ET
tag: v0.1.3
author: atw4757-byte
last_verified: 2026-04-21
---

# Show HN submission package — archon-memory-core

## Title (80 char max — HN truncates aggressively)

**Option A (recommended — leads with the number):**
> Show HN: A memory system for AI agents — 99.2% vs 0% on a 90-day benchmark

**Option B (leads with the failure mode):**
> Show HN: Every agent framework ships with broken memory. Here's the benchmark.

**Option C (leads with the fix):**
> Show HN: Supersede-aware memory for AI agents, with a preregistered benchmark

**Pick A.** Numbers hook HN. "99.2% vs 0%" is the conversation-starter.

---

## Post body

> Frontier labs are racing to build agents on top of context windows. We ran the obvious experiment — what happens to agent memory over a 90-day horizon when confusers start sharing vocabulary with the query?
>
> - **archon-memory-core (ours, with supersede-aware consolidation): 99.2% top-1 accuracy**
> - **Same retriever, no consolidation: 49.2%** (drops once at day 14, stays there)
> - **LangChain 32k context dump: 0% top-1** (the answer is in the buffer — the LLM just attends to the wrong chunk)
> - **Naive word-overlap retrieval: ~0%**
>
> 3-seed grid at scale L (250 queries × 2,300 confusers across a simulated 90-day horizon). Fully preregistered: https://github.com/atw4757-byte/archon-memory-core/blob/main/benchmark/amb_v2/PREREGISTERED.md
>
> The core insight — and the reason frontier labs don't ship this — is that memory is plumbing. Plumbing doesn't sell GPU hours. Bigger context windows do. So the industry defaults to "just dump more into context" even when the benchmark shows context length ≠ memory without ranking.
>
> What `archon-memory-core` does differently:
>
> 1. **Supersede-aware consolidation.** When a new fact contradicts an old one, the old fact is archived with a link to what replaced it. Not left to compete at retrieval time.
> 2. **Ranked top-1 retrieval.** Buffer memory returns chunks in recency order, so an LLM attending to "position 1" gets whatever arrived last — often a confuser. AMC returns a ranked top-1 that actually earns the slot.
> 3. **Preregistered adversarial benchmark (AMB v2.3).** Per-query *confusers* — vocabulary-overlapping distractors that force the retriever to actually rank, not just lexically match. Four corpus scales (mini/small/medium/large: 4/20/75/250 queries, with 27/156/660/2,300 confusers respectively). 3 seeds at mini/medium/large. Composite `quality_v2_3` metric weights top-1 accuracy, any-answer accuracy, confuser resistance, contradiction resolution, stale-fact rate, and salience preservation.
>
> `pip install archon-memory-core` — Apache 2.0. Benchmark harness + results + simulator included.
>
> Happy to answer questions about methodology, the adversarial setup, or why the LangChain 32k dump scores 0% on top-1 even when the answer is verifiably in the buffer.
>
> Benchmark details: https://divergencerouter.com/amc/
> Repo: https://github.com/atw4757-byte/archon-memory-core

---

## First comment (post 30–60 seconds after submission — the expected HN ritual)

> Author here. Three things I expect HN to ask:
>
> **"Isn't this just RAG?"** Partial. Retrieval is the easy half. The hard half is what happens when retrieval returns a confuser as top-1 — which is the v2.3 thesis and what the benchmark measures. We also resolve contradictions at ingest time, not at query time, which is the piece most RAG systems skip.
>
> **"Why not just use a bigger context window?"** That's the comparison in the chart. LangChain-style 32k dump scores 0% top-1 because buffer memory has no ranking — the LLM defaults to the most-prominent chunk, which is noise after confusers land. "Just use more context" is what the industry defaults to and it's what the benchmark falsifies.
>
> **"Why should I trust the numbers?"** Everything is preregistered. Seeds published (42/43/44). Scenarios, queries, confusers, and adapter code are all in the repo. See [`REPRODUCE.md`](https://github.com/atw4757-byte/archon-memory-core/blob/main/REPRODUCE.md) for copy-paste commands per scale — the large-v23 grid (the one behind the 99.2% / 0% chart) runs in ~20 min on a laptop.

---

## OG image / hero asset

Already live: https://divergencerouter.com/images/amb-v23-degradation.png

Three lines + bracket annotation. Matches the claim.

---

## Pre-flight checklist (run through before submitting)

- [x] v0.2.0 tagged and pushed (2026-04-21)
- [x] 50/50 tests pass on v0.2.0
- [x] Public API unchanged from v0.1.3
- [x] PREREGISTERED.md exists at benchmark/amb_v2/PREREGISTERED.md
- [ ] Verify `pip install archon-memory-core==0.2.0` actually installs on a fresh Python 3.11 env (PyPI publish needed)
- [ ] `REPRODUCE.md` large-v23 command reproduces chart numbers (amc-tuned top1@90 = 0.992, langchain-dump = 0.000, naive = 0.000) on a clean clone
- [ ] https://divergencerouter.com/amc/ loads on mobile (Safari iOS) with the chart visible
- [ ] https://github.com/atw4757-byte/archon-memory-core README matches the Show HN claims
- [ ] HN account has enough karma to avoid auto-flag (check with `archon-hn status`)
- [ ] Set a calendar block at submission + 3h to respond to comments — first two hours determine ranking

---

## After submission — response templates

**Skeptic ("benchmark is cherry-picked"):**
> Fair challenge. The preregistration was committed before the grid ran — see commit history. Scenarios and confusers are deterministic from seed. Rerun with a seed we haven't tested and tell me what you get.

**Adopter ("how do I use this in my agent loop?"):**
> `docs/INTEGRATIONS.md` has the LangChain + LlamaIndex + raw-API adapters. 5-line integration for the common case. Happy to help with specifics if you drop a snippet.

**Academic ("what's the theoretical contribution?"):**
> The contribution is the adversarial benchmark design — per-query confusers that force the retriever to rank under vocabulary overlap — and the empirical falsification of "bigger context = memory." The consolidation algorithm itself is straightforward supersede-aware archival. Preregistration, seeds, scenarios, and full results are in the repo for anyone who wants to rerun or extend.

**Competitor ("doesn't [tool X] already do this?"):**
> AMC is what falls out when you take the v2.3 benchmark seriously. If tool X scores 99%+ top-1 under the same preregistered setup, that's great — please submit to the leaderboard. That's the whole point of preregistering.

---

## What NOT to say

- Don't claim "we solved memory." We solved one measurable slice. Say exactly that.
- Don't pick fights with LangChain. The benchmark speaks; we don't need to.
- Don't shade the labs. "Frontier labs ignore memory" is a framing, not an accusation.
- Don't over-respond to hostile comments. Two responses max per thread. Let HN weigh in.
