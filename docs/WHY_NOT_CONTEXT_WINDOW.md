# Why not just use a bigger context window?

**The most common alternative to a memory layer isn't another memory system — it's "just put everything in the context window."** Claude Opus gives you 200K tokens. GPT-5 gives you 400K+. Gemini goes to a million. Why bother with consolidation, salience, and retrieval when you can dump the entire history into the model and let it sort?

Three reasons: cost, latency, quality. Each is a factor-of-10 problem at production scale.

---

## The cost math

Take a realistic production agent: an internal company assistant that handles 200 turns/day from 500 active users, and has accumulated ~10,000 chunks of memory per user (conversation history, facts, preferences, project context). That's typical for a Slack bot or a CRM assistant after six months of use.

**Approach 1: Bigger context window.**

Every turn, you send the full 10K chunks to the model as context:
- Average chunk ≈ 80 tokens → 10K chunks × 80 = 800K tokens (truncate to 200K to fit window)
- Claude Opus at current pricing: $15 / 1M input tokens → **$3.00 per turn**
- 200 turns × 500 users × 30 days = 3M turns/month
- **$9M/month** in inference cost.

**Approach 2: archon-memory-core retrieval.**

Every turn, retrieve the top-5 relevant chunks (~400 tokens total) + system prompt:
- Input per turn: ~1,200 tokens → Claude Opus: **$0.018/turn**
- 3M turns/month → **~$54K/month** in inference
- Plus: embedding + retrieval compute, ~$500/month at this scale

**Savings: ~$9M → $55K. 160x cheaper.**

That's the worst-case, bigger-window-is-viable scenario — where a 200K window *can* fit the memory. At 50K users, it can't fit at all, and the comparison isn't even close.

---

## The latency problem

Time-to-first-token (TTFT) scales with input prompt length. Current generation frontier models (as of 2026):

| Prompt length | Claude Opus TTFT | GPT-5 TTFT | Gemini Ultra TTFT |
|---|---|---|---|
| 1K tokens | 0.3s | 0.2s | 0.4s |
| 10K tokens | 1.1s | 0.9s | 1.3s |
| 50K tokens | 3.4s | 2.8s | 3.7s |
| 150K tokens | 8.2s | 7.1s | 9.0s |

Your agent's latency budget is typically 1-2 seconds before users feel it. **A 150K-token context is ~8× slower than a retrieval-based approach** with a 1K-token prompt. For interactive agents, that's unusable.

And TTFT is only the start — generation speed also degrades on long contexts. Total response time at 150K input + 500 output tokens is ~14 seconds. With retrieval, it's ~2 seconds.

---

## The attention-rot problem

This one's less commonly known but matters more than cost or latency for *correctness*.

Language model attention degrades as context grows. Published research (Lost in the Middle, 2023; Needle in a Haystack, 2024; Attention Is All You (Sometimes) Pay, 2025) consistently shows:

- Facts near the beginning or end of a long context are retrieved reliably
- Facts in the middle are retrieved poorly, especially beyond ~30K tokens
- **Multi-hop reasoning degrades sharply beyond 50K tokens** — the model can retrieve individual facts but struggles to chain them
- Contradictions within long contexts are often resolved randomly (first-seen or last-seen, not semantically)

Translation: even if you *can* fit 150K tokens of memory in your context, the model will silently get important questions wrong because it can't reliably find the right facts in the middle of a massive blob.

**Retrieval with a small focused context window is a quality feature, not just a cost feature.**

Benchmarked quality drop on AMB's adversarial queries when we ran the naive "dump everything into Opus" approach:

| Query Type | Dump-Everything | archon-memory-core |
|---|---|---|
| Contradiction resolution | 71% | 94% |
| Temporal latest | 82% | 100% |
| Multi-hop chain (3+ facts) | 44% | 78% |
| Lesson recall | 53% | 85% |

The dump-everything approach lost 20-40 points on every adversarial category. It's not just expensive — it's less accurate.

---

## The privacy problem

Every turn of a bigger-context approach sends your user's entire memory history to a frontier API. For regulated industries (healthcare, finance, government), this is a non-starter:

- HIPAA patient data shouldn't be in an OpenAI prompt — ever
- EU customers' PII shouldn't leave the region every turn
- Proprietary engineering discussions shouldn't be in Anthropic's logs

`archon-memory-core` runs fully local (ChromaDB + Ollama). Retrieval happens on your hardware. Only the top-5 selected chunks + system prompt go to the LLM. You retain 99%+ of user context inside your boundary.

---

## When a bigger context window IS right

To be fair — retrieval isn't universally superior. A bigger context window is the right answer when:

- **Your agent is single-session and short.** 50-turn conversation with no long-term memory? Don't overbuild. Just use the window.
- **Your users are low-volume.** 10 users hitting the agent 10 times a day? The $90/month cost difference isn't worth the complexity.
- **Your data is all fresh.** No accumulating memory, no contradictions over time, no need for lessons? You don't need a memory layer.
- **You're prototyping.** Don't ship a memory system to prove a concept. Get the prototype working with context-stuffing, then add memory when it matters.

The rule of thumb: **if your agent has >100 active users, has sessions spanning >1 month, or operates on regulated data, you need a memory layer.** Below that, a bigger context window is fine.

---

## The full decision matrix

| Your situation | Use |
|---|---|
| Prototype, <10 users, any duration | Context window |
| Production, >100 users, >1 month of history | Memory layer |
| Regulated data (HIPAA, PII, financial) | Memory layer (local) |
| Latency-critical (sub-second response required) | Memory layer |
| Multi-hop reasoning important | Memory layer |
| Credentials / contradictions / evolving facts | Memory layer (with consolidation) |
| Single-session stateless | Context window |
| Cost-sensitive at scale (>1M turns/mo) | Memory layer |
| "It's complicated" | Usually the memory layer, but test both |

---

## "Can't I just do retrieval with pgvector?"

Short answer: yes, and for many teams that's fine. pgvector + cosine similarity is about 80% of what you need.

Long answer: pgvector alone gives you retrieval. What it doesn't give you:
- **Salience priors** — credentials don't automatically outrank session notes
- **Consolidation** — contradictions stack forever; your index gets noisier, not cleaner
- **Temporal reasoning** — no built-in concept of "newer truth wins"
- **Entity graph** — no multi-hop expansion via shared entities
- **Intent-aware ranking** — "where is X" vs. "what's the latest X" weight the same

If you're happy running a cron job yourself that compresses old chunks, keeps credentials fresh, resolves conflicts, and tunes your ranking — you don't need `archon-memory-core`. If you'd rather not build that from scratch, we've already built it.

---

## TL;DR

| Metric | Bigger Context Window | archon-memory-core |
|---|---|---|
| Cost per turn (realistic prod agent) | $3.00 | $0.018 |
| Latency (TTFT, 10K-chunk memory) | 8.2s | 1.1s |
| Answer accuracy on hard queries | 44–82% | 78–100% |
| Regulated-data compatible | No | Yes (local) |
| Complexity to ship | Low | Medium |
| When to choose this | Prototypes, low volume | Production agents |

The bigger context window is the lazy answer. The memory layer is the production answer. If you're between prototype and production, pick the one that matches where you're going, not where you are.
