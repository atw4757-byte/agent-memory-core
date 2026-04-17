# Roadmap

Public roadmap for `agent-memory-core`. Updated monthly. The OSS library is the core; everything else is scaffolding around it.

**Last updated:** 2026-04-17

---

## Now (Q2 2026)

### OSS / Core
- **LangChain adapter** — `agent_memory_core.integrations.langchain` → ships with v0.2
- **LlamaIndex adapter** — `agent_memory_core.integrations.llamaindex` → ships with v0.2
- **Async-first API** — `AsyncMemoryStore` with full parity → v0.2
- **Longitudinal benchmark (AMB v2)** — 90-day simulated decay across 10K-chunk ingest → alpha May, GA June
- **Demo video** — 60-second failure → replay → fix walkthrough
- **Public AMB leaderboard** — `amb.divergencerouter.com` → launches 2026-05-15

### Content / Distribution
- Blog post #1: *"Why your agent's memory is failing"*
- Blog post #2: *"We ran 100K turns through five memory systems. Here's what broke."*
- Blog post #3: *"The $47,000 context window — why bigger isn't smarter"*
- arXiv preprint: *AMB: An Agentic Memory Benchmark*
- Conference CFPs: NeurIPS workshops, AI Engineer Summit, LangChain Interrupt

---

## Next (Q3 2026)

### Paid Tier Launch (gated on OSS adoption — 1K stars + 50 prod users)
- **Solo Dev** ($19/mo) — Memory health dashboard, 5 eval runs/mo, email alerts
- **Team** ($99/mo) — Replay debugger, shared workspaces (3 seats), 50 eval runs/mo, webhook alerts
- **Business** ($499/mo) — Hosted consolidation, retention policies, SOC2 Type I, SLA

See [PRICING.md](PRICING.md) for the full tier structure. [ENTERPRISE.md](ENTERPRISE.md) for private deploy.

### OSS / Core
- **Hybrid mode consolidation worker** — Docker image that runs locally, pulls jobs from our cloud queue
- **WebAssembly embedding model** — run cross-encoder re-ranking in-browser for preview/debug tooling
- **Stream-based ingest** — incremental ingest for very long agent sessions
- **Entity graph v2** — multi-hop + temporal edges

---

## Later (Q4 2026)

- **Enterprise tier GA** — VPC/private deploy, SSO/SAML, DPA, 24/7 support (from $3K/mo)
- **Multilingual AMB** — v3 of benchmark with Spanish, French, German, Japanese, Mandarin scenarios
- **AMB hidden challenge set** — protect against scenario-specific tuning
- **Consolidation fine-tunes** — ship tuned Mistral/Qwen LoRAs specifically for consolidation quality
- **Replay debugger — collaborative** — multi-user debugging sessions on shared recall events

---

## Researching (no committed date)

- **Cross-agent memory sharing protocols** — protocol spec for agents from different vendors to share memory safely
- **Memory diffing + version control** — "git for memory"
- **Federated learning on aggregated benchmark data** — the AMB flywheel as published research
- **Physical embodiment** — integration with robotics memory frameworks (e.g., MRS on ROS 2)
- **Byzantine-tolerant memory replication** — for high-stakes multi-agent systems

---

## Recently Shipped

### v0.1.2 — 2026-04-15
- Adaptive retrieval modes (short vs. long scenario detection)
- Fix for short-scenario regression
- AMB v1 full results published

### v0.1.1 — 2026-04-14
- PyPI classifiers + CI workflow
- CONTRIBUTING.md
- 13 GitHub topics for discoverability

### v0.1.0 — 2026-04-12
- Initial public release
- All 13 chunk types with salience priors
- 6-stage retrieval pipeline
- Nightly consolidation (Ollama)
- Entity graph + 2-hop expansion
- Working memory buffer
- AMB eval harness (200 queries, 10 scenarios)

---

## How Priorities Are Set

1. **OSS adoption comes first.** No paid feature ships before the OSS has a healthy community — 1K+ stars, 50+ production users, active contributor base.
2. **Eval-driven everything.** Every retrieval-algorithm change must measure before/after on AMB. If a PR doesn't move a score, it doesn't merge.
3. **Customer-reported bugs beat roadmap items.** If a paid customer reports a P1, the roadmap slips.
4. **Benchmark neutrality.** AMB remains open, auditable, and reproducible. Changes go through 60-day grace periods.

---

## How to Influence the Roadmap

- **OSS users:** [open an issue](https://github.com/atw4757-byte/agent-memory-core/issues/new) or upvote existing ones. Top-upvoted items get prioritized.
- **Paid customers:** submit feature requests to `support@divergencerouter.com`. Enterprise customers get quarterly roadmap reviews with their TAM.
- **Researchers:** contribute benchmark scenarios or reasoning types. Accepted scenarios are credited in the AMB paper.
