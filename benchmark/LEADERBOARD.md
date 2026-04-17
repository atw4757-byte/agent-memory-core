# AMB Leaderboard

**Submit your agent memory system. Get scored. Get listed.**

The Agentic Memory Benchmark (AMB) is becoming an open standard for evaluating long-horizon agent memory. This leaderboard is how we keep it honest — every system, scored on the same 200 queries, with the same harness, on the same hardware.

The public leaderboard lives at [**amb.divergencerouter.com**](https://divergencerouter.com/amb) (launches 2026-05-15).

---

## Why Submit

1. **Credibility.** Your users want to know your memory doesn't silently rot after month three. A published AMB score is the only widely-accepted way to prove it.
2. **Benchmark neutrality.** Submissions run on standardized hardware (ours or a neutral cloud spec) with a standardized harness. Your score is comparable to everyone else's.
3. **Differentiation where it matters.** Subscore breakdowns reveal where your system pulls ahead — temporal reasoning, contradiction resolution, credential durability. Marketing-ready.
4. **No cost.** AMB v1 submissions are free forever. AMB v2 (longitudinal) submissions are free for OSS systems and $250 per run for closed-source commercial systems (pays for compute).

---

## What Gets Scored

Every submission runs the full AMB v1 suite:

- 200 queries across 10 real-world scenarios
- 5 reasoning types (temporal, contradiction, multi-hop, lesson, aggregation)
- Metrics: Recall@5, Precision@5, Answer Accuracy, Temporal Correctness, Contradiction Resolution, Composite (weighted geometric mean)

See [README.md](README.md) for the methodology.

---

## How to Submit

### Option 1: Self-Hosted Python System

Your system must expose a Python adapter matching the `BenchmarkAdapter` protocol:

```python
from agent_memory_core.benchmark import BenchmarkAdapter

class MySystemAdapter(BenchmarkAdapter):
    def reset(self) -> None: ...
    def ingest(self, chunk: dict) -> None: ...
    def search(self, query: str, n: int = 5) -> list[dict]: ...
    def consolidate(self) -> None: ...  # called between sessions
```

Submit via GitHub PR:

1. Fork `atw4757-byte/agent-memory-core`
2. Add your adapter to `benchmark/adapters/{your_system}.py`
3. Run locally: `python benchmark/run_benchmark.py --adapter {your_system}`
4. Commit the result JSON: `benchmark/results/{your_system}-{YYYY-MM-DD}.json`
5. Open a PR. CI re-runs the benchmark on neutral hardware. If your submitted and CI-run scores match (within ±0.05 composite), we merge and list.

### Option 2: Hosted / SaaS System

For systems that don't expose a Python library (e.g., hosted memory APIs):

1. Email `benchmark@divergencerouter.com` with:
   - System name + website + OSS license (if applicable)
   - API documentation (auth, ingest endpoint, search endpoint)
   - Preferred auth method for our test run
2. We'll implement the adapter, run the benchmark on neutral hardware, publish the result, and invite you to verify.

### Option 3: Bring Your Own Harness

If you have a strong objection to our harness (e.g., you want to demonstrate tooling that runs outside Python), submit:

- A Docker image that runs AMB
- The same result JSON format
- A reproducibility script we can run independently

We'll validate reproducibility before listing.

---

## Result JSON Format

```json
{
  "system": "your-system-name",
  "version": "1.2.3",
  "date": "2026-05-20",
  "hardware": "neutral-us-east-m7i.xlarge",
  "composite": 8.74,
  "metrics": {
    "recall_at_5": 0.92,
    "precision_at_5": 0.88,
    "answer_accuracy": 0.61,
    "temporal_accuracy": 0.94,
    "contradiction_resolution": 0.85
  },
  "by_scenario": { "01_personal_assistant": {...}, "02_exec_cos": {...}, ... },
  "by_reasoning_type": { "contradiction_resolution": {...}, "temporal_latest": {...}, ... },
  "runtime_seconds": 842,
  "total_chunks_ingested": 14723,
  "submitter": "contact@your-org.com",
  "notes": "Optional — any relevant config, caveats, or version-specific context."
}
```

---

## Standardized Hardware

Neutral-run submissions use an **AWS EC2 m7i.xlarge** in us-east-1:
- 4 vCPU
- 16 GB RAM
- 250 GB gp3 SSD
- Ubuntu 24.04 LTS
- Python 3.12

For systems requiring GPU (e.g., local consolidation via Ollama), we run on **g5.xlarge** (1x A10G 24GB) with an hourly cost surcharge applied to the $250 commercial fee.

You can submit a "local hardware" result if your system is GPU-heavy and m7i.xlarge is infeasible — we'll list it with a hardware asterisk.

---

## Anti-Gaming

AMB v1's scenario data is published under [`benchmark/scenarios/`](scenarios). This is a *feature* — it forces systems to be evaluated on their architecture, not their ability to hide from a test set.

**AMB v2** (launching 2026-08) adds a **hidden challenge set** that is never published, only run by the benchmark authority. Commercial systems will be required to run against the hidden set to maintain a listing. This protects against training-data contamination and scenario-specific tuning.

We reserve the right to:
- **De-list** systems that can't reproduce their submitted score within ±0.05 composite
- **De-list** systems that demonstrate scenario-specific tuning by underperforming on the AMB v2 hidden set relative to v1
- **Publicly annotate** systems that refuse to run the hidden set

---

## Current Leaderboard

*(Preview — public launch 2026-05-15.)*

| Rank | System | Version | Composite | Answer | Temporal | Contradiction | Date |
|---|---|---|---|---|---|---|---|
| 🥇 | **agent-memory-core** | 0.1.2 | **9.01** | 0.69 | 1.00 | 0.94 | 2026-04-16 |
| 🥈 | Naive ChromaDB | 0.5.3 | 8.86 | 0.63 | 1.00 | 0.94 | 2026-04-16 |
| 🥉 | LangChain Window (k=10) | 0.3.14 | 8.67 | 0.65 | 1.00 | 0.92 | 2026-04-16 |
| — | *Oracle (full context)* | *ceiling* | *9.46* | *0.84* | *1.00* | *0.96* | *2026-04-16* |

**Invited to submit:** Mem0, MemGPT, Letta, Zep, Cognee, MemOS, MemSQL, pgvector pipeline reference, LangGraph Memory, CrewAI Memory.

If you maintain one of these, email `benchmark@divergencerouter.com` and we'll help you get scored.

---

## Integrity Commitment

- The benchmark authority (agent-memory-core maintainers) competes on the leaderboard like any other system.
- Our own scores are run on the same neutral hardware as everyone else's, with the same harness, by the same CI.
- If anyone beats us on the leaderboard, we congratulate them in a blog post. Competition is the point.
- We will never modify scenario data to favor our system. Changes to AMB (additions, corrections, versioning) are announced publicly and require a 60-day grace period before retroactive rescoring.
- All submissions, including ours, are auditable via the CI logs committed to this repo.

---

## Contact

- **Submissions, questions:** `benchmark@divergencerouter.com`
- **Press / partnership:** `press@divergencerouter.com`
- **Issues, corrections:** [open a GitHub issue](https://github.com/atw4757-byte/agent-memory-core/issues/new)
