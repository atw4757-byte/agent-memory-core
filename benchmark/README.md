# Agentic Memory Benchmark (AMB)

A rigorous, open benchmark for evaluating long-horizon memory systems. 10 real-world scenarios, 200 queries, adversarial traps designed to expose exactly where naive systems fail.

> **AMB v2 is alpha-released** (2026-04-18). v2 is pre-registered, stock/tuned-split, and sensitivity-swept. See [`amb_v2/README.md`](amb_v2/README.md) and [`amb_v2/PREREGISTERED.md`](amb_v2/PREREGISTERED.md). v1 (below) remains the legacy scenario corpus.

## The Problem With Existing Evals

Most memory evals test retrieval of static facts. Real agents deal with facts that *change*. A user's address updates. A stakeholder leaves a company. A preference reverses. A diet changes. A bug gets fixed.

A system that returns the first value it ever saw for a question about a user's coffee order — after that order changed three sessions ago — is useless in production. Existing benchmarks don't catch this.

AMB does.

## What It Tests

| Reasoning Type | Description | Why It's Hard |
|---|---|---|
| `contradiction_resolution` | Fact changed — return the current value | Naive systems return the most-seen or first-seen value |
| `temporal_latest` | Return the most recent value from a series of updates | Easy to get wrong when values appear multiple times |
| `temporal_historical` | Retrieve a *past* value without confusing it for current | Inversion of the above — must know *which* time is being asked about |
| `multi_hop` | Chain 2-3 facts to answer a question | Each hop can introduce a stale fact |
| `negative_with_history` | Something existed and was cancelled — return the cancelled state | Systems that retrieved the fact at any point will say it still exists |
| `lesson_lookup` | Recall a rule learned from a past mistake | Lessons often appear once and get buried |
| `aggregation` | Collect related facts across sessions into a coherent answer | Requires spanning the full context, not just one chunk |
| `causal_chain` | Explain *why* something happened | Requires understanding event sequence, not just state |
| `simple_lookup` | Retrieve a stable fact | Baseline; should be near 100% for any functional system |
| `rule_application` | Apply a learned rule to a new situation | Tests generalisation, not just recall |

## Scenarios

Each scenario simulates 5-8 conversation sessions across months of interaction.

| # | Scenario | Domain | Key Challenges |
|---|---|---|---|
| 01 | Personal Assistant | Life admin | Address change, coffee order update, cancelled meetings, rule aggregation |
| 02 | Executive Chief of Staff | B2B/strategy | Board member change, ARR updates, strategic pivot after personnel loss |
| 03 | Customer Support | SaaS support | Admin change, bug resolved vs. recurring, outstanding engineering tickets |
| 04 | Health Coach | Wellness | Diet switch (lacto-ovo → vegan), supplement dose trajectory, goal evolution |
| 05 | Software PM | Product | Architecture owner change, roadmap additions/removals, shipped vs. planned |
| 06 | Sales CRM | Enterprise sales | Economic buyer departure mid-deal, blocker-to-supporter flip, multi-version pricing |
| 07 | Travel Concierge | Travel | Airport split by route, hotel exception per city, constraint scope expansion |
| 08 | Tutor | Education | Misconception correction, teacher change mid-semester, grade trajectory |
| 09 | Household | Home management | Appliance replacement, recurring service issues, maintenance status |
| 10 | Research Assistant | Academia | Architecture evolution (3 changes), normalization bug → contribution, dataset addition |

## Query Distribution

Each scenario has **20 queries** (200 total) with the following distribution:

- Easy: ~6 queries/scenario
- Medium: ~9 queries/scenario
- Hard: ~5 queries/scenario

~40% of queries include a **trap** — a correct-sounding wrong answer that a naive system will return.

## Running the Benchmark

### Against agent-memory-core (default)

```bash
# Install the package
pip install -e .

# Run all scenarios
python benchmark/run_benchmark.py

# Single scenario
python benchmark/run_benchmark.py --scenario 01_personal_assistant

# Quiet (summary only)
python benchmark/run_benchmark.py --quiet
```

### Against a custom system

Implement the `MemorySystemAdapter` protocol:

```python
class MyMemoryAdapter:
    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        # Add conversation turn to your memory system
        ...

    def query(self, question: str) -> str:
        # Return a string answer
        ...

    def reset(self) -> None:
        # Clear all state between scenarios
        ...
```

Then run:

```bash
python benchmark/run_benchmark.py --adapter mymodule.MyMemoryAdapter
```

### Options

```
--adapter   Dotted path to custom adapter class
--scenario  Run only scenarios matching this substring
--output    Custom path for the JSON report
--k         Top-k for recall/precision (default 5)
--quiet     Suppress per-query output
```

## Metrics

All metrics are in [0.0, 1.0]. Composite score is on a 0–10 scale.

### `recall_at_k`
At least one of the top-k retrieved chunks contains any expected fact. Binary (1.0 or 0.0). The minimum bar — if this fails, the system didn't retrieve relevant information at all.

### `precision_at_k`
Fraction of top-k results that contain at least one expected fact. Measures retrieval signal density.

### `answer_completeness`
Fraction of expected facts present in the generated answer. Measures whether the system's response covers everything needed, not just part of it.

### `temporal_accuracy`
Penalises confusing current state with historical state. Returns 1.0 if the current value is in the answer, 0.0 if only the historical (stale) value is returned.

### `contradiction_resolution_rate`
Fraction of fact-contradiction pairs where the system correctly returns the newer value. Directly measures the core failure mode.

### Composite Score
```
composite = (0.25 × recall + 0.20 × precision + 0.25 × answer_completeness
           + 0.15 × temporal_accuracy + 0.15 × contradiction_rate) × 10
```

## Baseline Results

| System | Composite | Recall@5 | Precision@5 | Answer | Temporal | Contradiction |
|---|---|---|---|---|---|---|
| Naive ChromaDB (cosine only) | 3.1/10 | 68% | 42% | 51% | 34% | 29% |
| agent-memory-core v0.1 | 7.7/10 | 82% | 61% | 74% | 71% | 68% |

*Temporal accuracy and contradiction resolution are the primary differentiators between naive and production-quality memory systems.*

## Output Format

Results are saved as JSON to `benchmark/results/RUNID.json`. Structure:

```json
{
  "run_id": "20260409_143022",
  "adapter": "AgentMemoryCoreAdapter",
  "overall": {
    "composite": 7.7,
    "recall": 0.82,
    "precision": 0.61,
    "answer_completeness": 0.74,
    "temporal_accuracy": 0.71,
    "by_difficulty": {"easy": 0.89, "medium": 0.75, "hard": 0.61},
    "by_reasoning_type": {...}
  },
  "trap_avg": 6.2,
  "nontrap_avg": 8.4,
  "scenario_results": [...]
}
```

## Scenario JSON Format

Each scenario file follows this schema:

```json
{
  "scenario_id": "01_personal_assistant",
  "name": "...",
  "description": "...",
  "sessions": [
    {
      "session_id": 1,
      "label": "...",
      "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
  ],
  "queries": [
    {
      "query_id": 1,
      "query": "...",
      "expected_answer": "...",
      "reasoning_type": "contradiction_resolution",
      "difficulty": "hard",
      "trap": "Description of the wrong answer a naive system would give.",
      "relevant_sessions": [1, 3]
    }
  ]
}
```

## License

Apache 2.0. See root LICENSE file.
