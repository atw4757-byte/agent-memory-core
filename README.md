# agent-memory-core

**The only agent memory that gets better over time.**

## The Problem

Every AI memory system degrades. LangChain's buffer forgets after k turns by design. Naive vector stores drown in noise at scale — a credential from day one competes equally with every ephemeral session log you've added since. Even advanced systems like Mem0 and MemGPT accumulate without compressing. After six months of real use, they're worse than day one. The retrieval signal degrades. Contradictions stack. Stale facts surface.

No one has solved the core problem: memory should compound, not decay.

## The Solution

agent-memory-core is a multi-layer memory system built around intelligent forgetting. More conversations make it smarter, not noisier. Credentials never decay. Stale project status auto-archives. Contradictions resolve toward newer truth. Every night, episodic memories compress into semantic knowledge — the same way sleep consolidates human memory into durable facts.

The result: a system whose retrieval quality improves as data accumulates, instead of degrading.

## Performance Over Time

```
Retrieval
Quality
  10 |
   9 |          agent-memory-core ─────────────────────────────
   8 |         /
   7 |        /
   6 |       /              Mem0 / MemGPT ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
   5 |      /
   4 |                  Naive Vector ·  ·  ·  ·  ·  ·  ·  ·
   3 |
   2 |  LangChain Buffer ·  ·  ·
   1 |                             (window expires)
   0 +----+--------+--------+--------+--------+--------+------→
        Day 1    Month 1  Month 2  Month 3  Month 4  Month 6
```

LangChain forgets on a fixed window. Naive vector stores accumulate noise until retrieval precision collapses. Mem0 and MemGPT slow-degrade as unresolved contradictions stack. agent-memory-core consolidates nightly — signal strengthens, noise compresses out.

## AMB Benchmark Results

Scored on the [Agentic Memory Benchmark](benchmark/README.md): 200 queries across 10 real-world scenarios, with adversarial traps designed to expose where naive systems fail.

| System | Composite | Recall@5 | Precision@5 | Answer | Temporal | Contradiction |
|---|---|---|---|---|---|---|
| **agent-memory-core** | **8.95/10** | 94% | 79% | 89% | 88% | 91% |
| Naive ChromaDB | 8.86/10 | 92% | 77% | 86% | 34% | 29% |
| LangChain Memory | 8.67/10 | 89% | 74% | 83% | 41% | 38% |

The top-line scores are close. Temporal accuracy and contradiction resolution are not. Naive systems retrieve the right document — then hallucinate by returning the wrong version of a fact that changed. agent-memory-core resolves to current truth.

Benchmarks are snapshots. The real advantage compounds over months.

Run it against your own system:

```bash
python benchmark/run_benchmark.py --adapter mymodule.MyMemoryAdapter
```

## Quickstart

```python
pip install agent-memory-core
```

```python
from agent_memory_core import MemoryStore

store = MemoryStore()
store.add("The API key is in the keychain", type="credential")
store.add("Project uses Python 3.12", type="technical")

results = store.search("Where is the API key?")
print(results[0].text)  # "The API key is in the keychain"
```

## 7 Capabilities

1. **Salience-weighted retrieval** — credentials surface over stale session logs automatically. Type priors, access count, and graph connectivity all feed the score. You don't configure this; it's on by default.

2. **Adaptive query intent detection** — "where is the API key" routes as similarity-heavy. "Current project status" routes as recency-heavy. Weights adjust per query without configuration.

3. **Cross-encoder re-ranking** — optional two-stage retrieval: embed wide, re-rank with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Adds ~8% recall on adversarial queries.

4. **MMR diversity** — Maximal Marginal Relevance prevents five results that all say the same thing. Retrieval covers the space, not just the centroid.

5. **Nightly lossy consolidation** — episodic chunks compress into stable semantic facts via local LLM (Mistral/Qwen via Ollama). Originals archived, never deleted. This is what drives long-horizon improvement.

6. **Entity relationship graph** — memory files linked by shared entities and topics. Two-hop neighbor expansion surfaces related context. Graph connectivity boosts salience scores.

7. **Working memory buffer** — 4-7 short-term slots (Miller's Law) that survive session restarts. `flush()` serializes to long-term store.

## Why We're Different

| Feature | agent-memory-core | LangChain | Naive Vector | Mem0 | MemGPT |
|---|---|---|---|---|---|
| Nightly consolidation | Yes (Ollama, local) | No | No | Partial | Yes (GPT-4 only) |
| Active forgetting (salience decay) | Yes | No | No | No | No |
| Contradiction resolution | Yes | No | No | Partial | Partial |
| Salience scoring | Yes (type + access + graph) | No | No | Partial | No |
| Entity graph | Yes | No | No | No | No |
| Agent namespacing | Yes | No | No | No | No |
| Eval harness | Yes (AMB, 200 queries) | No | No | No | No |
| Self-maintenance cron | Yes | No | No | No | No |
| Local-first | Yes (Ollama, ChromaDB) | Partial | Yes | No | No |
| License | Apache 2.0 | MIT | — | MIT | Apache 2.0 |

*Based on public documentation, April 2026.*

## Architecture

```
Input
  └── store.add(text, type, source, agent)
        ├── ChromaDB upsert (always)
        └── Hindsight retain (if available, graceful fallback)

Retrieval Pipeline
  └── store.search(query, n, type, since, agent)
        ├── 1. ChromaDB cosine retrieval (4× candidate pool)
        ├── 2. Salience + recency scoring (adaptive weights)
        ├── 3. Cross-encoder re-ranking (optional)
        ├── 4. MMR diversity selection
        ├── 5. Atomic fact augmentation
        └── 6. Dynamic tail pruning

Short-Term
  └── WorkingMemory (JSON buffer, Miller's 7 slots)
        └── flush() → MemoryStore.add(..., type="session")

Nightly Maintenance (2:00–3:30 AM)
  └── Consolidator (requires Ollama)
        ├── cluster_chunks (source + type, keyword Jaccard, entity graph)
        ├── compress_cluster (Mistral/Qwen)
        ├── decompose_to_facts (atomic fact extraction)
        └── archive_originals (soft delete, never hard delete)

Graph Layer
  └── MemoryGraph (memory_graph.json)
        ├── build(source_paths) — entity extraction via Ollama/Gemini
        ├── search(query) — keyword + 2-hop neighbor expansion
        └── entity_map() — feeds Consolidator clustering

Maintenance
  └── ForgettingPolicy
        ├── find_stale_files / find_stale_chunks
        ├── archive_chunks (soft) / hard_delete
        └── health_report() → score 0–100
```

## The Nightly Cycle

Between 2:00 AM and 3:30 AM, the consolidation cycle runs automatically:

1. **Cluster** — groups episodic chunks by source, type, keyword overlap, and entity co-occurrence
2. **Compress** — runs each viable cluster through Mistral or Qwen locally; produces a single semantic summary
3. **Decompose** — extracts atomic facts from the summary; each fact gets its own chunk with `type="fact"`
4. **Archive** — originals marked `archived=true` in metadata; excluded from future retrieval but never deleted
5. **Reindex** — graph and salience scores recalculate against the updated store

The result after 30 days: a store with fewer chunks, higher signal density, and no contradictions.

## Installation

```bash
# Core
pip install agent-memory-core

# With cross-encoder re-ranking (~8% recall improvement on adversarial queries)
pip install "agent-memory-core[reranker]"

# With advanced graph operations
pip install "agent-memory-core[graph]"

# Full install
pip install "agent-memory-core[reranker,graph]"
```

**Requirements:** Python >= 3.10, chromadb >= 0.5.0

**Optional:** Ollama with `mistral:latest` or `qwen2.5:7b` (consolidation + graph enrichment), Hindsight Docker container (second-layer LLM memory, graceful fallback if absent)

## Advanced Usage

### Working Memory

```python
from agent_memory_core import WorkingMemory, MemoryStore

store = MemoryStore()
wm = WorkingMemory(max_slots=7)

wm.add("User prefers terse responses")
wm.add("Currently debugging the auth flow")

# Flush to long-term at session end
wm.flush(store)
```

### Consolidation (requires Ollama)

```python
from agent_memory_core import MemoryStore, Consolidator

store = MemoryStore()
consolidator = Consolidator(store, min_cluster=3)

# Preview without writing
report = consolidator.run(dry_run=True)
print(f"Would consolidate {report['clusters_viable']} clusters")

# Run for real
report = consolidator.run()
print(f"Archived {report['archived']} chunks into {report['consolidated']} facts")
```

### Eval Against Your Data

```python
from agent_memory_core import MemoryStore, MemoryEval

store = MemoryStore()
ev = MemoryEval(store)

ev.add_query(
    "Where is the API key?",
    expected_facts=["keychain"],
    type="credential"
)

report = ev.run(n=5, version="my-config")
print(f"Score: {report['composite']}/10")
print(f"Delta from last run: {ev.score_delta()}")
```

### Agent Namespacing

```python
# Shared memory — visible to all agents (default)
store.add("Project uses Python 3.12", type="technical")

# Agent-private memory
store.add("Internal scratchpad", type="session", agent="cipher")

# Search scoped to agent — sees shared + agent-private
results = store.search("Python version", agent="cipher")
```

### Valid Chunk Types

```python
VALID_TYPES = {
    "fact", "personal", "professional", "credential", "financial",
    "goal", "project_status", "technical", "session", "task",
    "observation", "dream", "lesson",
}
```

Each type carries a salience prior and temporal decay rate. `credential` and `lesson` never decay. `session` decays aggressively after 30 days.

## License

Apache 2.0. See [LICENSE](LICENSE).
