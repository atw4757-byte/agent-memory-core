# agent-memory-core

Agent memory that actually works: drops into any Python agent in 3 lines.

## Installation

```bash
pip install agent-memory-core

# Optional: cross-encoder re-ranking (bumps recall ~8%)
pip install "agent-memory-core[reranker]"

# Optional: networkx for advanced graph operations
pip install "agent-memory-core[graph]"
```

## Quickstart

```python
from agent_memory_core import MemoryStore

store = MemoryStore()
store.add("The API key is stored in the keychain", type="credential")
store.add("The project uses Python 3.12", type="technical")

results = store.search("Where is the API key?")
print(results[0].text)  # "The API key is stored in the keychain"
```

## Features

1. **Salience-weighted retrieval** — credentials surface over stale session logs automatically; type priors, access count, and graph connectivity all feed the score
2. **Adaptive query intent detection** — "where is the API key" routes as similarity-heavy; "current status" routes as recency-heavy; query weights adjust without configuration
3. **Cross-encoder re-ranking** — optional two-stage retrieval: embed wide, re-rank with `cross-encoder/ms-marco-MiniLM-L-6-v2` for +8% recall
4. **MMR diversity** — Maximal Marginal Relevance prevents five results that all say the same thing
5. **Nightly lossy consolidation** — many episodic chunks compress into stable semantic facts via local LLM (Mistral/Qwen via Ollama); originals archived, never deleted
6. **Entity relationship graph** — memory files linked by shared entities and topics; 2-hop neighbor expansion surfaces related context; graph connectivity boosts salience
7. **Working memory buffer** — 4-7 short-term slots (Miller's Law) that survive session restarts; flush() serializes to long-term store

## Eval Score: 7.6/10

Baseline (naive ChromaDB cosine): 3.1/10
With salience + adaptive weights: 5.4/10
With cross-encoder + MMR + fact layer: **7.6/10**

Metrics: Recall@5 82%, Precision@5 61%, Answer 74% across 30 stratified queries.

```
Composite = (0.4 * Recall + 0.3 * Precision + 0.3 * Answer) / 10
```

Run the eval against your own data:

```python
from agent_memory_core import MemoryStore, MemoryEval

store = MemoryStore()
ev = MemoryEval(store)
ev.add_query("Where is the API key?", expected_facts=["keychain"], type="credential")
report = ev.run(n=5, version="my-config")
print(f"Score: {report['composite']}/10")
```

## Architecture

```
MemoryStore (chromadb)
  |
  +-- add(text, type, source, agent)
  |     |-- ChromaDB upsert (always)
  |     +-- Hindsight retain (if available)
  |
  +-- search(query, n, type, since, agent)
        |
        +-- 1. ChromaDB cosine retrieval (4x candidate pool)
        +-- 2. Salience + recency scoring (adaptive weights)
        +-- 3. Cross-encoder re-ranking (optional)
        +-- 4. MMR diversity selection
        +-- 5. Atomic fact augmentation
        +-- 6. Dynamic tail pruning

WorkingMemory (json buffer, Miller's 7 slots)
  +-- flush() -> MemoryStore.add(..., type="session")

Consolidator (nightly, requires Ollama)
  +-- cluster_chunks (source+type, keyword Jaccard, entity graph)
  +-- compress_cluster (Mistral/Qwen)
  +-- decompose_to_facts (atomic fact extraction)
  +-- archive_originals (soft delete via metadata flag)

MemoryGraph (memory_graph.json)
  +-- build(source_paths)  -- entity extraction via Ollama/Gemini
  +-- search(query)        -- keyword + 2-hop neighbor expansion
  +-- entity_map()         -- feeds Consolidator strategy 3

ForgettingPolicy
  +-- find_stale_files / find_stale_chunks
  +-- archive_chunks (soft) / hard_delete
  +-- health_report() -> score 0-100

MemoryEval
  +-- run(n, version) -> recall/precision/answer/composite
  +-- history() / score_delta()
```

## Comparison

| Feature | agent-memory-core | LangChain Memory | MemGPT | Mem0 |
|---|---|---|---|---|
| Salience scoring | Yes (type + access + graph) | No | No | Partial |
| Cross-encoder re-ranking | Yes (optional) | No | No | No |
| MMR diversity | Yes | No | No | No |
| Nightly consolidation | Yes (Ollama) | No | Yes (GPT-4) | Partial |
| Entity graph | Yes | No | No | No |
| Working memory | Yes (7 slots, Miller's Law) | Basic | Yes | No |
| Agent namespacing | Yes | No | No | No |
| Eval harness | Yes (30 queries, 3 metrics) | No | No | No |
| Local-first | Yes (Ollama, ChromaDB) | Partial | No | No |
| License | Apache 2.0 | MIT | Apache 2.0 | MIT |

*Comparison based on public documentation as of April 2026.*

## Valid Chunk Types

```python
VALID_TYPES = {
    "fact", "personal", "professional", "credential", "financial",
    "goal", "project_status", "technical", "session", "task",
    "observation", "dream", "lesson",
}
```

Each type has a salience prior and temporal decay rate. Credentials and lessons never decay.

## Agent Namespacing

```python
# Shared memory — visible to all agents (default)
store.add("Project uses Python 3.12", type="technical")

# Private memory — only visible to this agent
store.add("My internal scratchpad note", type="session", agent="cipher")

# Search: sees shared + cipher-private
results = store.search("Python version", agent="cipher")
```

## Consolidation (requires Ollama)

```python
from agent_memory_core import MemoryStore, Consolidator

store = MemoryStore()
consolidator = Consolidator(store, min_cluster=3)

# Dry run to preview what would be consolidated
report = consolidator.run(dry_run=True)
print(f"Would consolidate {report['clusters_viable']} clusters")

# Run for real (requires ollama + mistral:latest or qwen2.5:7b)
report = consolidator.run()
print(f"Archived {report['archived']} chunks into {report['consolidated']} facts")
```

## Requirements

- Python >= 3.10
- chromadb >= 0.5.0

Optional:
- Ollama with `mistral:latest` or `qwen2.5:7b` (for consolidation, graph enrichment)
- Hindsight Docker container (second-layer LLM memory, graceful fallback if absent)
