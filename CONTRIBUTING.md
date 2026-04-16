# Contributing to agent-memory-core

## Development install

```bash
git clone https://github.com/Divergence-Router/agent-memory-core.git
cd agent-memory-core
pip install -e ".[reranker,graph,dev]"
```

## Running the tests

```bash
pytest tests/ -v
```

All tests are offline — no API keys or running services required.

## Running the benchmark

The Agentic Memory Benchmark (AMB) covers 10 scenarios and 200 queries.

```bash
# Run against agent-memory-core (default adapter)
python benchmark/run_benchmark.py

# Run a single scenario
python benchmark/run_benchmark.py --scenario 01_personal_assistant

# Run against a different adapter
python benchmark/run_benchmark.py --adapter benchmark.adapters.naive_vector.NaiveVectorAdapter
```

Results are written to `benchmark/results/`.

## Adding a new adapter

1. Create a new file in `benchmark/adapters/`, e.g. `my_adapter.py`.
2. Implement the three required methods:

```python
class MyAdapter:
    def ingest(self, session: list[dict]) -> None:
        """Store all turns from a session."""
        ...

    def query(self, q: str, n: int = 5) -> list[str]:
        """Return up to n text chunks relevant to the query."""
        ...

    def reset(self) -> None:
        """Clear all stored state between scenarios."""
        ...
```

3. Pass it to the benchmark runner with `--adapter my_adapter.MyAdapter`.
4. Open a PR — include benchmark results in the description.

## Code style

- No formatter enforced yet; match the style of the file you're editing.
- Type hints preferred. Docstrings on public methods.
- New features need at least one test in `tests/test_basic.py`.
