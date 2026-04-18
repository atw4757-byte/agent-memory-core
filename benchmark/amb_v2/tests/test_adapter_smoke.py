"""T-16..T-20 — Adapter smoke tests, parametrized.

Each adapter must:
  1. instantiate in both modes
  2. ingest the mini scenario without throwing
  3. respond to consolidate() in both modes (no-op in stock)
  4. return a non-None string from query()
  5. expose valid metadata
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark.amb_v2.adapters.base import validate_metadata
from benchmark.amb_v2.adapters.naive import NaiveAppendOnlyAdapter
from benchmark.amb_v2.scenarios import load_public_scenarios
from benchmark.amb_v2.simulator import simulate

FIXTURES = Path(__file__).parent / "fixtures"


def _adapter_factories():
    factories = [("naive", NaiveAppendOnlyAdapter)]
    try:
        from benchmark.amb_v2.adapters import agent_memory_core as amc_mod
        if amc_mod._AVAILABLE:
            factories.append(("agent-memory-core", amc_mod.AgentMemoryCoreAdapter))
    except ImportError:
        pass
    try:
        from benchmark.amb_v2.adapters import langchain_adapter as lc_mod
        if lc_mod._AVAILABLE:
            factories.append(("langchain", lc_mod.LangChainAdapter))
    except ImportError:
        pass
    return factories


ADAPTERS = _adapter_factories()


@pytest.fixture
def mini_bundles(tmp_path: Path):
    shutil.copy(FIXTURES / "mini_scenario.json", tmp_path / "mini.json")
    return load_public_scenarios(tmp_path)


@pytest.mark.parametrize("name,factory", ADAPTERS)
@pytest.mark.parametrize("mode", ["stock", "tuned"])
def test_adapter_smoke(name, factory, mode, mini_bundles):
    adapter = factory(mode=mode)
    validate_metadata(adapter.metadata)
    assert adapter.metadata["mode"] == mode

    for day, chunks in simulate(mini_bundles, seed=42, days=10, noise_rate=0.30):
        adapter.ingest(day, chunks)
        adapter.consolidate(day)

    answer = adapter.query("What is the user's home address?", "mini")
    assert isinstance(answer, str)
