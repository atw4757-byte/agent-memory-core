"""T-27 — Golden regression on the mini scenario.

The first run (or after an intentional methodology change) is committed as
``fixtures/golden_mini.json``. Every subsequent run compares against it.
Regenerate with::

    AMB_V2_REGEN_GOLDEN=1 pytest benchmark/amb_v2/tests/test_regression.py
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from benchmark.amb_v2.adapters.naive import NaiveAppendOnlyAdapter
from benchmark.amb_v2.harness import run_one
from benchmark.amb_v2.scenarios import load_public_scenarios

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "_golden" / "golden_mini.json"


def _run_mini(tmp_path: Path) -> dict:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    shutil.copy(FIXTURES / "mini_scenario.json", scenarios_dir / "mini.json")
    bundles = load_public_scenarios(scenarios_dir)
    adapter = NaiveAppendOnlyAdapter(mode="stock")
    return run_one(
        adapter=adapter, scenarios=bundles, seed=42, noise_rate=0.30,
        mode="stock", checkpoints=[0, 3, 7],
    )


def test_mini_scenario_golden(tmp_path: Path):
    result = _run_mini(tmp_path)

    if os.environ.get("AMB_V2_REGEN_GOLDEN") == "1":
        # Drop volatile fields before freezing.
        frozen = {k: v for k, v in result.items() if k != "completed_at"}
        GOLDEN.write_text(json.dumps(frozen, indent=2, sort_keys=True))
        pytest.skip("Regenerated golden; re-run without AMB_V2_REGEN_GOLDEN")

    assert GOLDEN.exists(), f"missing {GOLDEN} — run with AMB_V2_REGEN_GOLDEN=1 first"
    expected = json.loads(GOLDEN.read_text())
    actual = {k: v for k, v in result.items() if k != "completed_at"}

    # Compare structural fields exactly
    for key in ("adapter", "mode", "seed", "noise_rate", "spec_version",
                "checkpoints", "auc_quality"):
        assert actual[key] == expected[key], f"drift in {key}: {actual[key]!r} != {expected[key]!r}"
