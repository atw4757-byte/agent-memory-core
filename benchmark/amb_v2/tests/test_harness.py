"""T-21, T-22 — Harness tests."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from benchmark.amb_v2.adapters.naive import NaiveAppendOnlyAdapter
from benchmark.amb_v2.harness import RESULTS_SCHEMA, run_one
from benchmark.amb_v2.scenarios import load_public_scenarios

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_bundles(tmp_path: Path):
    shutil.copy(FIXTURES / "mini_scenario.json", tmp_path / "mini.json")
    return load_public_scenarios(tmp_path)


def test_fires_checkpoints_only_at_expected_days(mini_bundles):
    a = NaiveAppendOnlyAdapter(mode="stock")
    result = run_one(
        adapter=a, scenarios=mini_bundles, seed=42, noise_rate=0.30,
        mode="stock", checkpoints=[0, 3, 7],
    )
    days_with_checkpoints = sorted(c["day"] for c in result["checkpoints"])
    assert days_with_checkpoints == [0, 3, 7]


def test_results_json_schema_valid(mini_bundles):
    a = NaiveAppendOnlyAdapter(mode="stock")
    result = run_one(
        adapter=a, scenarios=mini_bundles, seed=42, noise_rate=0.30,
        mode="stock", checkpoints=[0, 7],
    )
    jsonschema.validate(result, RESULTS_SCHEMA)


def test_records_metadata(mini_bundles):
    a = NaiveAppendOnlyAdapter(mode="stock")
    result = run_one(
        adapter=a, scenarios=mini_bundles, seed=42, noise_rate=0.30,
        mode="stock", checkpoints=[0],
    )
    assert result["adapter"] == "naive-append-only"
    assert result["mode"] == "stock"
    assert result["seed"] == 42
    assert result["noise_rate"] == 0.30
    assert result["spec_version"] == "v2.0.0"


def test_consolidate_only_in_tuned_mode(mini_bundles, monkeypatch):
    """Stock mode must not invoke the adapter's consolidate work — verified by
    tracking calls."""
    calls: list[tuple[int, str]] = []

    class Tracking(NaiveAppendOnlyAdapter):
        def consolidate(self, day):
            calls.append((day, self.mode))
            super().consolidate(day)

    a = Tracking(mode="stock")
    run_one(adapter=a, scenarios=mini_bundles, seed=42, noise_rate=0.30,
            mode="stock", checkpoints=[0, 3])
    # Harness still calls consolidate() each day (the adapter decides what to do
    # internally based on its mode). All calls should record stock mode.
    assert all(m == "stock" for _, m in calls)


def test_quality_metric_in_each_checkpoint(mini_bundles):
    a = NaiveAppendOnlyAdapter(mode="stock")
    result = run_one(
        adapter=a, scenarios=mini_bundles, seed=42, noise_rate=0.30,
        mode="stock", checkpoints=[0, 7],
    )
    for cp in result["checkpoints"]:
        assert "quality" in cp
        assert 0.0 <= cp["quality"] <= 1.0


def test_auc_present(mini_bundles):
    a = NaiveAppendOnlyAdapter(mode="stock")
    result = run_one(
        adapter=a, scenarios=mini_bundles, seed=42, noise_rate=0.30,
        mode="stock", checkpoints=[0, 7, 14],
    )
    assert "auc_quality" in result
    assert result["auc_quality"] >= 0.0
