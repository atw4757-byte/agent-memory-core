"""T-28 — End-to-end mini test: grid → schema → chart → smoke assertions."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from benchmark.amb_v2.chart import render_decay_curves, render_sensitivity_grid
from benchmark.amb_v2.harness import RESULTS_SCHEMA
from benchmark.amb_v2.run_all import GridSpec, run_grid


FIXTURES = Path(__file__).parent / "fixtures"


def test_full_pipeline_on_mini(tmp_path: Path):
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    shutil.copy(FIXTURES / "mini_scenario.json", scenarios_dir / "mini.json")
    out_dir = tmp_path / "results"

    spec = GridSpec(
        adapters=["naive"],
        modes=["stock", "tuned"],
        seeds=[42],
        noise_rates=[0.20, 0.30, 0.45, 0.60],
        checkpoints=(0, 3, 7),
        scenarios_dir=scenarios_dir,
        out_dir=out_dir,
    )
    paths = run_grid(spec)
    # 1 adapter × 2 modes × 1 seed × 4 noise rates = 8
    assert len(paths) == 8

    for p in paths:
        data = json.loads(p.read_text())
        jsonschema.validate(data, RESULTS_SCHEMA)
        assert data["auc_quality"] > 0, f"no signal in {p.name}"

    charts = tmp_path / "charts"
    render_decay_curves(out_dir, charts / "decay-curves.svg")
    render_sensitivity_grid(out_dir, charts / "sensitivity-grid.svg")
    assert (charts / "decay-curves.svg").stat().st_size > 500
    assert (charts / "sensitivity-grid.svg").stat().st_size > 500
