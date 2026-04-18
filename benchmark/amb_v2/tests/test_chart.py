"""T-26 — chart generation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmark.amb_v2.chart import render_decay_curves, render_sensitivity_grid
from benchmark.amb_v2.run_all import GridSpec, run_grid


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def grid_results(tmp_path: Path) -> Path:
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
    run_grid(spec)
    return out_dir


def test_produces_decay_curves_svg(tmp_path: Path, grid_results: Path):
    out = tmp_path / "decay-curves.svg"
    render_decay_curves(grid_results, out)
    assert out.exists() and out.stat().st_size > 500


def test_produces_sensitivity_grid(tmp_path: Path, grid_results: Path):
    out = tmp_path / "sensitivity-grid.svg"
    render_sensitivity_grid(grid_results, out)
    assert out.exists() and out.stat().st_size > 500
