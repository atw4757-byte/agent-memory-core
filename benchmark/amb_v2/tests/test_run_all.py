"""T-25 — Grid runner tests."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmark.amb_v2.run_all import GridSpec, run_grid


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_dir(tmp_path: Path) -> Path:
    shutil.copy(FIXTURES / "mini_scenario.json", tmp_path / "mini.json")
    return tmp_path


def test_grid_dimensions(tmp_path: Path, mini_dir: Path):
    spec = GridSpec(
        adapters=["naive"],
        modes=["stock", "tuned"],
        seeds=[42, 43],
        noise_rates=[0.30],
        checkpoints=(0, 3),
        scenarios_dir=mini_dir,
        out_dir=tmp_path / "out",
    )
    paths = run_grid(spec)
    # 1 adapter × 2 modes × 2 seeds × 1 noise = 4 runs
    assert len(paths) == 4
    for p in paths:
        assert p.exists()
        data = json.loads(p.read_text())
        assert "auc_quality" in data


def test_skips_already_completed(tmp_path: Path, mini_dir: Path):
    spec = GridSpec(
        adapters=["naive"],
        modes=["stock"],
        seeds=[42],
        noise_rates=[0.30],
        checkpoints=(0, 3),
        scenarios_dir=mini_dir,
        out_dir=tmp_path / "out",
    )
    first = run_grid(spec)
    mtime = first[0].stat().st_mtime_ns
    # Second invocation: should reuse file, not rewrite
    second = run_grid(spec)
    assert second == first
    assert second[0].stat().st_mtime_ns == mtime


def test_filenames_encode_configuration(tmp_path: Path, mini_dir: Path):
    spec = GridSpec(
        adapters=["naive"],
        modes=["stock"],
        seeds=[42],
        noise_rates=[0.30],
        checkpoints=(0, 3),
        scenarios_dir=mini_dir,
        out_dir=tmp_path / "out",
    )
    paths = run_grid(spec)
    name = paths[0].name
    assert "naive" in name
    assert "stock" in name
    assert "s42" in name
    assert "n30" in name or "n0.30" in name
