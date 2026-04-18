"""T-24 — CLI entry point."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.amb_v2.run", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--adapter" in result.stdout
    assert "--mode" in result.stdout
    assert "--seed" in result.stdout
    assert "--noise-rate" in result.stdout


def test_cli_runs_naive_mini(tmp_path: Path):
    fixtures = Path(__file__).parent / "fixtures"
    out = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "benchmark.amb_v2.run",
            "--adapter", "naive",
            "--mode", "stock",
            "--seed", "42",
            "--noise-rate", "0.30",
            "--scenarios", str(fixtures),
            "--checkpoints", "0,3,7",
            "--out", str(out),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["adapter"] == "naive-append-only"
    assert data["mode"] == "stock"
    assert data["seed"] == 42
    assert len(data["checkpoints"]) == 3
