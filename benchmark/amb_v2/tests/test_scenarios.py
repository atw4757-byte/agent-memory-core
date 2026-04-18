"""T-04, T-05 — Scenario loader (public + held-out encrypted)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmark.amb_v2.scenarios import (
    ScenarioBundle,
    load_held_out,
    load_public_scenarios,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_public_scenarios_loads_mini(tmp_path: Path):
    shutil.copy(FIXTURES / "mini_scenario.json", tmp_path / "mini.json")
    bundles = load_public_scenarios(tmp_path)
    assert len(bundles) == 1
    b = bundles[0]
    assert isinstance(b, ScenarioBundle)
    assert b.scenario_id == "mini"
    assert b.is_held_out is False
    assert len(b.timeline) == 7
    assert len(b.queries) == 4


def test_load_public_scenarios_empty_dir(tmp_path: Path):
    assert load_public_scenarios(tmp_path) == []


def test_load_public_scenarios_skips_non_json(tmp_path: Path):
    (tmp_path / "README.md").write_text("# not a scenario")
    assert load_public_scenarios(tmp_path) == []


def test_load_held_out_missing_key_skips_silently(tmp_path: Path):
    """When the key file does not exist, held-out load is a no-op (returns [])."""
    (tmp_path / "h01.json.age").write_bytes(b"not really encrypted")
    assert load_held_out(tmp_path, key_path=tmp_path / "missing.key") == []


@pytest.mark.skipif(shutil.which("age") is None, reason="age binary not installed")
def test_load_held_out_round_trip(tmp_path: Path):
    """Encrypt a fixture with a generated age key, then decrypt via load_held_out."""
    keypair = tmp_path / "id.key"
    subprocess.run(["age-keygen", "-o", str(keypair)], check=True, capture_output=True)
    pub_line = next(
        line for line in keypair.read_text().splitlines()
        if line.startswith("# public key:")
    )
    pubkey = pub_line.split(": ", 1)[1].strip()

    plaintext = (FIXTURES / "mini_scenario.json").read_text()
    cipher = tmp_path / "h01.json.age"
    proc = subprocess.run(
        ["age", "-r", pubkey, "-o", str(cipher)],
        input=plaintext.encode(), check=True, capture_output=True,
    )
    assert cipher.exists()

    bundles = load_held_out(tmp_path, key_path=keypair)
    assert len(bundles) == 1
    assert bundles[0].is_held_out is True
    assert bundles[0].scenario_id == "mini"
