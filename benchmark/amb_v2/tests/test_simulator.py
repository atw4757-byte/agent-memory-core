"""T-06..T-09 — Simulator tests."""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import pytest

from benchmark.amb_v2.scenarios import load_public_scenarios
from benchmark.amb_v2.simulator import simulate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_bundles(tmp_path: Path):
    import shutil
    shutil.copy(FIXTURES / "mini_scenario.json", tmp_path / "mini.json")
    return load_public_scenarios(tmp_path)


def test_determinism_same_seed_same_output(mini_bundles):
    a = list(simulate(mini_bundles, seed=42, days=10))
    b = list(simulate(mini_bundles, seed=42, days=10))
    assert a == b


def test_different_seed_different_output(mini_bundles):
    a = list(simulate(mini_bundles, seed=1, days=20, noise_rate=0.45))
    b = list(simulate(mini_bundles, seed=2, days=20, noise_rate=0.45))
    assert a != b


def test_chunks_yielded_in_day_order(mini_bundles):
    out = list(simulate(mini_bundles, seed=42, days=10))
    days = [d for d, _ in out]
    assert days == sorted(days)
    assert days == list(range(10))


def test_yields_empty_list_for_quiet_days(mini_bundles):
    out = list(simulate(mini_bundles, seed=42, days=10, noise_rate=0.0))
    # No noise + only 5 days have scenario events → days 5..9 are empty
    by_day = dict(out)
    assert by_day[5] == []
    assert by_day[9] == []


def test_chunk_ids_unique_across_run(mini_bundles):
    out = list(simulate(mini_bundles, seed=42, days=90, noise_rate=0.45))
    ids = [c.id for _, chunks in out for c in chunks]
    assert len(ids) == len(set(ids))


def test_noise_rate_calibrated_at_30_pct(mini_bundles):
    out = list(simulate(mini_bundles, seed=42, days=90, noise_rate=0.30))
    chunks = [c for _, chunks in out for c in chunks]
    counts = Counter(c.type for c in chunks)
    total = sum(counts.values())
    noise_frac = counts.get("noise", 0) / total
    # ±5% tolerance — distribution noisy at small N, but mini has enough days
    assert 0.20 <= noise_frac <= 0.40


def test_noise_rate_calibrated_at_45_pct(mini_bundles):
    out = list(simulate(mini_bundles, seed=42, days=90, noise_rate=0.45))
    chunks = [c for _, chunks in out for c in chunks]
    counts = Counter(c.type for c in chunks)
    total = sum(counts.values())
    noise_frac = counts.get("noise", 0) / total
    assert 0.35 <= noise_frac <= 0.55


def test_soft_cap_emits_warning_when_exceeded(mini_bundles):
    """Force a day to exceed 200 chunks via huge noise rate × many bundles."""
    big = mini_bundles * 200  # 200x mini = ~1400 timeline events spread across days
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(simulate(big, seed=42, days=5, noise_rate=0.95))
        assert any("soft cap" in str(rec.message).lower() for rec in w)


def test_supersedes_preserved(mini_bundles):
    out = list(simulate(mini_bundles, seed=42, days=10))
    chunks = [c for _, chunks in out for c in chunks]
    update = next(c for c in chunks if c.type == "update")
    assert update.supersedes == "mini-d000-0"


def test_filler_density_default_is_zero(mini_bundles):
    """v2.2 scale knob: default=0 must preserve v2.1 byte-identical output."""
    a = list(simulate(mini_bundles, seed=42, days=10, noise_rate=0.30))
    b = list(simulate(mini_bundles, seed=42, days=10, noise_rate=0.30,
                      filler_facts_per_day=0))
    assert a == b


def test_filler_density_scales_chunk_count(mini_bundles):
    """filler_facts_per_day=K adds approximately K*days filler chunks."""
    base = list(simulate(mini_bundles, seed=42, days=90, noise_rate=0.30,
                         filler_facts_per_day=0))
    n_base = sum(len(cs) for _, cs in base)
    scaled = list(simulate(mini_bundles, seed=42, days=90, noise_rate=0.30,
                           filler_facts_per_day=10))
    n_scaled = sum(len(cs) for _, cs in scaled)
    # Roughly 900 fillers + noise recalibration. Must be substantially larger.
    assert n_scaled > n_base * 3, (
        f"scaled={n_scaled} not meaningfully larger than base={n_base}"
    )


def test_filler_chunks_do_not_leak_expected_answers(mini_bundles):
    """Fillers must not contain any query's expected_answer as a substring."""
    out = list(simulate(mini_bundles, seed=42, days=90, noise_rate=0.30,
                        filler_facts_per_day=20))
    filler_chunks = [c for _, cs in out for c in cs if c.id.startswith("filler-")]
    expected_answers = [q.expected_answer for b in mini_bundles for q in b.queries]
    for c in filler_chunks:
        for ans in expected_answers:
            assert ans.lower() not in c.text.lower(), (
                f"filler {c.id!r} contains expected_answer {ans!r}: {c.text}"
            )


def test_filler_is_deterministic(mini_bundles):
    a = list(simulate(mini_bundles, seed=42, days=30, noise_rate=0.30,
                      filler_facts_per_day=5))
    b = list(simulate(mini_bundles, seed=42, days=30, noise_rate=0.30,
                      filler_facts_per_day=5))
    assert a == b
