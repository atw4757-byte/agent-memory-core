"""T-10..T-14 — Metrics tests."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from benchmark.amb_v2 import metrics
from benchmark.amb_v2.metrics import (
    QueryResult,
    answer_accuracy,
    auc_quality,
    contradiction_resolution,
    quality_at,
    salience_preservation,
    stale_fact_rate,
)


def qr(qid="q1", scenario="x", actual="A", expected="A", aliases=None,
       resolution="stable", chunk_type="fact", superseded_value=None):
    return QueryResult(
        query_id=qid, scenario_id=scenario,
        actual_answer=actual, expected_answer=expected,
        aliases=tuple(aliases or ()),
        resolution_type=resolution,
        chunk_type=chunk_type,
        superseded_value=superseded_value,
    )


# T-10 answer accuracy ────────────────────────────────────────────────

def test_answer_accuracy_all_correct_returns_1():
    rs = [qr(actual="X", expected="X"), qr(actual="Y", expected="Y")]
    assert answer_accuracy(rs) == 1.0


def test_answer_accuracy_all_wrong_returns_0():
    rs = [qr(actual="A", expected="X"), qr(actual="B", expected="Y")]
    assert answer_accuracy(rs) == 0.0


def test_answer_accuracy_alias_table_honored():
    rs = [qr(actual="NYC", expected="New York", aliases=["NYC", "New York City"])]
    assert answer_accuracy(rs) == 1.0


def test_answer_accuracy_case_insensitive():
    rs = [qr(actual="apple", expected="Apple")]
    assert answer_accuracy(rs) == 1.0


def test_answer_accuracy_empty_returns_0():
    assert answer_accuracy([]) == 0.0


# T-11 contradiction resolution ──────────────────────────────────────

def test_contradiction_resolution_filters_to_changed_queries():
    rs = [
        qr(qid="q1", actual="X", expected="X", resolution="stable"),
        qr(qid="q2", actual="NEW", expected="NEW", resolution="contradiction"),
        qr(qid="q3", actual="OLD", expected="NEW", resolution="contradiction"),
    ]
    # Only q2 + q3 are contradictions; q2 correct, q3 wrong → 0.5
    assert contradiction_resolution(rs) == 0.5


def test_contradiction_resolution_no_contradiction_queries_returns_1():
    """When there are no contradictions in the eligible set, score is 1.0 (no failures)."""
    rs = [qr(resolution="stable")]
    assert contradiction_resolution(rs) == 1.0


# T-12 stale fact rate ───────────────────────────────────────────────

def test_stale_fact_rate_flags_superseded_returns():
    rs = [
        qr(qid="q1", actual="OLD", expected="NEW", superseded_value="OLD"),
        qr(qid="q2", actual="NEW", expected="NEW", superseded_value="OLD"),
    ]
    assert stale_fact_rate(rs) == 0.5


def test_stale_fact_rate_returns_0_when_all_current():
    rs = [qr(actual="NEW", expected="NEW", superseded_value="OLD")]
    assert stale_fact_rate(rs) == 0.0


def test_stale_fact_rate_skips_queries_without_supersede():
    """Queries with no superseded_value can't be stale; they don't count."""
    rs = [qr(actual="anything", expected="X", superseded_value=None)]
    assert stale_fact_rate(rs) == 0.0


# T-13 salience preservation ─────────────────────────────────────────

def test_salience_filters_to_credentials():
    rs = [
        qr(actual="pwd", expected="pwd", chunk_type="credential"),
        qr(actual="X", expected="Y", chunk_type="fact"),  # ignored
    ]
    assert salience_preservation(rs) == 1.0


def test_salience_top_1_only():
    rs = [
        qr(actual="wrong", expected="right", chunk_type="credential"),
        qr(actual="right", expected="right", chunk_type="credential"),
    ]
    assert salience_preservation(rs) == 0.5


def test_salience_no_credentials_returns_1():
    """No credential queries → no failures → 1.0."""
    rs = [qr(chunk_type="fact")]
    assert salience_preservation(rs) == 1.0


# T-14 composite + AUC ───────────────────────────────────────────────

def test_quality_formula_weights_frozen_in_source():
    """Refuse silent weight drift. AST inspection of metrics.py asserts literals present."""
    src = Path(inspect.getsourcefile(metrics)).read_text()
    tree = ast.parse(src)
    consts: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            consts.append(node.value)
    for w in (0.40, 0.30, 0.15):
        assert w in consts, f"composite weight {w} not found in metrics.py source"


def test_quality_at_all_perfect():
    score = quality_at(
        answer=1.0, contradiction=1.0, stale=0.0, salience=1.0,
    )
    assert score == pytest.approx(1.0)


def test_quality_at_all_zero():
    score = quality_at(answer=0.0, contradiction=0.0, stale=1.0, salience=0.0)
    assert score == pytest.approx(0.0)


def test_quality_at_known_mix():
    # 0.40*0.8 + 0.30*0.5 + 0.15*(1-0.2) + 0.15*0.9 = 0.32 + 0.15 + 0.12 + 0.135 = 0.725
    score = quality_at(answer=0.8, contradiction=0.5, stale=0.2, salience=0.9)
    assert score == pytest.approx(0.725)


def test_auc_trapezoid_constant_curve():
    """Constant Quality of 0.8 over days 0..90 → AUC = 0.8 * 90 = 72."""
    pts = [(0, 0.8), (7, 0.8), (14, 0.8), (30, 0.8), (60, 0.8), (90, 0.8)]
    assert auc_quality(pts) == pytest.approx(72.0)


def test_auc_trapezoid_linear_curve():
    """Linear from 1.0 → 0.0 over 0..90 → mean 0.5 × 90 = 45."""
    pts = [(0, 1.0), (90, 0.0)]
    assert auc_quality(pts) == pytest.approx(45.0)


def test_auc_handles_unsorted_input():
    """If checkpoints arrive out of order, sort by day before integrating."""
    pts = [(90, 0.0), (0, 1.0)]
    assert auc_quality(pts) == pytest.approx(45.0)
