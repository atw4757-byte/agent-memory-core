"""
benchmark/metrics.py — Scoring functions for the Agentic Memory Benchmark (AMB).

All scoring is at the answer-string level. The benchmark treats answers as strings
and expected_answer / expected_facts as ground truth. Each metric returns a float
in [0.0, 1.0] unless otherwise noted.

Metrics
-------
recall_at_k(results, expected_facts, k)
    At least one result in the top-k contains any expected fact. Binary.

precision_at_k(results, expected_facts, k)
    Fraction of top-k results containing at least one expected fact.

answer_completeness(answer, expected_facts)
    Fraction of expected facts found in the generated answer text.

temporal_accuracy(answer, current_truth, historical_truth)
    Penalises confusion between current and historical state.
    Returns 1.0 (correct), 0.5 (partial/ambiguous), or 0.0 (wrong state used).

contradiction_resolution_rate(answers, contradictions)
    Fraction of contradiction pairs where the system returned the NEWER value.

composite_score(recall, precision, answer_completeness,
                temporal_accuracy, contradiction_rate)
    Weighted composite on a 0–10 scale.
    Weights: recall 0.25, precision 0.20, answer 0.25,
             temporal 0.15, contradiction 0.15.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for soft matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_fact(text: str, fact: str) -> bool:
    """Return True if *fact* appears as a substring in *text* (normalised)."""
    return _normalise(fact) in _normalise(text)


def _any_fact(text: str, facts: list[str]) -> bool:
    return any(_contains_fact(text, f) for f in facts)


def _result_text(result) -> str:
    """Extract text from a MemoryResult, dict, or plain string."""
    if hasattr(result, "text"):
        return result.text
    if isinstance(result, dict):
        return result.get("text", "")
    return str(result)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    results: list,
    expected_facts: list[str],
    k: int = 5,
) -> float:
    """Recall@k — 1.0 if any top-k result contains any expected fact, else 0.0.

    Parameters
    ----------
    results:        List of MemoryResult / dict / str.
    expected_facts: Substrings that constitute a correct retrieval.
    k:              How many top results to consider.

    Returns
    -------
    float: 1.0 or 0.0.
    """
    if not results or not expected_facts:
        return 0.0
    top_k = results[:k]
    for r in top_k:
        if _any_fact(_result_text(r), expected_facts):
            return 1.0
    return 0.0


def precision_at_k(
    results: list,
    expected_facts: list[str],
    k: int = 5,
) -> float:
    """Precision@k — fraction of top-k results containing at least one expected fact.

    Parameters
    ----------
    results:        List of MemoryResult / dict / str.
    expected_facts: Ground-truth substrings.
    k:              How many top results to evaluate.

    Returns
    -------
    float in [0.0, 1.0].
    """
    if not results or not expected_facts:
        return 0.0
    top_k = results[:k]
    relevant = sum(1 for r in top_k if _any_fact(_result_text(r), expected_facts))
    return relevant / len(top_k)


def answer_completeness(
    answer: str,
    expected_facts: list[str],
) -> float:
    """Answer completeness — fraction of expected facts present in the answer.

    Each fact is checked as a substring (normalised). This measures whether
    the answer covers all required information, not just any of it.

    Parameters
    ----------
    answer:         The generated answer string.
    expected_facts: All facts that must appear.

    Returns
    -------
    float in [0.0, 1.0]. 1.0 means all facts found.
    """
    if not expected_facts:
        return 1.0
    if not answer:
        return 0.0
    hits = sum(1 for f in expected_facts if _contains_fact(answer, f))
    return hits / len(expected_facts)


def temporal_accuracy(
    answer: str,
    current_truth: str,
    historical_truth: str,
) -> float:
    """Temporal accuracy — rewards using current state, penalises using stale state.

    The key failure mode in long-horizon memory is returning a value that was
    true historically but has since been superseded.

    Scoring
    -------
    1.0  — answer contains current_truth (correct, regardless of historical mention)
    0.5  — answer contains neither (no relevant claim made)
    0.0  — answer contains historical_truth but not current_truth (stale recall)

    Parameters
    ----------
    answer:           The generated answer string.
    current_truth:    The currently-correct fact substring.
    historical_truth: A fact that was true earlier but is now superseded.

    Returns
    -------
    float: 1.0, 0.5, or 0.0.
    """
    has_current = _contains_fact(answer, current_truth)
    has_historical = _contains_fact(answer, historical_truth)

    if has_current:
        return 1.0
    if has_historical and not has_current:
        return 0.0
    # Neither: ambiguous / no claim
    return 0.5


def contradiction_resolution_rate(
    answers: list[str],
    contradictions: list[dict],
) -> float:
    """Contradiction resolution rate — fraction of contradictions correctly resolved.

    Each contradiction is a dict:
        {
            "answer_index": int,          # which answer in `answers` to check
            "old_value":    str,          # the superseded (wrong) value
            "new_value":    str,          # the current (correct) value
        }

    A contradiction is 'resolved' if the answer contains new_value but NOT
    old_value (or contains both but explicitly signals the new one is current).
    We use the simpler rule: new_value present → resolved.

    Parameters
    ----------
    answers:        List of answer strings (one per query).
    contradictions: List of contradiction dicts as described above.

    Returns
    -------
    float in [0.0, 1.0]. 1.0 means all contradictions correctly resolved.
    """
    if not contradictions:
        return 1.0
    resolved = 0
    for c in contradictions:
        idx = c.get("answer_index", 0)
        if idx >= len(answers):
            continue
        ans = answers[idx]
        if _contains_fact(ans, c["new_value"]):
            resolved += 1
    return resolved / len(contradictions)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

_COMPOSITE_WEIGHTS = {
    "recall":           0.25,
    "precision":        0.20,
    "answer":           0.25,
    "temporal":         0.15,
    "contradiction":    0.15,
}


def composite_score(
    recall: float,
    precision: float,
    answer_completeness_score: float,
    temporal_accuracy_score: float = 1.0,
    contradiction_rate: float = 1.0,
    weights: Optional[dict[str, float]] = None,
    scale: float = 10.0,
) -> float:
    """Weighted composite memory quality score.

    Parameters
    ----------
    recall:                  Recall@k (fraction or 0/1).
    precision:               Precision@k.
    answer_completeness_score: Fraction of expected facts in the answer.
    temporal_accuracy_score: Temporal accuracy (1.0/0.5/0.0 or mean over queries).
    contradiction_rate:      Contradiction resolution rate.
    weights:                 Override the default weight dict. Must sum to 1.0.
    scale:                   Output scale. Default 10.0 (i.e., score out of 10).

    Returns
    -------
    float: Composite score in [0.0, scale].
    """
    w = weights or _COMPOSITE_WEIGHTS
    total_weight = sum(w.values())

    raw = (
        w.get("recall", 0)        * recall
        + w.get("precision", 0)   * precision
        + w.get("answer", 0)      * answer_completeness_score
        + w.get("temporal", 0)    * temporal_accuracy_score
        + w.get("contradiction", 0) * contradiction_rate
    )
    # Normalise in case weights don't sum exactly to 1.0
    normalised = raw / total_weight if total_weight > 0 else 0.0
    return round(normalised * scale, 2)


# ---------------------------------------------------------------------------
# Convenience: score a batch of (answer, query_def) pairs
# ---------------------------------------------------------------------------

def score_query(
    answer: str,
    query_def: dict,
    retrieved_results: Optional[list] = None,
    k: int = 5,
) -> dict:
    """Score a single query/answer pair against a query_def from an AMB scenario.

    Parameters
    ----------
    answer:           The string answer produced by the memory system.
    query_def:        One entry from scenario["queries"]. Must have:
                        - "expected_answer" (str)
                        - optionally "current_truth" and "historical_truth"
                          for temporal queries
    retrieved_results: Raw retrieval results (for recall/precision). If None,
                      the answer string is used as a proxy (treats it as a
                      single result).
    k:                Top-k for recall/precision.

    Returns
    -------
    dict with keys: recall, precision, answer_completeness, temporal_accuracy,
                    composite, query_id, difficulty, reasoning_type.
    """
    expected_answer = query_def.get("expected_answer", "")
    # For recall/precision, split expected_answer into key phrases
    # (words longer than 4 chars, deduped) as lightweight expected_facts proxy
    expected_facts = _extract_key_phrases(expected_answer)

    # Fall back: use answer as single retrieval result if no retrieved results
    results_for_scoring = retrieved_results if retrieved_results is not None else [answer]

    rec  = recall_at_k(results_for_scoring, expected_facts, k=k)
    prec = precision_at_k(results_for_scoring, expected_facts, k=k)
    comp = answer_completeness(answer, expected_facts)

    # Temporal accuracy — only meaningful for queries with explicit truth pairs
    current_truth    = query_def.get("current_truth", "")
    historical_truth = query_def.get("historical_truth", "")
    if current_truth and historical_truth:
        temp = temporal_accuracy(answer, current_truth, historical_truth)
    else:
        temp = 1.0  # Not a temporal test — no penalty

    score = composite_score(rec, prec, comp, temp, contradiction_rate=1.0)

    return {
        "query_id":          query_def.get("query_id"),
        "reasoning_type":    query_def.get("reasoning_type", "unknown"),
        "difficulty":        query_def.get("difficulty", "unknown"),
        "recall":            rec,
        "precision":         prec,
        "answer_completeness": comp,
        "temporal_accuracy": temp,
        "composite":         score,
        "answer":            answer,
        "expected_answer":   expected_answer,
    }


def _extract_key_phrases(text: str, min_len: int = 4) -> list[str]:
    """Extract meaningful words from expected_answer as lightweight expected_facts."""
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'_-]*", text)
    # Keep words that are long enough and likely meaningful
    # Also keep numbers (could be prices, IDs, percentages)
    seen = set()
    phrases = []
    for w in words:
        w_lower = w.lower()
        if len(w) >= min_len and w_lower not in seen:
            seen.add(w_lower)
            phrases.append(w)
    return phrases


# ---------------------------------------------------------------------------
# Aggregate metrics across a scenario
# ---------------------------------------------------------------------------

def aggregate_scenario_scores(query_scores: list[dict]) -> dict:
    """Aggregate per-query scores into scenario-level statistics.

    Parameters
    ----------
    query_scores: List of dicts returned by score_query().

    Returns
    -------
    dict with mean and per-difficulty/per-reasoning-type breakdowns.
    """
    if not query_scores:
        return {}

    n = len(query_scores)
    means = {
        "recall":              sum(q["recall"] for q in query_scores) / n,
        "precision":           sum(q["precision"] for q in query_scores) / n,
        "answer_completeness": sum(q["answer_completeness"] for q in query_scores) / n,
        "temporal_accuracy":   sum(q["temporal_accuracy"] for q in query_scores) / n,
        "composite":           sum(q["composite"] for q in query_scores) / n,
        "n_queries":           n,
    }

    # Breakdown by difficulty
    by_difficulty: dict[str, list[float]] = {}
    for q in query_scores:
        d = q.get("difficulty", "unknown")
        by_difficulty.setdefault(d, []).append(q["composite"])
    means["by_difficulty"] = {
        d: round(sum(scores) / len(scores), 3)
        for d, scores in by_difficulty.items()
    }

    # Breakdown by reasoning type
    by_type: dict[str, list[float]] = {}
    for q in query_scores:
        t = q.get("reasoning_type", "unknown")
        by_type.setdefault(t, []).append(q["composite"])
    means["by_reasoning_type"] = {
        t: round(sum(scores) / len(scores), 3)
        for t, scores in by_type.items()
    }

    # Round means
    for key in ["recall", "precision", "answer_completeness", "temporal_accuracy", "composite"]:
        means[key] = round(means[key], 3)

    return means
