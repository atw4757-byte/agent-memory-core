"""Metrics — pure functions over QueryResults.

Composite formula is FROZEN. Changing any weight requires a v2.1 release and
a 60-day grace period. The AST-inspection test guards against silent drift.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    scenario_id: str
    actual_answer: str
    expected_answer: str
    aliases: tuple[str, ...]
    resolution_type: str
    chunk_type: str
    superseded_value: str | None
    top1_chunk: str | None = None
    returned_chunks: tuple[str, ...] = ()


def _norm(s: str) -> str:
    return s.strip().lower()


def _matches(actual: str, expected: str, aliases: tuple[str, ...]) -> bool:
    """Substring-containment matching.

    Adapters often return retrieval context, not extracted answers. We score
    whether the expected answer (or any alias) appears as a substring in the
    actual answer. Case-insensitive; whitespace-normalized.
    """
    a = _norm(actual)
    if _norm(expected) in a:
        return True
    for alias in aliases:
        if _norm(alias) in a:
            return True
    return False


def answer_accuracy(results: list[QueryResult]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if _matches(r.actual_answer, r.expected_answer, r.aliases))
    return correct / len(results)


def top1_answer_accuracy(results: list[QueryResult]) -> float:
    """v2.3 muddiness metric: answer must appear in the TOP-ranked chunk.

    Simulates LLM misattribution: real models bias toward the most-prominent
    retrieved chunk. If top-1 is a confuser or trap, generation goes wrong
    even though the right answer sits at rank 3. Current ``answer_accuracy``
    gives full credit for "answer somewhere in top-k"; this metric doesn't.

    Falls back to actual_answer when top1_chunk is None (pre-v2.3 results).
    """
    if not results:
        return 0.0
    correct = 0
    for r in results:
        target = r.top1_chunk if r.top1_chunk is not None else r.actual_answer
        if _matches(target, r.expected_answer, r.aliases):
            correct += 1
    return correct / len(results)


def confuser_resistance(
    results: list[QueryResult],
    confuser_texts: dict[str, tuple[str, ...]],
) -> float:
    """v2.3 muddiness metric: fraction of queries where NO confuser chunk
    appeared in top-k returned_chunks.

    Input map: query_id → tuple of confuser chunk texts for that query.
    For each result with returned_chunks populated, check whether any
    returned chunk contains any of its confusers. Score 1 = resisted,
    0 = polluted.

    Queries without confusers, or without returned_chunks populated,
    are skipped. Returns 1.0 when none qualify (no failures triggered).
    """
    eligible = [r for r in results
                if r.returned_chunks and confuser_texts.get(r.query_id)]
    if not eligible:
        return 1.0
    resisted = 0
    for r in eligible:
        confusers = confuser_texts[r.query_id]
        polluted = any(
            any(_norm(c) in _norm(chunk) for c in confusers)
            for chunk in r.returned_chunks
        )
        if not polluted:
            resisted += 1
    return resisted / len(eligible)


def contradiction_resolution(results: list[QueryResult]) -> float:
    """Fraction of contradiction-eligible queries that returned the *new* answer.

    If there are no contradiction queries in the eligible set, returns 1.0
    (no failure mode triggered).
    """
    contradictions = [r for r in results if r.resolution_type == "contradiction"]
    if not contradictions:
        return 1.0
    correct = sum(1 for r in contradictions if _matches(r.actual_answer, r.expected_answer, r.aliases))
    return correct / len(contradictions)


def stale_fact_rate(results: list[QueryResult]) -> float:
    """Fraction of queries (with a known superseded value) where the system's
    answer contains the OLD value but NOT the current one.

    Lower is better. Skips queries with no superseded_value. Note: if the
    answer contains BOTH old and new (common in retrieval-context returns),
    we credit the system for retrieving the current truth — only penalize
    when the answer is purely stale.
    """
    eligible = [r for r in results if r.superseded_value is not None]
    if not eligible:
        return 0.0
    stale = 0
    for r in eligible:
        a = _norm(r.actual_answer)
        old = _norm(r.superseded_value or "")
        new = _norm(r.expected_answer)
        if old in a and new not in a:
            stale += 1
    return stale / len(eligible)


def salience_preservation(results: list[QueryResult]) -> float:
    """For credential-type queries: fraction returned correctly in top-1.

    Returns 1.0 when no credential queries exist.
    """
    creds = [r for r in results if r.chunk_type == "credential"]
    if not creds:
        return 1.0
    correct = sum(1 for r in creds if _matches(r.actual_answer, r.expected_answer, r.aliases))
    return correct / len(creds)


def quality_at(*, answer: float, contradiction: float, stale: float, salience: float) -> float:
    """FROZEN composite. Do NOT change weights without bumping spec version.

    Quality@T = 0.40·answer + 0.30·contradiction + 0.15·(1 − stale) + 0.15·salience
    """
    return (
        0.40 * answer
        + 0.30 * contradiction
        + 0.15 * (1.0 - stale)
        + 0.15 * salience
    )


def quality_at_v2_3(
    *,
    top1_answer: float,
    any_answer: float,
    contradiction: float,
    stale: float,
    salience: float,
    confuser_resist: float,
) -> float:
    """v2.3 composite. Top-1 weighted more than any-of-top-k to reflect that
    real LLM generation mis-attributes when the top chunk is wrong.

    Weights (sum to 1.0):
      0.25·top1_answer + 0.15·any_answer + 0.20·contradiction
      + 0.10·(1 − stale) + 0.10·salience + 0.20·confuser_resist
    """
    return (
        0.25 * top1_answer
        + 0.15 * any_answer
        + 0.20 * contradiction
        + 0.10 * (1.0 - stale)
        + 0.10 * salience
        + 0.20 * confuser_resist
    )


def auc_quality(checkpoints: list[tuple[int, float]]) -> float:
    """Trapezoid integration over (day, quality) checkpoints.

    Input may be unsorted; we sort by day before integrating.
    """
    if not checkpoints or len(checkpoints) < 2:
        return 0.0
    pts = sorted(checkpoints, key=lambda p: p[0])
    area = 0.0
    for (d0, q0), (d1, q1) in zip(pts, pts[1:]):
        area += (d1 - d0) * (q0 + q1) / 2.0
    return area


def temporal_improvement(checkpoints: list[tuple[int, float]]) -> float:
    """Late-half mean quality minus early-half mean quality.

    Measures whether the system IMPROVES as more context accumulates.
    A true memory system should score higher here than a stateless retrieval
    loop. Positive = learns over time; zero = flat; negative = degrades.

    Separate from AUC (total area): two systems can have identical AUC with
    very different trajectories. Temporal improvement isolates the slope.

    Returns 0.0 when fewer than 2 checkpoints provided.
    """
    if not checkpoints or len(checkpoints) < 2:
        return 0.0
    pts = sorted(checkpoints, key=lambda p: p[0])
    mid = len(pts) // 2
    early = [q for _, q in pts[:mid]]
    late = [q for _, q in pts[mid:]]
    if not early or not late:
        return 0.0
    return sum(late) / len(late) - sum(early) / len(early)
