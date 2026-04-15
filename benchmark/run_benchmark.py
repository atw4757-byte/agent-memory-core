#!/usr/bin/env python3
"""
benchmark/run_benchmark.py — Agentic Memory Benchmark (AMB) runner.

Loads all 10 scenarios (200 queries total), feeds each session into a
MemorySystem adapter, runs all queries, scores results, and writes a
JSON report.

Usage
-----
# Run against agent-memory-core (default)
    python benchmark/run_benchmark.py

# Run against a custom adapter
    python benchmark/run_benchmark.py --adapter my_module.MyAdapter

# Run a single scenario
    python benchmark/run_benchmark.py --scenario 01_personal_assistant

# Save results to a specific path
    python benchmark/run_benchmark.py --output results/my_run.json

# Quiet mode (no per-query output)
    python benchmark/run_benchmark.py --quiet

Adapter Interface
-----------------
Any memory system can be benchmarked by implementing the MemorySystemAdapter
protocol:

    class MyAdapter:
        def ingest_turn(self, session_id: int, role: str, content: str) -> None:
            # Feed a single conversation turn into the memory system.

        def query(self, question: str) -> str:
            # Ask a question; return the answer as a string.

        def reset(self) -> None:
            # Clear all state between scenarios.

The default adapter wraps agent-memory-core's MemoryStore with a simple
add-on-user-turn + search-on-query pattern, producing an answer by
concatenating top-5 retrieved chunks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

# Allow running from project root without install
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from benchmark.metrics import (
    aggregate_scenario_scores,
    score_query,
    composite_score,
)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MemorySystemAdapter(Protocol):
    """Interface any memory system must implement to run the benchmark."""

    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        """Feed one conversation turn into the system."""
        ...

    def query(self, question: str) -> str:
        """Return an answer string given a natural language question."""
        ...

    def reset(self) -> None:
        """Clear all memory state (called between scenarios)."""
        ...


# ---------------------------------------------------------------------------
# Default adapter: agent-memory-core
# ---------------------------------------------------------------------------

class AgentMemoryCoreAdapter:
    """Wraps agent-memory-core MemoryStore for benchmarking.

    Ingestion: adds each user turn as a 'session' chunk.
    Query:     searches top-5 chunks, concatenates their text as the answer.
    """

    def __init__(self, db_path: str | None = None) -> None:
        try:
            from agent_memory_core import MemoryStore
        except ImportError as e:
            raise ImportError(
                "agent-memory-core is not installed. "
                "Run: pip install -e . from the project root."
            ) from e

        import tempfile
        # Use a temp dir per run to avoid cross-scenario contamination
        self._tmp = tempfile.mkdtemp(prefix="amb_")
        self._store = MemoryStore(db_path=db_path or self._tmp)

    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        if role == "user":
            self._store.add(
                content,
                type="session",
                source=f"session_{session_id}",
            )

    def query(self, question: str) -> str:
        results = self._store.search(question, n=5)
        if not results:
            return ""
        return " | ".join(r.text for r in results)

    def reset(self) -> None:
        import shutil
        import tempfile
        # Drop the old DB and start fresh
        try:
            shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:
            pass
        self._tmp = tempfile.mkdtemp(prefix="amb_")
        from agent_memory_core import MemoryStore
        self._store = MemoryStore(db_path=self._tmp)


# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenarios(scenario_filter: str | None = None) -> list[dict]:
    """Load all (or one) scenario JSON files from the scenarios/ directory."""
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No scenario files found in {SCENARIOS_DIR}")

    scenarios = []
    for f in files:
        if scenario_filter and scenario_filter not in f.stem:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            scenarios.append(data)
        except json.JSONDecodeError as e:
            print(f"  [WARN] Failed to parse {f.name}: {e}", file=sys.stderr)

    if not scenarios:
        raise ValueError(f"No scenarios matched filter: {scenario_filter!r}")

    return scenarios


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: dict,
    adapter: MemorySystemAdapter,
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """Run one scenario end-to-end and return per-query scores + aggregates."""
    scenario_id   = scenario.get("scenario_id", "unknown")
    scenario_name = scenario.get("name", scenario_id)
    sessions      = scenario.get("sessions", [])
    queries       = scenario.get("queries", [])

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Scenario: {scenario_name}")
        print(f"  Sessions: {len(sessions)} | Queries: {len(queries)}")
        print(f"{'='*70}")

    # ----- Phase 1: ingest all sessions -----
    t_ingest_start = time.perf_counter()
    for session in sessions:
        session_id = session.get("session_id", 0)
        for turn in session.get("turns", []):
            adapter.ingest_turn(session_id, turn["role"], turn["content"])
    t_ingest = time.perf_counter() - t_ingest_start

    if verbose:
        total_turns = sum(len(s.get("turns", [])) for s in sessions)
        print(f"  Ingested {total_turns} turns in {t_ingest:.2f}s")
        print()

    # ----- Phase 2: run queries -----
    query_scores = []
    t_query_start = time.perf_counter()

    for q in queries:
        question = q.get("query", "")
        if not question:
            continue

        try:
            answer = adapter.query(question)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        scored = score_query(answer, q, retrieved_results=None, k=k)

        if verbose:
            difficulty = q.get("difficulty", "?")
            rtype      = q.get("reasoning_type", "unknown")[:24]
            status     = "PASS" if scored["composite"] >= 5.0 else "FAIL"
            trap_flag  = " [TRAP]" if q.get("trap") else ""
            print(
                f"  [{status}] Q{q['query_id']:>3}  "
                f"diff={difficulty[:4]:<4}  "
                f"type={rtype:<24}  "
                f"score={scored['composite']:4.1f}/10"
                f"{trap_flag}"
            )

        query_scores.append(scored)

    t_query = time.perf_counter() - t_query_start
    aggregates = aggregate_scenario_scores(query_scores)

    if verbose:
        print(f"\n  --- {scenario_name} Summary ---")
        print(f"  Recall:               {aggregates['recall']:.1%}")
        print(f"  Precision:            {aggregates['precision']:.1%}")
        print(f"  Answer Completeness:  {aggregates['answer_completeness']:.1%}")
        print(f"  Temporal Accuracy:    {aggregates['temporal_accuracy']:.1%}")
        print(f"  Composite Score:      {aggregates['composite']:.2f}/10")
        print(f"  Query time:           {t_query:.2f}s ({t_query/len(queries)*1000:.0f}ms/query)")

        # Show hardest traps
        trap_queries = [q for q in queries if q.get("trap")]
        trap_scores  = [s for s in query_scores if s["query_id"] in {t["query_id"] for t in trap_queries}]
        if trap_scores:
            avg_trap = sum(s["composite"] for s in trap_scores) / len(trap_scores)
            print(f"  Trap query avg score: {avg_trap:.2f}/10 ({len(trap_scores)} trap queries)")

    return {
        "scenario_id":   scenario_id,
        "scenario_name": scenario_name,
        "n_sessions":    len(sessions),
        "n_queries":     len(queries),
        "ingest_time_s": round(t_ingest, 3),
        "query_time_s":  round(t_query, 3),
        "aggregates":    aggregates,
        "query_scores":  query_scores,
    }


def run_benchmark(
    adapter: MemorySystemAdapter,
    scenarios: list[dict],
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """Run the full benchmark across all scenarios and compile a report."""
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_name = type(adapter).__name__

    if verbose:
        print(f"\nAgentic Memory Benchmark (AMB)")
        print(f"Run ID:    {run_id}")
        print(f"Adapter:   {adapter_name}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Queries:   {sum(len(s.get('queries', [])) for s in scenarios)}")
        print(f"K:         {k}")

    scenario_results = []
    all_query_scores: list[dict] = []

    for scenario in scenarios:
        adapter.reset()
        result = run_scenario(scenario, adapter, k=k, verbose=verbose)
        scenario_results.append(result)
        all_query_scores.extend(result["query_scores"])

    # Overall aggregates
    overall = aggregate_scenario_scores(all_query_scores)

    # Per-reasoning-type breakdown across all scenarios
    by_type: dict[str, list[float]] = {}
    for q in all_query_scores:
        t = q.get("reasoning_type", "unknown")
        by_type.setdefault(t, []).append(q["composite"])
    overall_by_type = {
        t: round(sum(v) / len(v), 3)
        for t, v in sorted(by_type.items(), key=lambda x: sum(x[1]) / len(x[1]))
    }

    # Trap vs. non-trap breakdown
    trap_scores    = [q for q in all_query_scores if _is_trap_query(q, scenarios)]
    nontrap_scores = [q for q in all_query_scores if not _is_trap_query(q, scenarios)]
    trap_avg    = sum(q["composite"] for q in trap_scores) / len(trap_scores) if trap_scores else 0.0
    nontrap_avg = sum(q["composite"] for q in nontrap_scores) / len(nontrap_scores) if nontrap_scores else 0.0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"\n  OVERALL SCORES (across {len(all_query_scores)} queries)")
        print(f"  {'Recall:':25s} {overall['recall']:.1%}")
        print(f"  {'Precision:':25s} {overall['precision']:.1%}")
        print(f"  {'Answer Completeness:':25s} {overall['answer_completeness']:.1%}")
        print(f"  {'Temporal Accuracy:':25s} {overall['temporal_accuracy']:.1%}")
        print(f"  {'COMPOSITE SCORE:':25s} {overall['composite']:.2f}/10")
        print(f"\n  Trap queries:     {trap_avg:.2f}/10 avg  ({len(trap_scores)} queries)")
        print(f"  Non-trap queries: {nontrap_avg:.2f}/10 avg  ({len(nontrap_scores)} queries)")
        print(f"\n  By Reasoning Type (weakest to strongest):")
        for rtype, score in overall_by_type.items():
            bar = "#" * int(score)
            print(f"    {rtype:<35s} {score:4.1f}  {bar}")
        print(f"\n  Per-Scenario Scores:")
        for r in scenario_results:
            print(f"    {r['scenario_id']:<35s} {r['aggregates']['composite']:4.2f}/10")

    report = {
        "run_id":          run_id,
        "adapter":         adapter_name,
        "k":               k,
        "timestamp":       datetime.now().isoformat(),
        "n_scenarios":     len(scenarios),
        "n_queries":       len(all_query_scores),
        "overall":         overall,
        "overall_by_reasoning_type": overall_by_type,
        "trap_avg":        round(trap_avg, 3),
        "nontrap_avg":     round(nontrap_avg, 3),
        "scenario_results": [
            {
                "scenario_id":   r["scenario_id"],
                "scenario_name": r["scenario_name"],
                "composite":     r["aggregates"]["composite"],
                "recall":        r["aggregates"]["recall"],
                "precision":     r["aggregates"]["precision"],
                "answer_completeness": r["aggregates"]["answer_completeness"],
                "temporal_accuracy":   r["aggregates"]["temporal_accuracy"],
                "n_queries":     r["n_queries"],
                "ingest_time_s": r["ingest_time_s"],
                "query_time_s":  r["query_time_s"],
                "by_difficulty": r["aggregates"].get("by_difficulty", {}),
                "query_scores":  r["query_scores"],
            }
            for r in scenario_results
        ],
    }
    return report


def _is_trap_query(query_score: dict, scenarios: list[dict]) -> bool:
    """Return True if this query_id corresponds to a trap query in the scenario data."""
    qid = query_score.get("query_id")
    for scenario in scenarios:
        for q in scenario.get("queries", []):
            if q.get("query_id") == qid and q.get("trap"):
                return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_adapter(adapter_path: str | None) -> MemorySystemAdapter:
    """Load adapter from dotted module path or return default."""
    if not adapter_path:
        return AgentMemoryCoreAdapter()

    # Expected format: "module.path.ClassName"
    parts = adapter_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Adapter must be a dotted path like 'mymodule.MyAdapter', got: {adapter_path!r}"
        )
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()
    if not isinstance(instance, MemorySystemAdapter):
        raise TypeError(
            f"{class_name} does not implement MemorySystemAdapter protocol "
            "(needs ingest_turn, query, reset methods)."
        )
    return instance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Memory Benchmark (AMB) — evaluate any memory system across 200 queries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Dotted path to custom adapter class (e.g. 'mymod.MyAdapter'). "
             "Default: agent-memory-core AgentMemoryCoreAdapter.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run only scenarios whose filename contains this string.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the JSON report. Defaults to benchmark/results/RUNID.json.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k results for recall/precision metrics. Default 5.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query output.",
    )
    args = parser.parse_args()

    # Load adapter
    try:
        adapter = _load_adapter(args.adapter)
    except Exception as e:
        print(f"[ERROR] Could not load adapter: {e}", file=sys.stderr)
        sys.exit(1)

    # Load scenarios
    try:
        scenarios = load_scenarios(args.scenario)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Run
    report = run_benchmark(
        adapter=adapter,
        scenarios=scenarios,
        k=args.k,
        verbose=not args.quiet,
    )

    # Save
    output_path = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f"{report['run_id']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
