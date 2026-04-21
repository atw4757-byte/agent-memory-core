#!/usr/bin/env python3
"""
benchmark/run_all.py — Run AMB against all competitor adapters and print a comparison table.

Adapters run:
  1. full_context         — Upper bound: stores and returns ALL turns (oracle ceiling)
  2. archon_memory_core    — Our full system with salience, reranking, working memory
  3. naive_vector         — Plain ChromaDB, no salience, no reranking (baseline)
  4. langchain_window     — LangChain ConversationBufferWindowMemory k=10

Output:
  - Per-adapter benchmark results (quiet mode, no per-query noise)
  - Comparison table printed to stdout
  - benchmark/results/comparison.json

Usage:
    python benchmark/run_all.py
    python benchmark/run_all.py --verbose          # show per-query scores
    python benchmark/run_all.py --scenario 01      # single scenario
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Allow running from project root without install
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# Suppress Pydantic v1 compatibility warnings from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

from benchmark.run_benchmark import run_benchmark, load_scenarios

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Adapter registry — (display_name, module_path, class_name)
# ---------------------------------------------------------------------------

ADAPTERS = [
    (
        "FullContext (oracle ceiling)",
        "benchmark.adapters.full_context",
        "FullContextAdapter",
    ),
    (
        "archon-memory-core (full)",
        "benchmark.adapters.archon_memory_core_adapter",
        "AgentMemoryCoreFullAdapter",
    ),
    (
        "Naive ChromaDB (baseline)",
        "benchmark.adapters.naive_vector",
        "NaiveVectorAdapter",
    ),
    (
        "LangChain Window (k=10)",
        "benchmark.adapters.langchain_adapter",
        "LangChainWindowAdapter",
    ),
]


def load_adapter(module_path: str, class_name: str):
    """Import and instantiate an adapter. Returns (instance, None) or (None, error_str)."""
    import importlib
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(), None
    except ImportError as e:
        return None, f"SKIP (import error: {e})"
    except Exception as e:
        return None, f"SKIP (error: {e})"


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    col_widths = {
        "name":    36,
        "comp":    10,
        "recall":  10,
        "prec":    10,
        "compl":   12,
        "temp":    10,
        "queries": 9,
        "time":    10,
    }

    def row(name, comp, recall, prec, compl, temp, n_q, t_q):
        return (
            f"  {name:<{col_widths['name']}}"
            f"  {comp:<{col_widths['comp']}}"
            f"  {recall:<{col_widths['recall']}}"
            f"  {prec:<{col_widths['prec']}}"
            f"  {compl:<{col_widths['compl']}}"
            f"  {temp:<{col_widths['temp']}}"
            f"  {n_q:<{col_widths['queries']}}"
            f"  {t_q}"
        )

    header = row("System", "Composite", "Recall", "Precision", "Ans.Compl.", "Temporal", "Queries", "Q.Time(s)")
    divider = "  " + "-" * (sum(col_widths.values()) + 2 * len(col_widths) + 8)

    print(f"\n{'='*80}")
    print(f"  AGENTIC MEMORY BENCHMARK — COMPARISON RESULTS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(header)
    print(divider)

    # Sort by composite score descending
    sorted_results = sorted(results, key=lambda r: r.get("composite", 0), reverse=True)

    for r in sorted_results:
        status = ""
        if r.get("skipped"):
            print(f"  {r['name']:<{col_widths['name']}}  SKIPPED — {r.get('skip_reason', '')}")
            continue

        overall = r.get("overall", {})
        comp   = f"{overall.get('composite', 0):.2f}/10"
        recall = f"{overall.get('recall', 0):.1%}"
        prec   = f"{overall.get('precision', 0):.1%}"
        compl  = f"{overall.get('answer_completeness', 0):.1%}"
        temp   = f"{overall.get('temporal_accuracy', 0):.1%}"
        n_q    = str(r.get("n_queries", "?"))
        t_q    = f"{r.get('total_query_time_s', 0):.2f}s"

        print(row(r["name"], comp, recall, prec, compl, temp, n_q, t_q))

    print(divider)
    print()

    # Delta table: show gain/loss vs naive baseline
    baseline = next((r for r in results if "Naive" in r.get("name", "") and not r.get("skipped")), None)
    our_system = next((r for r in results if "archon-memory-core" in r.get("name", "") and not r.get("skipped")), None)

    if baseline and our_system:
        b_comp = baseline["overall"].get("composite", 0)
        our_comp = our_system["overall"].get("composite", 0)
        delta = our_comp - b_comp
        sign = "+" if delta >= 0 else ""
        print(f"  archon-memory-core vs naive baseline: {sign}{delta:.2f} pts composite")

    oracle = next((r for r in results if "oracle" in r.get("name", "").lower() and not r.get("skipped")), None)
    if oracle and our_system:
        o_comp = oracle["overall"].get("composite", 0)
        our_comp = our_system["overall"].get("composite", 0)
        gap = o_comp - our_comp
        pct = (our_comp / o_comp * 100) if o_comp > 0 else 0
        print(f"  archon-memory-core vs oracle ceiling:  {our_comp:.2f}/{o_comp:.2f} ({pct:.0f}% of ceiling)")

    print()


def run_all(scenario_filter: str | None = None, verbose: bool = False) -> list[dict]:
    """Run all adapters and return list of result dicts."""
    scenarios = load_scenarios(scenario_filter)
    print(f"\nLoaded {len(scenarios)} scenario(s), "
          f"{sum(len(s.get('queries', [])) for s in scenarios)} queries total.")
    print(f"Running {len(ADAPTERS)} adapters...\n")

    all_results = []

    for display_name, module_path, class_name in ADAPTERS:
        print(f"  [{display_name}]", flush=True)
        adapter, err = load_adapter(module_path, class_name)

        if adapter is None:
            print(f"    {err}\n")
            all_results.append({
                "name":        display_name,
                "skipped":     True,
                "skip_reason": err,
            })
            continue

        report = run_benchmark(
            adapter=adapter,
            scenarios=scenarios,
            k=5,
            verbose=verbose,
        )

        # Compute total query time across scenarios
        total_query_time = sum(
            r.get("query_time_s", 0)
            for r in report.get("scenario_results", [])
        )

        result = {
            "name":               display_name,
            "skipped":            False,
            "adapter_class":      class_name,
            "run_id":             report["run_id"],
            "n_scenarios":        report["n_scenarios"],
            "n_queries":          report["n_queries"],
            "overall":            report["overall"],
            "trap_avg":           report.get("trap_avg", 0),
            "nontrap_avg":        report.get("nontrap_avg", 0),
            "composite":          report["overall"].get("composite", 0),
            "total_query_time_s": round(total_query_time, 3),
            "by_reasoning_type":  report.get("overall_by_reasoning_type", {}),
            "scenario_results":   report.get("scenario_results", []),
        }
        all_results.append(result)
        print(f"    composite={result['composite']:.2f}/10  "
              f"recall={result['overall'].get('recall', 0):.1%}  "
              f"queries={result['n_queries']}\n")

    return all_results


def save_comparison(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_adapters":   len(results),
        "results":      results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AMB against all adapters and print a comparison table.",
    )
    parser.add_argument("--scenario", default=None, help="Filter to one scenario by name fragment.")
    parser.add_argument("--verbose", action="store_true", help="Show per-query output for each adapter.")
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "comparison.json"),
        help="Output path for comparison JSON. Default: benchmark/results/comparison.json",
    )
    args = parser.parse_args()

    results = run_all(scenario_filter=args.scenario, verbose=args.verbose)

    print_comparison_table(results)

    output_path = Path(args.output)
    save_comparison(results, output_path)
    print(f"  Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
