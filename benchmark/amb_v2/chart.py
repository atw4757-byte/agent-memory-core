"""Chart generation — decay curves + sensitivity grid.

Inputs are directories of result JSON files produced by :mod:`run_all`.
Outputs are SVG/PNG files suitable for a README or landing page.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _group_by(results: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def render_decay_curves(results_dir: Path, out_path: Path,
                         noise_rate: float = 0.30) -> None:
    """One line per (adapter, mode) at a fixed noise rate; x=day, y=quality."""
    results = [r for r in _load_results(results_dir)
               if abs(r.get("noise_rate", -1) - noise_rate) < 1e-6]
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = _group_by(results, "adapter", "mode")
    for (adapter, mode), runs in sorted(groups.items()):
        # Average across seeds per day
        by_day: dict[int, list[float]] = defaultdict(list)
        for r in runs:
            for cp in r["checkpoints"]:
                by_day[cp["day"]].append(cp["quality"])
        days = sorted(by_day)
        qualities = [sum(by_day[d]) / len(by_day[d]) for d in days]
        style = "-" if mode == "tuned" else "--"
        ax.plot(days, qualities, style, marker="o", label=f"{adapter} ({mode})")
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Quality")
    ax.set_ylim(0, 1)
    ax.set_title(f"AMB v2 — decay curves (noise_rate={noise_rate:.2f})")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_sensitivity_grid(results_dir: Path, out_path: Path) -> None:
    """Grid of AUC-quality vs noise_rate, one panel per mode, one line per adapter."""
    results = _load_results(results_dir)
    modes = sorted({r["mode"] for r in results})
    if not modes:
        raise RuntimeError(f"no result files in {results_dir}")
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    groups = _group_by(results, "adapter", "mode", "noise_rate")
    for ax, mode in zip(axes, modes):
        per_adapter: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (adapter, m, nr), runs in groups.items():
            if m != mode:
                continue
            aucs = [r["auc_quality"] for r in runs]
            per_adapter[adapter].append((nr, sum(aucs) / len(aucs)))
        for adapter, pts in sorted(per_adapter.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=adapter)
        ax.set_xlabel("Noise rate")
        ax.set_title(f"mode = {mode}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("AUC quality")
    fig.suptitle("AMB v2 — sensitivity across noise rates")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="amb-v2-chart")
    p.add_argument("--results", type=Path, required=True,
                   help="Directory with result .json files")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--noise-rate", type=float, default=0.30,
                   help="Noise rate to plot decay curves at (default 0.30)")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_decay_curves(args.results, args.out_dir / "decay-curves.svg",
                         noise_rate=args.noise_rate)
    render_sensitivity_grid(args.results, args.out_dir / "sensitivity-grid.svg")
    print(f"[amb-v2-chart] wrote charts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
