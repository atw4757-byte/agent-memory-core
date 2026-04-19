"""Grid runner — sweeps (adapter × mode × seed × noise_rate) and writes results.

Filenames are deterministic; existing results are reused (idempotent). Use
``--quick`` for a smoke grid (1 adapter, 1 seed, 1 noise, stock-only) that
completes in <2 min on the mini scenario.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.amb_v2.adapters.base import Mode
from benchmark.amb_v2.harness import DEFAULT_CHECKPOINTS, run_one
from benchmark.amb_v2.run import ADAPTER_REGISTRY
from benchmark.amb_v2.scenarios import load_public_scenarios, load_held_out

DEFAULT_NOISE_RATES = (0.20, 0.30, 0.45, 0.60)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_MODES: tuple[Mode, ...] = ("stock", "tuned")


@dataclass
class GridSpec:
    adapters: list[str]
    modes: list[Mode]
    seeds: list[int]
    noise_rates: list[float]
    scenarios_dir: Path
    out_dir: Path
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS
    days: int | None = None
    held_out_key: Path | None = None
    held_out_dir: Path | None = None
    filler_facts_per_day: int = 0


def _result_path(out_dir: Path, adapter: str, mode: str, seed: int, noise_rate: float) -> Path:
    noise_tag = f"n{int(round(noise_rate * 100)):02d}"
    return out_dir / f"{adapter}__{mode}__s{seed}__{noise_tag}.json"


def run_grid(spec: GridSpec) -> list[Path]:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    bundles = list(load_public_scenarios(spec.scenarios_dir))
    if spec.held_out_key is not None:
        held_dir = spec.held_out_dir or spec.scenarios_dir
        bundles.extend(load_held_out(held_dir, key_path=spec.held_out_key))
    if not bundles:
        raise RuntimeError(f"no scenarios loaded from {spec.scenarios_dir}")

    paths: list[Path] = []
    for adapter_name in spec.adapters:
        cls = ADAPTER_REGISTRY[adapter_name]
        for mode in spec.modes:
            for seed in spec.seeds:
                for nr in spec.noise_rates:
                    out = _result_path(spec.out_dir, adapter_name, mode, seed, nr)
                    paths.append(out)
                    if out.exists():
                        continue
                    adapter = cls(mode=mode)
                    result = run_one(
                        adapter=adapter, scenarios=bundles, seed=seed,
                        noise_rate=nr, mode=mode,
                        checkpoints=spec.checkpoints, days=spec.days,
                        filler_facts_per_day=spec.filler_facts_per_day,
                    )
                    out.write_text(json.dumps(result, indent=2))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amb-v2-grid",
        description="AMB v2 grid runner (adapter × mode × seed × noise_rate).")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adapters", type=str,
                        default=",".join(sorted(ADAPTER_REGISTRY.keys())))
    parser.add_argument("--modes", type=str, default="stock,tuned")
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--noise-rates", type=str,
                        default=",".join(str(n) for n in DEFAULT_NOISE_RATES))
    parser.add_argument("--checkpoints", type=str,
                        default=",".join(str(d) for d in DEFAULT_CHECKPOINTS))
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--held-out-key", type=Path, default=None)
    parser.add_argument("--held-out-dir", type=Path, default=None,
                        help="Directory with *.json.age held-out scenarios (defaults to --scenarios)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke grid: 1 adapter, 1 seed, 1 noise, stock-only, short checkpoints")
    parser.add_argument("--filler-facts-per-day", type=int, default=0,
                        help="v2.2 scale knob: inject K filler facts per simulated day "
                             "to stress retrieval-pool density. Default 0 = v2.1 parity.")
    args = parser.parse_args(argv)

    if args.quick:
        spec = GridSpec(
            adapters=["naive"],
            modes=["stock"],
            seeds=[42],
            noise_rates=[0.30],
            checkpoints=(0, 3, 7),
            scenarios_dir=args.scenarios,
            out_dir=args.out_dir,
        )
    else:
        spec = GridSpec(
            adapters=[a.strip() for a in args.adapters.split(",") if a.strip()],
            modes=[m.strip() for m in args.modes.split(",") if m.strip()],  # type: ignore[misc]
            seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
            noise_rates=[float(n) for n in args.noise_rates.split(",") if n.strip()],
            checkpoints=tuple(sorted({int(c) for c in args.checkpoints.split(",") if c.strip()})),
            days=args.days,
            scenarios_dir=args.scenarios,
            out_dir=args.out_dir,
            held_out_key=args.held_out_key,
            held_out_dir=args.held_out_dir,
            filler_facts_per_day=args.filler_facts_per_day,
        )

    paths = run_grid(spec)
    print(f"[amb-v2-grid] wrote {len(paths)} result files to {spec.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
