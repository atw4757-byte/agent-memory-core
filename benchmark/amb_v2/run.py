"""CLI entry point for a single AMB v2 run.

Usage:
    python -m benchmark.amb_v2.run \
        --adapter naive \
        --mode stock \
        --seed 42 \
        --noise-rate 0.30 \
        --scenarios path/to/scenarios \
        --out result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.amb_v2.adapters.base import Mode
from benchmark.amb_v2.adapters.naive import NaiveAppendOnlyAdapter
from benchmark.amb_v2.harness import DEFAULT_CHECKPOINTS, run_one
from benchmark.amb_v2.scenarios import load_public_scenarios, load_held_out

ADAPTER_REGISTRY: dict[str, Any] = {"naive": NaiveAppendOnlyAdapter}

try:
    from benchmark.amb_v2.adapters import agent_memory_core as _amc
    if _amc._AVAILABLE:
        ADAPTER_REGISTRY["agent-memory-core"] = _amc.AgentMemoryCoreAdapter
except ImportError:
    pass

try:
    from benchmark.amb_v2.adapters import langchain_adapter as _lc
    if _lc._AVAILABLE:
        ADAPTER_REGISTRY["langchain"] = _lc.LangChainAdapter
except ImportError:
    pass


def _parse_checkpoints(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amb-v2", description="AMB v2 single run.")
    p.add_argument("--adapter", required=True, choices=sorted(ADAPTER_REGISTRY.keys()))
    p.add_argument("--mode", required=True, choices=["stock", "tuned"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--noise-rate", type=float, required=True)
    p.add_argument("--scenarios", type=Path, required=True,
                   help="Directory with public scenario .json files")
    p.add_argument("--held-out-key", type=Path, default=None,
                   help="Path to age identity file for decrypting held-out scenarios")
    p.add_argument("--checkpoints", type=str,
                   default=",".join(str(d) for d in DEFAULT_CHECKPOINTS))
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cls = ADAPTER_REGISTRY[args.adapter]
    adapter = cls(mode=args.mode)

    bundles = list(load_public_scenarios(args.scenarios))
    if args.held_out_key is not None:
        bundles.extend(load_held_out(args.scenarios, args.held_out_key))

    if not bundles:
        print(f"[amb-v2] no scenarios loaded from {args.scenarios}", file=sys.stderr)
        return 2

    cps = _parse_checkpoints(args.checkpoints)
    result = run_one(
        adapter=adapter,
        scenarios=bundles,
        seed=args.seed,
        noise_rate=args.noise_rate,
        mode=args.mode,
        checkpoints=cps,
        days=args.days,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[amb-v2] wrote {args.out} (auc={result['auc_quality']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
