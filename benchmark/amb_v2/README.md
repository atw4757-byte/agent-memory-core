# AMB v2 — Agentic Memory Benchmark v2 (alpha)

Rigorous, pre-registered, reproducible benchmark for long-horizon memory
systems. v2 fixes the fairness issues in v1 with:

- **D10 Stock vs Tuned mode** — every adapter runs in both modes.
  No single-number ranking.
- **D11 Sensitivity** — 4 noise rates swept per adapter; an ordinal
  claim must hold across ≥3 of 4 to be reported.
- **D12 Real-data track** (v2.1) — before any production recommendation.
- **Pre-registration** with SHA-256-pinned implementation files.
- **Held-out scenarios** authored by an independent LLM (Gemini 2.5 Pro),
  committed encrypted with `age`.

See [`PREREGISTERED.md`](PREREGISTERED.md) for the frozen methodology
and pre-published predictions.

## Status

- **v2.0-alpha** (2026-04-18): 3 adapters (naive, agent-memory-core,
  langchain-buffer) on the `mini` test scenario.
  - [Results report](results/alpha-v2.0/REPORT.md) — **null result**;
    mini is too small to differentiate adapters.
- **v2.0.1** (planned): 3 Cipher-generated held-out scenarios,
  LlamaIndex + Mem0 adapters.
- **v2.1** (planned): real-data validation track.

## Quick start

```bash
# One-time setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e . -r benchmark/amb_v2/requirements.txt

# Smoke grid (<1 min)
python -m benchmark.amb_v2.run_all --quick \
    --scenarios benchmark/amb_v2/tests/fixtures \
    --out-dir /tmp/amb2-quick

# Full alpha grid (~40s on a mid-2020s laptop)
python -m benchmark.amb_v2.run_all \
    --scenarios benchmark/amb_v2/scenarios \
    --out-dir benchmark/amb_v2/results/alpha-v2.0 \
    --adapters naive,agent-memory-core,langchain \
    --modes stock,tuned \
    --seeds 42,43,44 \
    --noise-rates 0.20,0.30,0.45,0.60

# Render charts
python -m benchmark.amb_v2.chart \
    --results benchmark/amb_v2/results/alpha-v2.0 \
    --out-dir benchmark/amb_v2/results/alpha-v2.0/charts
```

## Composite quality formula (FROZEN)

```
quality = 0.40 · answer_accuracy
        + 0.30 · contradiction_resolution
        + 0.15 · (1 − stale_fact_rate)
        + 0.15 · salience_preservation
```

See [`PREREGISTERED.md`](PREREGISTERED.md) §2 for exact definitions.

## Adding a new adapter

1. Subclass the informal `DecayAdapter` protocol in
   [`adapters/base.py`](adapters/base.py). Required methods:
   `ingest(day, chunks)`, `consolidate(day)`, `query(question,
   scenario_id) -> str`, plus a `metadata` property with `name`,
   `version`, and a truthful `implements_consolidation: bool`.
2. Register it in [`run.py`](run.py)'s `ADAPTER_REGISTRY`.
3. The `tests/test_adapter_smoke.py` framework will auto-pick it up if
   its `_AVAILABLE` flag is `True`.

## Layout

```
benchmark/amb_v2/
├── PREREGISTERED.md        # frozen methodology + predictions
├── README.md               # this file
├── requirements.txt        # pinned deps
├── chunks.py queries.py    # data model
├── scenarios.py            # public + held-out (age-encrypted) loaders
├── simulator.py            # noise-calibrated event stream
├── metrics.py              # frozen formula + AUC
├── harness.py              # run_one() orchestration
├── run.py                  # single-run CLI
├── run_all.py              # grid sweep
├── chart.py                # decay curves + sensitivity grid
├── adapters/               # naive, agent-memory-core, langchain
├── scripts/                # held-out generation driver
├── scenarios/              # public scenarios
├── held_out/               # *.age ciphertext only
├── results/                # result JSONs (alpha-v2.0/ committed)
└── tests/                  # 72 green, 91% coverage
```
