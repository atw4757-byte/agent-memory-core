# Reproducing the AMB v2.3 numbers

Everything behind the Show HN chart is in this repo. Seeds are fixed
(42/43/44), scenarios are deterministic from seed, and the adapter
registry is frozen. If you follow the commands below you will get the
same numbers we cite.

If you get different numbers, open an issue with your seed set and the
output `*.json` files — we want to know.

## Setup (one-time)

```bash
git clone https://github.com/atw4757-byte/agent-memory-core
cd agent-memory-core
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e . -r benchmark/amb_v2/requirements.txt
```

## Smoke (mini-v23, ~2 min)

Use this to confirm your environment before running the full grid.
Mini is too small to differentiate adapters cleanly (see PREREGISTERED.md
§5) — it exists for smoke, not for headline claims.

```bash
python -m benchmark.amb_v2.run_all \
  --scenarios benchmark/amb_v2/scenarios-v23/mini \
  --out-dir /tmp/amb-mini \
  --adapters agent-memory-core,naive,langchain-dump \
  --modes stock,tuned \
  --seeds 42,43,44 \
  --noise-rates 0.45 \
  --days 90 \
  --checkpoints 0,7,14,30,60,90
```

## Large-v23 (the 99.2% / 0% chart, ~20 min on a laptop)

This is the grid behind the Show HN chart. 250 queries × 2,300
confusers × 3 seeds × 3 adapters × 2 modes.

```bash
python -m benchmark.amb_v2.run_all \
  --scenarios benchmark/amb_v2/scenarios-v23/large \
  --out-dir /tmp/amb-large \
  --adapters agent-memory-core,naive,langchain-dump \
  --modes stock,tuned \
  --seeds 42,43,44 \
  --noise-rates 0.45 \
  --days 90 \
  --checkpoints 0,7,14,30,60,90
```

Expected at day 90, averaged over seeds 42/43/44:

| Adapter              | Mode  | top-1 @90 |
| -------------------- | ----- | --------- |
| agent-memory-core    | tuned | 0.992     |
| agent-memory-core    | stock | 0.492     |
| langchain-dump       | tuned | 0.000     |
| langchain-dump       | stock | 0.000     |
| naive-append-only    | tuned | 0.000     |
| naive-append-only    | stock | 0.000     |

Per-seed and per-checkpoint numbers are written as JSON to
`--out-dir`. Each file name is
`{adapter}__{mode}__seed{N}__noise{R}.json`.

## Render the chart

```bash
python -m benchmark.amb_v2.chart \
  --results /tmp/amb-large \
  --out-dir /tmp/amb-large/charts
```

## What "deterministic from seed" means (and doesn't)

- **Scenarios, queries, confusers:** fully deterministic — same seed →
  same timeline bytes.
- **Embedding model:** ChromaDB's default ONNX build of
  `all-MiniLM-L6-v2` (see
  `chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2`). The
  cross-encoder reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  Different hardware (ARM vs x86) can produce embedding deltas at the
  4th decimal, which rarely but occasionally flips a top-1 tie. If
  your top-1 differs from ours by more than a seed's worth of
  variance, please open an issue with your `torch.__version__` and
  CPU arch.
- **LLM scoring:** none of the headline numbers use LLM-as-judge.
  `top-1 answer accuracy` is exact-string match against
  `expected_answer` after normalization (see
  `benchmark/amb_v2/metrics.py::top1_answer_accuracy`).

## Scenarios and preregistration

- Frozen methodology: [`benchmark/amb_v2/PREREGISTERED.md`](benchmark/amb_v2/PREREGISTERED.md)
- Composite quality formula: `PREREGISTERED.md` §2
- Held-out (Gemini-authored) scenarios: `benchmark/amb_v2/scenarios-v23/*.json.age`
  — encrypted; key is held by the maintainer and released after a
  leaderboard submission window closes.

## Adapters

Three public adapters ship with the benchmark:

| Adapter              | Module                                       |
| -------------------- | -------------------------------------------- |
| agent-memory-core    | `benchmark.amb_v2.adapters.amc`              |
| naive-append-only    | `benchmark.amb_v2.adapters.naive`            |
| langchain-dump       | `benchmark.amb_v2.adapters.langchain_dump`   |

Adding a new adapter: see `benchmark/amb_v2/README.md` §"Adding a new
adapter".
