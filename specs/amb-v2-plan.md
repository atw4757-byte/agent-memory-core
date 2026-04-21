---
title: "AMB v2 — Phase 2 Plan"
status: Phase 2 DRAFT
created: 2026-04-18
spec: specs/amb-v2-spec.md (commit 5079dcc, post-C1)
adversary_review: Memory/adversary-reviews/2026-04-18_amb-v2-adversary.md
next_phase: Phase 3 Tasks (TDD)
---

## What this doc is

The bridge between the Phase 1 spec (what we're building and why) and the Phase 3 task list (the TDD-ordered atomic units of work). This doc names every file, defines every contract, and budgets the work day-by-day. If a reader can't pick this up cold and start writing tests for the simulator, this doc has failed.

## Build directive

Land everything under `benchmark/amb_v2/` — a sibling of the existing `benchmark/` v1 layout, not a replacement. v1 stays unchanged; v2 is additive. After v2 is alpha-stable, `benchmark/run_benchmark.py` (v1) gains a deprecation notice.

```
benchmark/amb_v2/
├── __init__.py
├── PREREGISTERED.md          # Pre-registration artifact (D8). Frozen pre-run.
├── README.md                 # How to run + caveats (incl. v2.1 commitment)
├── requirements.txt          # Pinned adapter deps (D dependencies section)
│
├── chunks.py                 # Chunk dataclass (D data model)
├── queries.py                # Query dataclass + loader for v2 query set
├── scenarios.py              # Scenario loader (handles public + held-out encrypted)
│
├── simulator.py              # D6 — pure simulator, (scenario, seed) → events
├── metrics.py                # D4 — 4 sub-metrics + composite + AUC
├── harness.py                # D7 — main loop (renamed from "run.py" in spec — clearer)
├── run.py                    # CLI entry point: --adapter --seed --noise-rate --mode
├── run_all.py                # Convenience: full grid (4 adapters × 2 modes × 4 noise rates × N seeds)
│
├── adapters/
│   ├── __init__.py
│   ├── base.py               # DecayAdapter Protocol (D3)
│   ├── archon_memory_core.py  # Our adapter (wraps existing v1 adapter)
│   ├── langchain.py          # ConversationSummaryBufferMemory wrapper
│   ├── llamaindex.py         # ChatMemoryBuffer + simple semantic recall
│   └── mem0.py               # Mem0 wrapper (NEW for v2 — wasn't in v1)
│
├── chart.py                  # D13 — decay curves + sensitivity grid
│
├── scenarios/                # 7 public, temporally extended from v1
│   ├── 01_personal_assistant.json
│   ├── 02_executive_cos.json
│   ├── 03_customer_support.json
│   ├── 05_software_pm.json
│   ├── 06_sales_crm.json
│   ├── 08_tutor.json
│   └── 10_research_assistant.json
│
├── held_out/                 # 3 Cipher-authored, encrypted at rest
│   ├── README.md             # How they were generated, exact prompt, when decrypted
│   ├── h01.json.age          # age-encrypted
│   ├── h02.json.age
│   └── h03.json.age
│
├── results/                  # Output JSONs from runs, .gitignored except headline
│   └── headline-v2.0/        # The pre-registered alpha results commit
│
└── tests/
    ├── __init__.py
    ├── conftest.py           # fixtures: mini scenario, fixed seed
    ├── test_chunks.py
    ├── test_queries.py
    ├── test_scenarios.py
    ├── test_simulator.py     # Layer 1
    ├── test_metrics.py       # Layer 2
    ├── test_harness.py       # Layer 3
    ├── test_adapter_base.py  # Layer 4 — protocol tests
    ├── test_adapter_smoke.py # Layer 4 — per-adapter smoke
    ├── test_regression.py    # Layer 5 — golden output
    ├── test_chart.py         # smoke test on chart generation
    ├── fixtures/
    │   ├── mini_scenario.json   # 5-day scenario for fast tests
    │   └── golden_results.json  # expected output of mini_scenario @ seed=42
    └── test_e2e_mini.py      # full pipeline on mini scenario
```

## File-by-file specifications

### `chunks.py` (~30 LOC)
```python
from dataclasses import dataclass
from typing import Literal

ChunkType = Literal["fact","update","noise","credential","preference","session"]

@dataclass(frozen=True)
class Chunk:
    id: str
    scenario_id: str
    day: int
    text: str
    type: ChunkType
    supersedes: str | None = None
```
**Tests:** `test_chunks.py` — frozen-ness, type validation via Literal, `supersedes` defaults None.

### `queries.py` (~50 LOC)
```python
from dataclasses import dataclass, field
from typing import Literal

ResolutionType = Literal["stable","contradiction","aggregation","trajectory"]

@dataclass
class Query:
    query_id: str
    scenario_id: str
    question: str
    expected_answer: str
    reasoning_type: str
    difficulty: Literal["easy","medium","hard"]
    trap: str | None
    checkpoint_eligibility: frozenset[int]
    resolution_type: ResolutionType

def load_queries(scenario_path: str) -> list[Query]: ...
```
**Tests:** `test_queries.py` — JSON round-trip, `checkpoint_eligibility` parses set, missing fields error.

### `scenarios.py` (~80 LOC)
```python
def load_public_scenarios(dir: Path) -> list[ScenarioBundle]: ...
def load_held_out(dir: Path, key: bytes) -> list[ScenarioBundle]: ...

@dataclass
class ScenarioBundle:
    scenario_id: str
    name: str
    timeline: list[tuple[int, str, str]]  # (day, event_type, payload)
    queries: list[Query]
    is_held_out: bool
```
- Public scenarios are JSON.
- Held-out scenarios are `age`-encrypted JSON. Decryption key sourced from `~/.archon/amb-heldout.key`. Without the key, held-out scenarios silently skip (so external contributors can run public-only).
**Tests:** `test_scenarios.py` — public load, held-out decrypt round-trip with test key, missing-key path.

### `simulator.py` (~150 LOC) — **D6**
Generates the deterministic `(day, chunk)` event stream from a scenario.

```python
def simulate(
    scenarios: list[ScenarioBundle],
    seed: int,
    noise_rate: float = 0.45,
    days: int = 90,
) -> Iterator[tuple[int, list[Chunk]]]: ...
```
- Pure: same `(scenarios, seed, noise_rate, days)` → byte-identical event stream.
- Mixes scenario timeline events with synthetic noise chunks at the configured rate.
- Honors the soft 200-chunks/day cap with a clipping warning, not a hard error.
- Returns events grouped by day (so harness ingests per-day).

**Tests:** Layer 1 (per spec). Plus:
- `test_simulator_noise_rate_calibrated` — actual emitted distribution within ±2% of target across {20, 30, 45, 60}%.
- `test_simulator_handles_empty_day` — yields `(day, [])` for no-op days.
- `test_simulator_chunks_have_unique_ids`.

### `metrics.py` (~150 LOC) — **D4**
Pure functions over query results.

```python
def answer_accuracy(results: list[QueryResult]) -> float: ...
def contradiction_resolution(results: list[QueryResult]) -> float: ...
def stale_fact_rate(results: list[QueryResult]) -> float: ...
def salience_preservation(results: list[QueryResult]) -> float: ...

def quality_at(results: list[QueryResult]) -> float:
    # Pre-registered: 0.40·answer + 0.30·contradiction + 0.15·(1-stale) + 0.15·salience
    ...

def auc_quality(checkpoints: list[Checkpoint]) -> float:
    # Trapezoid integration over (day, quality) pairs
    ...
```
**Tests:** Layer 2 (per spec). Frozen-formula test verifies the constants didn't drift.

### `adapters/base.py` (~40 LOC) — **D3**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DecayAdapter(Protocol):
    def __init__(self, mode: Literal["stock","tuned"]) -> None: ...
    def ingest(self, day: int, chunks: list[Chunk]) -> None: ...
    def consolidate(self, day: int) -> None: ...   # noop in stock mode
    def query(self, question: str, scenario_id: str) -> str: ...

    @property
    def metadata(self) -> dict:
        # {"name": str, "version": str, "implements_consolidation": bool}
        ...
```
- `mode="stock"` MUST cause `consolidate()` to be a no-op.
- `mode="tuned"` MAY do work in `consolidate()`. Adapters that can't (e.g. naive vector) just return.
- `metadata.implements_consolidation` is the truthful self-declaration. Lying disqualifies leaderboard runs.

**Tests:** `test_adapter_base.py` — every adapter satisfies the Protocol; `mode="stock"` is a no-op for consolidate; metadata schema valid.

### `adapters/archon_memory_core.py` (~100 LOC)
Wraps the existing v1 adapter (`benchmark/adapters/archon_memory_core_adapter.py`). In `tuned` mode, calls into the existing consolidation API. In `stock` mode, skips consolidation but otherwise functions normally.

### `adapters/langchain.py` (~80 LOC)
Wraps `ConversationSummaryBufferMemory` from langchain_community. `tuned` mode triggers `prune()` once per day. `stock` mode uses default behavior. `implements_consolidation: false`.

### `adapters/llamaindex.py` (~80 LOC) — NEW
Wraps `ChatMemoryBuffer` + `VectorStoreIndex` for semantic recall over conversation history. `tuned` mode rebuilds the vector index nightly. `stock` mode appends only.

### `adapters/mem0.py` (~80 LOC) — NEW
Wraps the `mem0` library. `tuned` mode triggers Mem0's built-in update/consolidation flow. `stock` mode disables Mem0's auto-update. `implements_consolidation: true`.

### `harness.py` (~120 LOC) — **D7**
The main loop. Pure orchestration; no business logic.

```python
def run_one(
    adapter: DecayAdapter,
    scenarios: list[ScenarioBundle],
    seed: int,
    noise_rate: float,
    mode: Literal["stock","tuned"],
    checkpoints: list[int] = [0, 7, 14, 30, 60, 90],
) -> RunResult: ...
```
**Tests:** Layer 3 (per spec). Plus:
- `test_harness_invokes_consolidate_only_in_tuned_mode`.
- `test_harness_records_metadata_in_results`.

### `run.py` (~80 LOC)
CLI entry. Invokes `run_one` once with parsed args. Writes results JSON.
```
python -m benchmark.amb_v2.run \
    --adapter archon-memory-core --seed 42 \
    --noise-rate 0.30 --mode tuned
```

### `run_all.py` (~60 LOC)
Iterates the full grid (4 adapters × 2 modes × 4 noise rates × M seeds). Writes everything to `results/`. Parallelized with `concurrent.futures`. Total target ≤ 8 hours wall on Bosgame.

### `chart.py` (~150 LOC) — **D13**
Reads `results/*.json`, produces:
- `decay-curves.svg` (headline noise=30%, two lines per adapter)
- `decay-curves.png`
- `decay-table.md` (full grid)
- `sensitivity-grid.svg` (4 noise rates as small multiples)

**Tests:** `test_chart.py` — smoke: produces non-empty SVG given fixture JSON.

### `PREREGISTERED.md`
Hand-written doc, frozen pre-run. Contains:
- Methodology summary (extracted from spec)
- Composite formula and weights, verbatim
- SHA-256 of `simulator.py`, `metrics.py`, `harness.py`, scenario files, held-out encrypted blobs
- Pre-registered Quality@90 predictions per adapter per mode (8 numbers) ±0.10
- Cipher held-out generation prompt, verbatim
- Author signoff timestamp
- Commit pin (the run uses HEAD of this commit, no later)

## Day-by-day budget

5 focused days. Each day has a deliverable; commit at the end.

### Day 1 — Foundations (Phase 3 begin)
- Tests + impl: `chunks.py`, `queries.py`, `scenarios.py`
- Test fixtures: `mini_scenario.json`
- Commit C2: data model + scenario loader landed.
- Acceptance: `pytest tests/test_chunks.py tests/test_queries.py tests/test_scenarios.py` green.

### Day 2 — Simulator + Metrics
- Tests + impl: `simulator.py`, `metrics.py`
- All Layer 1 + Layer 2 tests pass.
- Commit C3.
- Acceptance: simulator determinism property test green over 10 random seeds; composite formula frozen-test green.

### Day 3 — Adapters + Harness
- Tests + impl: `adapters/base.py`, all 4 adapter wrappers, `harness.py`
- `requirements.txt` pinned to working versions.
- Commit C4.
- Acceptance: `pytest tests/test_adapter_*.py tests/test_harness.py` green; Layer 4 smoke per adapter green.

### Day 4 — End-to-end + Chart + Pre-registration
- Tests + impl: `run.py`, `run_all.py`, `chart.py`, `test_e2e_mini.py`, `test_regression.py` (golden output)
- Cipher generates held-out scenarios (separate driver script, see below)
- Encrypt held-out, commit ciphertext + hashes
- Write `PREREGISTERED.md` with all hashes + predictions
- Commit C5: alpha-ready harness + pre-registration.
- Acceptance: `python -m benchmark.amb_v2.run_all --quick` (mini scenario, single seed) completes in <2 min and produces valid results JSON + chart.

### Day 5 — Phase 5 (Cipher review) + Phase 6 (alpha run)
- `archon-adversary --role reviewer` over the implementation
- Address legitimate review items
- Run full grid on Bosgame (4 × 2 × 4 × 1 seed = 32 runs, ~8 hours)
- Generate alpha charts
- Commit results to `results/headline-v2.0/`
- Update `ROADMAP.md` + `README.md` + main project page with v2.0 alpha link
- Tag `v0.2.0-amb-v2-alpha`, cut GitHub Release.

## Held-out scenario generation (Day 4 sub-task)

Drive Cipher via a deterministic prompt, captured verbatim into `PREREGISTERED.md`. Approximate shape:

> **System:** You are an external benchmark author. You have read the public AMB v2 spec at `<URL>` and the public scenarios at `<URL>`. You have NOT seen the archon-memory-core source code. Generate a scenario in the same JSON format as the public scenarios, with these constraints: 90-day timeline, 5–8 sessions, includes at least 3 contradictions (updates), at least 2 long-tail credential queries, and at least 1 trajectory query. Adversarially target memory failure modes you would expect a stock LangChain or LlamaIndex system to fail at while a thoughtful long-term memory system would handle.

Run 3 times with 3 different scenario themes (e.g. "small business owner", "long-running research project", "elderly parent caregiving"). Save outputs, encrypt with `age`, commit ciphertext + decrypt-key into `~/.archon/amb-heldout.key`.

## Dependencies + risks for Phase 2

- **mem0 adapter is new.** v1 didn't have it. Add to `requirements.txt`, smoke-test on mini scenario before committing the adapter.
- **age encryption tooling.** macOS Homebrew has `age`. Add note to README. CI doesn't need to decrypt; only the run-host does.
- **Held-out generation depends on Cipher availability.** If Cipher Agent API is down on Day 4, fall back to direct Gemini API call (key in `~/.archon/credentials/gemini.key`).
- **Bosgame compute window for Day 5 alpha run.** Must coordinate so we're not stomping on the divergence pipelines. Run during Andy's 7-hr work day so pipelines can pause if needed.

## Phase 2 acceptance

- [ ] This plan committed and pushed
- [ ] Andy's review pass (look for missed dependencies, missed adapters, scope creep)
- [ ] Phase 3 task list (TDD ordered) generated from this plan
