---
title: "AMB v2 — Phase 3 Tasks (TDD)"
status: Phase 3 DRAFT
created: 2026-04-18
spec: specs/amb-v2-spec.md (5079dcc)
plan: specs/amb-v2-plan.md (aa9091f)
next_phase: Phase 4 Implement
---

## How to read this doc

Each task is a vertical slice: one test file → one production file → green tests. Tasks are ordered so each one only depends on prior tasks. Every task ends in a commit. Commit IDs are gates — never skip ahead.

**Format per task:**
- **ID** (T-NN)
- **Deliverable** (the test file + the production file)
- **Acceptance** (green test command)
- **Commit** (the C-gate it lands behind)

## Day 1 — Foundations (commit C2)

### T-01 — Chunk dataclass
- **Test:** `tests/test_chunks.py`
  - test_chunk_is_frozen
  - test_chunk_type_literal_validates
  - test_chunk_supersedes_defaults_none
  - test_chunk_id_required
- **Impl:** `chunks.py`
- **Acceptance:** `pytest benchmark/amb_v2/tests/test_chunks.py -v`

### T-02 — Query dataclass + JSON loader
- **Test:** `tests/test_queries.py`
  - test_query_round_trip_json
  - test_checkpoint_eligibility_parses_set
  - test_resolution_type_literal_validates
  - test_load_queries_missing_file_raises
  - test_load_queries_malformed_json_raises_clear_error
- **Impl:** `queries.py`
- **Acceptance:** `pytest benchmark/amb_v2/tests/test_queries.py -v`

### T-03 — Mini scenario fixture
- **Deliverable:** `tests/fixtures/mini_scenario.json` — 5 simulated days, 1 contradiction, 1 stable fact, 1 noise event, 1 credential. Hand-authored.
- **Acceptance:** Fixture loads via `load_public_scenarios` once T-04 lands; for now, `python -c "import json; json.load(open(...))"` validates.

### T-04 — Scenario loader (public)
- **Test:** `tests/test_scenarios.py::test_load_public_scenarios`
- **Impl:** `scenarios.py` — public path only.
- **Acceptance:** Mini scenario loads cleanly into `ScenarioBundle`.

### T-05 — Scenario loader (held-out, encrypted)
- **Test:** `tests/test_scenarios.py::test_load_held_out_with_key`, `::test_load_held_out_missing_key_skips`
- **Impl:** `scenarios.py` — adds `load_held_out` using `age` via subprocess. Skips silently if key file missing.
- **Acceptance:** Test fixture: encrypt a tiny JSON with a test age key, decrypt round-trip succeeds.

### **C2 commit** — "amb-v2: data model + scenario loader (T-01..T-05)"
Acceptance: `pytest benchmark/amb_v2/tests/test_chunks.py tests/test_queries.py tests/test_scenarios.py` — all green.

## Day 2 — Simulator + Metrics (commit C3)

### T-06 — Simulator: determinism
- **Test:** `tests/test_simulator.py::test_determinism_same_seed_same_output`
- **Impl:** `simulator.py` skeleton — emits chunks from scenario timelines only (no synthetic noise yet).
- **Acceptance:** Two runs with seed=42 produce byte-identical event lists.

### T-07 — Simulator: synthetic noise injection
- **Test:** `tests/test_simulator.py::test_noise_rate_calibrated_at_30_pct`, `::test_noise_rate_calibrated_at_45_pct`
- **Impl:** Add noise generator. Noise chunks tagged `type="noise"`. Distribution within ±2% of target across full 90-day run.
- **Acceptance:** Statistical assertion over 10K-chunk run.

### T-08 — Simulator: temporal ordering + per-day grouping
- **Test:** `tests/test_simulator.py::test_chunks_yielded_in_day_order`, `::test_yields_empty_list_for_quiet_days`
- **Impl:** Group emissions by day, yield `(day, list[Chunk])`.
- **Acceptance:** No day N+1 chunk before any day N chunk; quiet days yield empty lists.

### T-09 — Simulator: chunk ID uniqueness + soft cap warning
- **Test:** `::test_chunk_ids_unique_across_run`, `::test_soft_cap_emits_warning_at_200`
- **Impl:** ID format `<scenario_short>-d<day:03d>-<seq>`. `warnings.warn` if any single day emits >200 chunks.
- **Acceptance:** No collisions; warning fires.

### T-10 — Metric: answer accuracy
- **Test:** `tests/test_metrics.py::test_answer_accuracy_all_correct_returns_1`, `::test_all_wrong_returns_0`, `::test_alias_table_honored`
- **Impl:** `metrics.py::answer_accuracy` — exact match + alias table from query metadata.
- **Acceptance:** All 3 tests green.

### T-11 — Metric: contradiction resolution
- **Test:** `::test_contradiction_resolution_filters_to_changed_queries`, `::test_returns_1_when_all_new_answers`
- **Impl:** Filter queries where `resolution_type == "contradiction"`; compute fraction returning the new answer.
- **Acceptance:** Tests green.

### T-12 — Metric: stale fact rate
- **Test:** `::test_stale_fact_rate_flags_superseded_returns`, `::test_returns_0_when_all_current`
- **Impl:** Detect when adapter answer matches the `supersedes`-pointed-to chunk's value, not the current one.
- **Acceptance:** Tests green.

### T-13 — Metric: salience preservation
- **Test:** `::test_salience_filters_to_credentials`, `::test_top_1_only`
- **Impl:** Filter `type="credential"` queries; binary score on top-1 hit.
- **Acceptance:** Tests green.

### T-14 — Composite + AUC
- **Test:** `::test_quality_formula_weights_frozen`, `::test_quality_at_edge_cases`, `::test_auc_trapezoid_known_curve`
- **Impl:** `quality_at()` with frozen 0.40/0.30/0.15/0.15. `auc_quality()` trapezoid.
- **Acceptance:** Frozen-formula test asserts `0.40, 0.30, 0.15, 0.15` literals appear in source via AST inspection.

### **C3 commit** — "amb-v2: simulator + metrics (T-06..T-14)"
Acceptance: All Layer 1 + Layer 2 tests green; coverage ≥90% on `simulator.py` and `metrics.py`.

## Day 3 — Adapters + Harness (commit C4)

### T-15 — DecayAdapter Protocol + base contract tests
- **Test:** `tests/test_adapter_base.py::test_protocol_has_required_methods`, `::test_metadata_schema`
- **Impl:** `adapters/base.py`
- **Acceptance:** Tests green.

### T-16 — Adapter smoke test framework
- **Test:** `tests/test_adapter_smoke.py` — one parametrized test that takes `adapter_factory` and runs it through mini scenario stock+tuned, asserting only that no exceptions raised and metadata is valid.
- **Impl:** Test infrastructure only.
- **Acceptance:** Test exists and is parametrized.

### T-17 — archon-memory-core adapter (stock + tuned)
- **Test:** Add `archon_memory_core` to T-16 parametrize.
- **Impl:** `adapters/archon_memory_core.py` — wraps existing v1 adapter; `consolidate()` no-ops in stock, calls v1 consolidator in tuned.
- **Acceptance:** Smoke test green for both modes.

### T-18 — langchain adapter
- **Test:** Add `langchain` to T-16 parametrize.
- **Impl:** `adapters/langchain.py` — `ConversationSummaryBufferMemory` wrapper. `tuned` mode triggers `prune()`.
- **Acceptance:** Smoke green.

### T-19 — llamaindex adapter
- **Test:** Add `llamaindex` to T-16 parametrize.
- **Impl:** `adapters/llamaindex.py` — `ChatMemoryBuffer` + `VectorStoreIndex`. Tuned rebuilds index nightly.
- **Acceptance:** Smoke green.

### T-20 — mem0 adapter
- **Test:** Add `mem0` to T-16 parametrize.
- **Impl:** `adapters/mem0.py` — wraps `mem0.Memory`. Tuned uses Mem0's auto-update; stock disables it.
- **Acceptance:** Smoke green. **Risk:** if `mem0` install fails on macOS, defer to v2.0.1 and document in plan.

### T-21 — Harness: main loop
- **Test:** `tests/test_harness.py::test_fires_checkpoints_only_at_expected_days`, `::test_ingests_in_temporal_order`, `::test_records_metadata`, `::test_consolidate_only_in_tuned_mode`
- **Impl:** `harness.py::run_one`
- **Acceptance:** Tests green.

### T-22 — Harness: results schema
- **Test:** `::test_results_json_schema_valid` — JSON-schema validate against published schema.
- **Impl:** Add JSON schema file `results-schema.json`. Validate output.
- **Acceptance:** Test green; sample output passes schema.

### T-23 — Adapter requirements pinning
- **Deliverable:** `benchmark/amb_v2/requirements.txt` — pinned versions for langchain, llama-index, mem0, archon-memory-core (self-reference).
- **Acceptance:** `pip install -r requirements.txt` in fresh venv succeeds; smoke tests still green.

### **C4 commit** — "amb-v2: 4 adapters + harness (T-15..T-23)"
Acceptance: All adapter smoke + harness tests green; requirements installable.

## Day 4 — E2E + Chart + Pre-registration (commit C5)

### T-24 — CLI entry point
- **Test:** `tests/test_run_cli.py::test_cli_invokes_run_one_with_parsed_args` (uses `subprocess` and a stub adapter).
- **Impl:** `run.py` with argparse.
- **Acceptance:** Test green; `python -m benchmark.amb_v2.run --help` works.

### T-25 — Run-all grid runner
- **Test:** `tests/test_run_all.py::test_grid_dimensions`, `::test_skips_already_completed`
- **Impl:** `run_all.py` with `concurrent.futures` + idempotency on existing result files.
- **Acceptance:** Tests green.

### T-26 — Chart generation
- **Test:** `tests/test_chart.py::test_produces_decay_curves_svg`, `::test_produces_sensitivity_grid`
- **Impl:** `chart.py` with matplotlib. Uses fixture results JSON.
- **Acceptance:** SVG files non-empty; PNG files non-empty.

### T-27 — Golden output regression test
- **Test:** `tests/test_regression.py::test_mini_scenario_golden_output`
- **Impl:** Run mini scenario at seed=42 once, capture output as `tests/fixtures/golden_results.json`. Test re-runs and compares.
- **Acceptance:** Test green; any drift fails loudly.

### T-28 — End-to-end mini test
- **Test:** `tests/test_e2e_mini.py::test_full_pipeline_on_mini` — runs harness on mini scenario through 1 adapter (archon-memory-core), validates schema, generates chart, asserts non-zero quality.
- **Acceptance:** Runs in <30s; all assertions pass.

### T-29 — Cipher held-out generation driver
- **Deliverable:** `scripts/generate_held_out.py` — calls Cipher Agent API with the prompt verbatim from `PREREGISTERED.md`, writes 3 plaintext scenarios, encrypts with `age`, places in `held_out/`.
- **Acceptance:** Run once. Verify decrypt round-trip. Commit ciphertext only (plaintext .gitignored).

### T-30 — PREREGISTERED.md
- **Deliverable:** `benchmark/amb_v2/PREREGISTERED.md`
  - Methodology summary
  - Composite formula verbatim
  - SHA-256 of all impl files + scenario files + held-out ciphertext
  - Cipher generation prompt verbatim
  - Pre-registered Quality@90 predictions per (adapter, mode) — 8 numbers
  - Author signoff (Archon, on Andy's behalf)
  - Commit pin
- **Acceptance:** File committed; hashes recomputable from HEAD; never modified after C5.

### **C5 commit** — "amb-v2: alpha-ready harness + PREREGISTERED (T-24..T-30)"
Acceptance: `python -m benchmark.amb_v2.run_all --quick` (mini scenario, 1 seed, 1 noise rate) completes in <2 min and produces valid results JSON + chart.

## Day 5 — Cipher review + alpha run (commits C6, C7)

### T-31 — Cipher reviewer pass over implementation
- **Driver:** `archon-adversary --role reviewer --file <each impl file>`
- **Output:** `Memory/code-reviews/2026-04-23_amb-v2-implementation.md` — combined review, dispositions per critique.
- **Acceptance:** Every legitimate critique either fixed (with commit) or explicitly rejected (with rationale in disposition file).

### T-32 — Address reviewer findings
- Code commits as needed.
- **Commit C6** — "amb-v2: Phase 5 review fixes"

### T-33 — Full alpha run on Bosgame
- **Driver:** `python -m benchmark.amb_v2.run_all --noise-rates 0.20,0.30,0.45,0.60 --modes stock,tuned --adapters archon-memory-core,langchain,llamaindex,mem0 --seeds 42`
- **Wall-clock budget:** ≤ 8 hours.
- **Output:** `benchmark/amb_v2/results/headline-v2.0/*.json` (32 result files) + 4 charts.

### T-34 — Publish alpha
- Update `ROADMAP.md`: AMB v2 alpha shipped, link to results.
- Update `README.md`: AMB v2 section with chart embed.
- Update `benchmark/amb_v2/README.md`: how to reproduce, v2.1 commitment.
- Tag `v0.2.0-amb-v2-alpha`.
- Cut GitHub Release with chart attached.
- Update divergencerouter.com homepage with chart link.
- **Commit C7** — "amb-v2: alpha results published (v0.2.0-amb-v2-alpha)"

## Cross-cutting requirements

- **Coverage:** ≥90% on `chunks.py`, `queries.py`, `scenarios.py`, `simulator.py`, `metrics.py`, `harness.py`. ≥70% on adapters (smoke is sufficient).
- **Type checking:** `mypy --strict benchmark/amb_v2/` must pass at every commit.
- **Lint:** `ruff check benchmark/amb_v2/` clean at every commit.
- **No external API calls in test path.** Everything runs offline.
- **Determinism:** Every test that uses randomness must seed it explicitly.

## Phase 3 acceptance

- [x] This task list committed and pushed
- [ ] Andy review pass (look for missing tasks, ordering issues, scope creep)
- [ ] Phase 4 (Implement) begins on T-01

## Risks discovered during Phase 3 drafting

- **mem0 may not install cleanly on macOS arm64.** Plan: defer mem0 adapter to v2.0.1 if T-20 fails. Alpha ships with 3 adapters in that case.
- **age availability on Bosgame.** Need to install or use a pure-python fallback (e.g. `cryptography` Fernet). Document in T-05.
- **Reviewer pass on Day 5 may surface blocking issues.** Buffer of 2-3 hours allocated; if review reveals architectural problems, slip alpha to Day 6 rather than cut corners.
