# PROGRESS — append-only log

Newest first. Never rewrite past entries. One entry per task that changes behavior
or structure; record what was done, what was learned, and the regression result.

---

## 2026-06-15 — Layering fixes (post-cleanup polish)

Resolved the two items flagged at the end of the cleanup, each gated:
- **Fix A** — extracted the condition-key helpers (`_condition_key`,
  `_normalize_string_keys`, `load_fitness`) from `src/experiments/r0/analyses.py`
  into `src/data/datasets/conditions.py`. They belong in the data layer because the
  data modules AND the ranking pipeline both consume them. Net: nothing in `src/data`
  or `src/ranking` imports upward into `src/experiments` — clean layering.
- **Fix B** — deduped `eval/contract.py`'s per-gene-correlation against the
  canonical `harness.per_gene_correlations` (verified equivalent: test_ranking_metrics
  17/17 unchanged). **Learned:** contract's flat bootstrap and harness's hierarchical
  org→gene bootstrap are genuinely DIFFERENT methods, not duplicates — kept both,
  documented; did not merge (would have swapped a flat CI for a hierarchical one).

Regression: R-EVAL fast gate bit-exact (Δ=0.0000); pytest 119 passed.

---

## 2026-06-15 — Ranking-branch cleanup (reorganize + aggressive prune + modular runner)

**Goal.** Produce a clean `ranking` branch as the basis for the top-k objective:
prune anything not needed / that didn't work (learnings preserved in the decision
ledger + synthesis), and make further tests easier (modular core, one shared
runner, consistent reporting).

**Done (each step gated by R-EVAL + pytest; all bit-exact unless noted):**
1. **R-EVAL regression harness** — one pinned command reproduces the locked-best
   model (pointwise_huber) vs chem-kNN at split seed 0, k=5. Fast gate (Keio+Caulo+
   MR1): model NDCG@5 0.4516 / chem-kNN 0.5091, bit-exact reproducible. Baseline at
   `data_contract/ranking/reval_baseline.json`.
2. **Extracted the model** (`ResidualBlock` + `AdapterResidualMLP`) out of the
   legacy T-tier into `src/ranking/models.py`.
3. **Decoupled ranking from the T-tier** — moved the fingerprint loader and the
   eval layer (`harness.py` + `contract.py`) and the loss family into `src/ranking`;
   no ranking module imports any `tier*` module.
4. **Relocated pipeline + trainers** into `src/ranking/{pipeline,train}.py` and
   added `src/ranking/runner.py` (declarative `ArmSpec` + `run_experiment` +
   standardized reporting). `reval` and `rloss` handlers are now thin runner specs.
5. **Pruned the T-regime + dead code** — 130 files / −13.4k lines: stage0-5,
   tier1-6, diagnostics, `src/{train,models,domain,training,evaluation}`, the failed
   hybrids, dead data utils, 51 T-regime tests, T*/stage* configs, root cruft;
   deregistered S0-S5/T1-T6 CLI handlers. Learnings preserved in the decision ledger
   + SCIENTIFIC_SYNTHESIS; mapping in `docs/PRUNED_INDEX.md`.
6. **Docs scheme** — rewrote CLAUDE.md (durable only + doc rule), created
   ARCHITECTURE.md / PLAN.md / PROGRESS.md / docs/PRUNED_INDEX.md; deleted the stale
   T-regime planning docs (RPLAN/REFACTORPLAN/PROJECT_TEXTBOOK_T6/TEACHING_BLUEPRINT/
   STAGE_LEARNINGS — content folded into ARCHITECTURE or preserved in the ledger).

**Learned / verified.** Training is deterministic on a fixed device → the gate is
bit-exact, which made every refactor step verifiable. The reusable substrate for
the top-k work (loss family incl. NDCG-direct losses, `RankingBatch`, eval/baseline
harness) was kept; only single-use glue and the T-regime were deleted.

**Regression.** R-EVAL fast gate PASS (Δ=0.0000) after every step; pytest green
(125 ranking-relevant tests). Full 23-org headline (anchors ~0.435/0.485) run
separately to confirm the published numbers.

**Next.** See PLAN.md — wire `RankingBatch` into training (expected to move numbers;
that's the first top-k modeling step, not a regression).
