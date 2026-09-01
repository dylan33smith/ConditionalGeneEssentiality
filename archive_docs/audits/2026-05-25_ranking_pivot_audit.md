# Ranking-Pivot Audit & Alignment Report

**Date:** 2026-05-25
**Method:** 4-phase multi-agent workflow — Phase 1 (discovery, Explorer agent),
Phase 2 (research-rigor, Auditor agent), Phase 3 (fixes, main thread),
Phase 4 (verification + this report).
**Scope:** the R-regime ("ranking pivot") — R0 + R-LOCK-1..4 + contracts + code.

---

## 1. PHASE-3 CHANGE TALLY (every file modified, and why)

All changes are **documentation / docstring / comment** edits. **Zero code-logic
changes** — Phase 1 confirmed the R-regime leaf code (metrics, samplers, sign
convention, bootstrap) is mathematically correct, so no behavior was altered.
All 107 tests still pass.

| File | Change | Why (severity) |
|---|---|---|
| `docs/RPLAN.md` §2.3 | `replicate_handling_val` pooling key `expName` → `condition_key` | **SPEC BUG.** Pooling within `expName` (the replicate id) is a no-op; would inflate the primary metric with intra-condition noise. Code/contract already use `condition_key`; only the plan was wrong. |
| `docs/RPLAN.md` §2.2 | `holdout_unit` `expName` → `condition_key`; `holdout_k` → `holdout_fraction 0.20`; added `condition_key=(expDesc,media,temperature)` | **LEAKY DOC.** Stale "Default" column under a "locked" header described an expName-level split that would leak replicate siblings across partitions. Now matches the locked `split_protocol.yaml`. |
| `docs/RPLAN.md` §14, `metric_contract.yaml`, `R-LOCK-4-DEC-001.md` | H-RANK-01 "per-gene Spearman constant across genes" → "varies across genes" | **FALSE MATH CLAIM.** Baseline gives every gene the same *prediction vector* but each gene's Spearman correlates it against that gene's own fit, so per-gene Spearman varies. Baseline stays valid; only the rationale was wrong (the unit test even relied on the variation). |
| `docs/RPLAN.md` §5 | Added post-execution note (tail_g supersedes IQR; fig 07 dropped, fig 11 added; temperature in key); fixed figure list + count | **DOC DRIFT.** §5 described the pre-tail_g plan as if current. |
| `docs/RPLAN.md` §8 | Rewrote H-R-FUSE-01 mechanism: "early-concat absorbed gene-mean" → "FiLM parameterizes gene×condition interaction directly" | **WRONG MECHANISM.** Any gene-input architecture (incl. FiLM) can fit a per-gene offset; the real claim is about multiplicative interaction. Experiment unchanged; justification corrected. |
| `docs/RPLAN.md` §1 table | R0 "10 figures" → "11 figures" | Doc drift. |
| `CLAUDE.md` R0 row | "10 figures" → "11 figures (07 dropped, 11 added)" | Doc drift. |
| `CLAUDE.md` T6 row | "theory predicts fingerprints will win" → "theory *permits*; prior ≈ 50/50" | **OVERSTATED PRIOR.** ≥95% medium-chemistry overlap (S1) means held-out conditions may be recombinations of seen chemistries where multihot is already sufficient (why T6-A found no gain). |
| `CLAUDE.md` R1 row | "READY TO START" → "CONTRACTS LOCKED; integration pending" + 5 integration items + open blockers | **OVERSTATED READINESS.** Contracts/leaf-code are locked but nothing is wired (no eligibility impl, no split materializer, no training-loop wiring, no v2 dispatcher, no R1 handler). |
| `eligibility_policy.yaml` | Added ⚠ train-only-leakage waiver block above `r_replicate_org` | **DATA LEAKAGE (documented, not silently fixed).** Per-org `r_replicate` was computed on full data, not train-only. |
| `R-LOCK-1-DEC-001.md` risks | Added the `r_replicate` train-only BLOCKER + R1 resolution requirement | Same leakage, recorded in the decision ledger. |
| `src/experiments/r0/analyses.py` | 6 docstrings `(expDesc, media)` → `(expDesc, media, temperature)`; `signal_to_noise` clip wording | Docstring drift (code already used `_condition_key` incl. temperature). |
| `R-LOCK-3/4-DEC-001.md` | Corrected 3 wrong line-count claims (380→265, 140→168, 290→301) | Trivial accuracy. |

**Tracked vs untracked:** the entire R-regime (R0, R-LOCKs, ranking modules,
contracts, this report) is **untracked** — nothing has been committed since
`0294a69`. This is a governance gap: the audited work has no commit history.
**Recommendation: commit the R-regime work** so the locks and their fixes are
reproducible and diffable.

---

## 2. RANKING-PIVOT SOUNDNESS ASSESSMENT

**Verdict: the pivot is mathematically sound and the leaf code is correct and
well-tested, but it is NOT yet integrated, and the research framing has
substantive gaps that must close before R1 produces a defensible result.**

**Sound (verified):**
- The core premise is correct. Within-gene Spearman is invariant to per-gene
  additive/monotone offsets, so it genuinely removes the gene-mean component
  that T-regime MSE absorbed. The pivot targets the right quantity.
- Sign convention (low fit = essential) is consistent everywhere
  (`ranking_batch.py`, pairwise sign, decision docs). No flips.
- Metric code is correct: per-gene Spearman/Kendall skip constants and short
  genes; bootstrap resamples **genes** with replacement; task-relevant noise
  floor pairs replicates correctly; PairwiseSampler pairs within-gene.
- Leakage is closed *by construction* in the split design: `condition_key`
  holdout pulls replicate siblings together (the DvH 98%-replicated case shows
  this was a real, severe risk that fraction-by-expName would have hit).

**Not sound yet / not done:**
- **Integration gap.** The two load-bearing contracts — eligibility/`w_g`
  weighting (R-LOCK-1) and split materialization (R-LOCK-2) — exist only as
  YAML + prose. Grep confirms **no implementation** of `tail_min_org`,
  `tail_ref_org`, `w_g`, or a split materializer anywhere in `src/`. No
  training loop is wired to `RankingBatch`; no run-manifest v2 dispatcher
  exists (only the hardcoded v1 validator at `stage0/run.py:63,206`).
- **One locked artifact carries a train-only leakage** (`r_replicate` on full
  data) — small magnitude, now documented as a blocker.
- **The headline noise floor that justified the pivot (~0.43) is a disowned
  proxy.** R-LOCK-1 itself flags it measures the wrong quantity; the
  task-relevant floor (R-LOCK-4) is implemented but **never computed on real
  data** (only synthetic unit tests). No R-tier gain is interpretable until
  the true ceiling is measured.

---

## 3. DATA SPLIT / LOADER ACCURACY VERDICT

**Split design: CORRECT and leakage-free by construction — but UNIMPLEMENTED.**
- `condition_key = (expDesc, media, temperature)` holdout with
  replicate-group-together provably prevents replicate-sibling leakage.
- Temperature-in-key is right (2.4% of `(expDesc,media)` groups hid distinct
  temperatures).
- Fraction-based (0.20) handles the 17× org-size spread correctly; 38/44 orgs
  retain ≥100 val-eligible genes.
- **Caveat:** this guarantee is only as good as the future materializer, which
  does not exist yet. R0's `split_feasibility` *simulates* it correctly but
  emits no reusable split manifest.

**Loader (`RankingBatch` + samplers): CORRECT and unit-tested (15 tests), but
NOT wired to any dataset or training loop.** `build_s5_dataset.py` remains the
T-regime substrate; no `RankingDataset` applies eligibility weighting or
condition-holdout.

**Eligibility: train-only violation in the materialized `r_replicate` table**
(documented as a blocker, §1). Otherwise the `tail_g` math matches the decision.

---

## 4. CRITICAL RECOMMENDATIONS FOR PUBLICATION-READY PHASE R

These come from the Phase-2 research audit and **require project-lead decisions**
(they modify locked decisions or add new work — NOT applied unilaterally):

### Must-fix before R1 trains anything
1. **Compute the task-relevant noise floor on the real locked val rows** and
   reconcile the three circulating numbers (0.43 proxy / 0.506 R0 / unmeasured
   task-relevant). No result is interpretable without the true ceiling.
   **→ DONE 2026-05-25 (partial):** materialized the R-LOCK-2 condition-holdout
   split (`src/data/datasets/build_ranking_split.py`), leakage-checked on real
   data (PASS), and computed the task-relevant floor on the 8 high-replicate
   orgs' real val rows: **median per-gene cross-condition replicate Spearman =
   0.358** (mean 0.361; n=24,337 val genes; 3,668 skipped for <5 paired
   conditions). This is the TRUE ceiling for the ranking metric and is **lower
   than the 0.43–0.506 cross-gene proxies** — replicates agree less on
   *within-gene condition ranking* than on *within-condition gene ranking*.
   Implication: the "room to improve" from the H-RANK-01 baseline up to ~0.36
   is narrower than the pivot's headline 0.43 suggested; every R-tier promotion
   delta must be judged against ~0.36, not 0.43. **Remaining:** extend to all
   replicate-bearing orgs and recompute per-fold for the final locked split.
2. **Add strong baselines to the H-RANK-01 gate**, above all **matrix
   factorization / low-rank completion** on the gene×condition `fit` matrix.
   The primary split is **transductive over genes** (every gene's embedding is
   in train; only condition *combinations* are novel), which makes the task
   close to matrix completion — so MF is the baseline a reviewer will demand.
   Also add **chemistry-kNN** and the **additive `α[gene]+β[condition]` model**.
3. **Resolve the train-only `r_replicate` blocker** — recompute eligibility
   train-only per fold, or accept an explicit waiver with a sensitivity check.
4. **Add a cold-gene diagnostic split** (hold out whole genes) to show the
   model uses embedding biology, not memorized per-gene offsets. Pre-register
   the interpretation: "primary ≈ H-RANK-01 ≪ cell_holdout ⇒ null result."

### Should-fix for rigor
5. **Add retrieval metrics (NDCG@k, precision@k)** to match the stated "find
   the top stressors" motivation; Spearman alone under-serves the top-k framing.
6. **Re-anchor promotion deltas to a practical effect size** (a fraction of the
   H-RANK-01→noise-floor gap), not to bootstrap-CI width (circular; with
   n≈3000 genes the CI is tiny so 0.01 may be practically vacuous).
7. **Hierarchical bootstrap (org→gene)**, not flat gene bootstrap — genes
   within an org share conditions/noise and aren't independent; flat CIs are
   over-confident.
8. **Multiple-comparison budget (FDR/family-wise)** across the ~12 planned
   promotion tests (R1, R2, + 8 deferred tiers), and convert the per-lock "may
   be revised after R1" escape hatches into a single pre-registered
   recalibration checkpoint.
9. **Per-organism breakdown tables** given the 17× size spread and global-mean
   aggregation that can let large orgs dominate.

### Hypothesis fixes (already partially applied to docs)
10. **R1 (H-R-CHEM-01):** softened to "theory permits" (done); add a
    chemistry-space nearest-train-condition diagnostic — if held-out conditions
    are multihot near-duplicates of seen ones, fingerprints can't help and R1
    is a confirmatory null.
11. **R2 (H-R-FUSE-01):** mechanism rewritten around interaction modeling (done).
12. **Consider promoting R-LOSS ahead of R1** — studying chemistry encoders
    under a pointwise-MSE proxy while claiming to study *ranking* is itself
    questionable; the loss is the more fundamental axis.
13. **R-CURRIC** (pseudo-labeling) as specified has a **circular evaluation**
    (pseudo-labels from a high-IQR model, success measured on high-IQR genes).
    Require a pseudo-label-free control + unbiased held-out gene set, or drop.
14. **R-LOCK-2 dev subset (Keio)** has 49% singleton conditions, so the
    task-relevant noise floor may be unreliable on the dev org — verify before
    relying on its dev-figure reference line.

---

## 5. PHASE-4 VERIFICATION RESULTS

- `pytest tests/` → **107 passed, 2 skipped** (pre-existing stubs), 2 warnings
  (pre-existing stage2, unrelated). No regression.
- Edited YAML contracts parse (`eligibility_policy.yaml`, `metric_contract.yaml`).
- v2 JSON schema valid.
- All R-regime modules + R0 import cleanly after docstring edits.
- No residual stale "10 figures" / "constant across genes" / leaky
  "holdout_unit: expName" references remain.

---

## 6. TECHNICAL BLOCKERS (documented, not fixed — per the no-breaking-changes constraint)

1. **R-regime integration is absent** (eligibility impl, split materializer,
   training-loop wiring, v2 manifest dispatcher, R1 handler). This is the R1
   implementation task itself, not an audit fix.
2. **Train-only `r_replicate` leakage** in a locked artifact (small magnitude;
   must be resolved at R1 integration).
3. **Task-relevant noise floor unmeasured on real data.**
4. **Missing strong baselines (matrix factorization etc.)** — a research
   decision, not a code fix.
5. **All R-regime work uncommitted** (untracked since `0294a69`).
