# Tier 1-B.3 Report — Controlled Concentration-Inclusion Test (binary vs log1p)

**Status:** completed (2026-05-12). Outcome: **revert lock to binary**.
**Supersedes T1-DEC-003 on the chemistry-encoding choice.** See
[`decisions/tier1/T1-DEC-004.md`](../decisions/tier1/T1-DEC-004.md).

---

## TL;DR

- **T1-B never directly tested "with vs without concentration information."** It
  compared transforms (raw, log1p, bounded) but all three arms included
  concentrations. T1-B.3 closes that gap with a controlled head-to-head:
  pure binary multihot vs log1p-with-concentrations.
- **RMSE is statistically tied** (CIs overlap; binary edge of 0.0009 is below
  threshold). **MAE shows a small statistically-significant log1p advantage
  of 0.0017** (CIs disjoint by 0.0015) — but below the 0.005 threshold and in
  the H-METRIC-01 disagreement regime against RMSE.
- **Lock reverts to binary.** Concentrations are in mixed units
  (`feature_contract.yaml::concentration_policy`); including them isn't
  honestly defensible without unit normalization. The empirical case for
  inclusion is below threshold, so we err on the side of cautious
  representation.
- **T1-DEC-003 (lock log1p) is superseded by T1-DEC-004 (lock binary).**
  Locked T1 substrate is now: `multihot_canonical_id` encoder + binary
  presence/absence — no concentration values at all.

## Why this experiment exists

T1-B tested whether `raw amount` vs `log1p(amount)` vs `bounded(amount)`
matters when concentrations *are* included. It concluded log1p ≈ bounded ≫ raw
(T1-DEC-003).

The unstated assumption: that we should include concentrations in the first
place. Per `feature_contract.yaml`:

> "units_1..units_4 are NOT consumed in S4. concentration_* values are pooled
> without unit conversion — do NOT use raw concentrations for cross-experiment
> comparisons or magnitude claims until T1 normalizes units."

So "amount=10" might mean 10 mM in one experiment and 10 g/L in another.
T1-B fed those mixed-unit values to the model anyway because they happened to
exist in the S4 artifact. T1-B.3 asks the proper question: does removing the
concentrations entirely (presence-only) hurt the model?

## Setup

- **Hypothesis:** "Including unit-mixed concentration values on ~2% of
  chemistry cells provides measurable signal beyond presence/absence."
- **Locked controls:** all same as T1-A and T1-B (multi_org_balanced,
  feature artifact de21504134c84a6c, weighted_full S5 policy, shallow
  concat-linear MLP, seeds {0,1,2}, 8 epochs, hidden=256).
- **Only differences:** the two chemistry matrices.

| Arm | Cell encoding |
|---|---|
| **binary** | 1.0 wherever chemistry is present, 0.0 elsewhere. **No concentration info.** Same as T1-A's multihot. |
| **log1p** | 1.0 for cells with NaN amount; `log1p(amount)` for cells with recorded concentration. The T1-B locked default. |

## Headline metrics (mean ± std across 3 seeds)

| Arm | RMSE | MAE |
|---|---:|---:|
| binary | 0.5150 ± 0.0020 | 0.2957 ± 0.0014 |
| log1p | 0.5159 ± 0.0021 | 0.2940 ± 0.0014 |
| **gap (binary − log1p)** | **−0.0009** | **+0.0017** |

S2-locked threshold for `multi_org_balanced`: **0.0212 RMSE, 0.005 MAE.**
Both gaps are far below threshold.

## Bootstrap CIs (seed 0, 1000 row-resamples)

| Arm | RMSE 95% CI | MAE 95% CI |
|---|---|---|
| binary | [0.5113, 0.5149] | [0.2945, 0.2956] |
| log1p | [0.5118, 0.5155] | [0.2919, 0.2930] |

- **RMSE: CIs overlap heavily.** binary [0.5113, 0.5149] and log1p [0.5118,
  0.5155] share most of their support. RMSE is tied.
- **MAE: CIs are disjoint.** binary [0.2945, 0.2956] sits **above** log1p
  [0.2919, 0.2930] by ~0.0015. log1p is statistically below binary on MAE,
  but the gap is **below the 0.005 threshold** (60% of it).

## The H-METRIC-01 regime, third occurrence

This is the third T1 experiment where RMSE and MAE disagree about the winner:

- T1-A: tie on RMSE, media_id MAE win (sub-threshold)
- T1-B: raw clearly loses, log1p ≈ bounded
- **T1-B.3**: tie on RMSE, log1p MAE win (sub-threshold)

The pattern is consistent: encoders that include concentration values
(log1p, media_id when val media all present) tend to fit the central mass
slightly better (lower MAE) but pay a tiny RMSE cost from less-controlled
tail behavior. Under heavy-tailed `fit` residuals, this trade-off is
expected (S1 fig 13, S2 residual quantile diagnostics).

## Decision: lock binary

The decision rule fires `promote_binary_drop_concentrations` because:

1. **RMSE is tied** (CI overlap; binary marginally better by 0.0009).
2. **MAE gap is statistically real but sub-threshold** (0.0017 vs 0.005
   threshold).
3. **Per S2-DEC-001:** a winner requires both RMSE *and* MAE gaps to
   exceed threshold in the same direction. log1p does not qualify.

Even setting aside the strict rule, two epistemic factors break ties
toward binary:

4. **Unit-mixing concern.** The mixed-unit concentrations cannot be
   honestly described as "concentration information" — they're "numeric
   strings from the workbook with no unit standardization." Including them
   without unit normalization invites a reviewer to ask "what do these
   numbers mean?" with no good answer.
5. **The empirical case is weak.** A 0.0017 MAE advantage on the central
   mass doesn't justify the representation complexity. If concentrations
   eventually get unit-normalized (REFACTORPLAN §12), revisit.

## What changes

| Lock | Before T1-B.3 | After T1-B.3 |
|---|---|---|
| Condition encoder | multihot_canonical_id | multihot_canonical_id (unchanged) |
| Numeric transform | log1p (T1-DEC-003) | **binary (T1-DEC-004 supersedes)** |
| Substrate artifact | de21504134c84a6c (unchanged) | de21504134c84a6c (unchanged) |

The S4 artifact does NOT change — we just choose to ignore the `amount` /
`log1p_amount` columns when building T1+ chemistry matrices. The columns
remain available in case a future tier wants to revisit (e.g., after unit
normalization).

## Carryforward / mandatory diagnostic

T1-A.2's largest_by_rows mandatory diagnostic carries over: binary +
multihot is *exactly* the encoding T1-A used, and T1-A's largest_by_rows
stress test (T1-DEC-002) showed multihot beats media_id by 3.26 RMSE on that
protocol. No new largest_by_rows run is required for T1-B.3.

## Figures

| # | Figure | What it shows |
|---|---|---|
| 01 | `01_val_metrics_per_arm.png` | RMSE+MAE bars per arm with cross-seed error |
| 02 | `02_train_val_curves_per_arm.png` | Train/val curves per arm × seed |

## Open items

- **The MAE-only log1p advantage is real but small.** If a later tier
  (T2 / T3 / T4) finds that central-mass error reduction matters
  disproportionately, we can revisit the binary-vs-log1p choice. T1-B.3's
  data is reproducible from `artifacts/cache/t1b3/` and the arm-comparison
  code in `_t1b.py::run_t1b3`.
- **Unit normalization is in §12 Deferred.** When/if `units_1..units_4` are
  parsed and used to normalize `amount`, the H-ENC-02 question deserves
  another look. Filed.

## Outputs

- [`artifacts/runs/t1b3/t1b3_summary.json`](../../artifacts/runs/t1b3/t1b3_summary.json)
- [`artifacts/runs/t1b3/t1b3_metrics.parquet`](../../artifacts/runs/t1b3/t1b3_metrics.parquet)
- [`artifacts/runs/t1b3/t1b3_summaries.parquet`](../../artifacts/runs/t1b3/t1b3_summaries.parquet)
- [`research_log/figures/tier1_b3/`](../figures/tier1_b3/) — 2 figures
- [`research_log/decisions/tier1/T1-DEC-004.md`](../decisions/tier1/T1-DEC-004.md)
