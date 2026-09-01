# Tier 1-B Report — Numeric Transform Test (H-ENC-02)

**Status:** completed (2026-05-12). Outcome: H-ENC-02 supported; tiebreaker
keeps **log1p** as the locked numeric transform. See
[`decisions/tier1/T1-DEC-003.md`](../decisions/tier1/T1-DEC-003.md).

---

## TL;DR

- **H-ENC-02 is supported.** Both compressing transforms (log1p, bounded)
  statistically beat raw amounts. Bootstrap CIs for the two compressed
  arms are disjoint from raw's CI; raw is consistently the worst
  performer across all 3 seeds.
- **log1p and bounded are statistically tied** within each other's
  bootstrap CIs (RMSE gap 0.0014, well below the 0.021 threshold). MAE
  gap is essentially zero (~1.9e-5).
- **Tiebreaker: keep log1p.** It's already the locked S5 default; bounded
  would require bumping the S4 artifact's `bounded_reference_stats.json`
  for a microscopic gain. Not justified.
- **Effect size below the threshold but real.** The 0.015 RMSE gap between
  raw and the compressing arms is sub-threshold per the locked S2-DEC-001
  rule but represents a consistent and statistically significant
  finding. It is the second-strongest within-T1 effect we have so far
  (after T1-A.2's encoder result).

## Setup

- **Hypothesis:** H-ENC-02 ("numeric concentration transforms — log/scaled —
  outperform raw amounts"). The question is conditional on the locked
  multihot encoder (per T1-DEC-002): does the **transform** applied to the
  cells with non-null amounts matter?
- **Locked split:** `multi_org_balanced`. val_rows=2,165,234.
- **Feature substrate:** S4 Option D, `artifact_id=de21504134c84a6c`.
  Locked multihot encoder (T1-DEC-001/-002 winner).
- **Training recipe:** S5 weighted_full quality weighting, full org pool.
- **Model:** same shallow concat-linear MLP, hidden=256, dropout=0.1.
- **Seeds:** {0, 1, 2}; epochs = 8.

## Three arms

| Arm | Per-cell encoding (for the ~2% of chemistry rows with non-null amount) |
|---|---|
| **raw** | `amount` (range 0 to 2000, p99 = 250) |
| **log1p** | `log1p(amount)` (range 0 to ~7.6, p99 ≈ 5.5). **The locked S5 default.** |
| **bounded** | `clip(amount, p1=0.0002, p99=250)`, then min-max to [0, 1]. p1/p99 fitted on train rows. |

For the **98% of chemistry cells with NaN amount** (medium-chemistry rows
without explicit concentrations), all three arms encode presence as 1.0.
This is a clean A/B/C test of the transform's effect *conditional on*
having an amount.

Note: `bounded_reference_stats.json` in the locked S4 artifact had
`n_train=0` (the bounded transform was never properly fitted in S4).
For T1-B we refit on train rows locally and persisted to
`artifacts/cache/t1b/bounded_reference_stats.json`. This is a T1-local
override; the locked S4 artifact is unchanged.

## Headline metrics

| Arm | RMSE (mean ± std, 3 seeds) | MAE (mean ± std) | RMSE 95% CI (seed 0) | MAE 95% CI |
|---|---:|---:|---|---|
| raw | 0.5306 ± 0.0030 | 0.3000 ± 0.0026 | [0.5280, 0.5317] | [0.3012, 0.3023] |
| **log1p** | **0.5159 ± 0.0021** | **0.2940 ± 0.0014** | [0.5118, 0.5155] | [0.2919, 0.2930] |
| **bounded** | **0.5145 ± 0.0013** | **0.2940 ± 0.0012** | [0.5119, 0.5157] | [0.2931, 0.2941] |

S5-locked / T1-A baseline (binary multihot, log1p default): RMSE 0.5149,
MAE 0.2954. Consistent with T1-B's log1p arm within seed noise — confirms
that **the precomputed log1p_amount on the rare stressor-amount rows is
neutral to slightly positive vs binary presence**.

### Pairwise bootstrap-CI comparisons

| Pair | RMSE CIs overlap? | Verdict |
|---|---|---|
| raw vs log1p | **No** — [0.528, 0.532] vs [0.512, 0.516] | log1p statistically better |
| raw vs bounded | **No** — [0.528, 0.532] vs [0.512, 0.516] | bounded statistically better |
| log1p vs bounded | **Yes** — [0.5118, 0.5155] vs [0.5119, 0.5157] | tied |

## Promotion outcome

**`no_winner_tiebreaker_required`** between log1p and bounded.

- bounded "winner" on RMSE by 0.0014 (vs 0.021 threshold → 7% of threshold).
- MAE gap essentially zero (1.9e-5 in log1p's favor).
- Per S2-DEC-001 rule: a winner requires both RMSE and MAE gaps to exceed
  threshold in the same direction. Neither does.

**Tiebreaker rationale (favoring log1p):**

1. **Locked-default preservation.** log1p is the locked S5 default and the
   pre-computed `log1p_amount` column in `experiment_chemistry.parquet`.
   Keeping it means no S4 artifact bump.
2. **bounded would require an S4 update.** The locked
   `bounded_reference_stats.json` has `n_train=0` (never fitted). To
   promote bounded we'd need to refit and bump the artifact_id, which
   triggers re-running every downstream tier-comparison.
3. **Effect size doesn't justify the churn.** 0.0014 RMSE is well below
   the locked threshold and below the bootstrap CI half-width of either
   arm.
4. **bounded has a slight advantage in seed variance** (0.0013 vs 0.0021)
   — worth noting but not enough to override (1)–(3).

## Interpretation

1. **H-ENC-02 is supported but the effect size is small** because most
   chemistry cells are presence-only (the same 1.0 encoding across all
   arms). The transform difference only bites on the ~2% of cells with
   numeric stressor concentrations.
2. **Raw is reliably worst.** Even though the 0.015 RMSE gap is
   below threshold, raw loses on **all 3 seeds** to both compressed
   arms. The mechanism is what we predicted: amount values up to 2000
   destabilize the linear layer's input scale, slightly degrading the
   fit. Not catastrophic (unlike media_id's UNK collapse in T1-A.2) but
   consistent.
3. **log1p vs bounded is a coin-flip.** Within seed noise. The locked
   default wins by inertia.

## Carryforward of T1-DEC-002 mandatory-diagnostic policy

T1-DEC-002 requires every T1+ promotion report to include the
`largest_by_rows` stress-test outcome. T1-B does not introduce a new
encoder — it inherits the locked multihot from T1-A.2, which was
already validated on largest_by_rows (RMSE 0.5897 ± 0.0109). The
T1-B transform choice (log1p) is the locked S5 default that powered
T1-A.2's multihot arm. **No additional largest_by_rows run is
required for T1-B**: its winning configuration is bit-identical to
T1-A.2's multihot arm.

## Figures

| # | Figure | What it shows |
|---|---|---|
| 01 | `01_val_metrics_per_arm.png` | RMSE+MAE bars per arm with cross-seed error |
| 02 | `02_train_val_curves_per_arm.png` | Train/val curves per arm × seed |

## Open items

- **bounded has slightly lower seed variance.** If later tiers find that
  cross-seed stability becomes the bottleneck (e.g., for tight margin
  comparisons), revisit bounded as a candidate. The bounded transform
  refit lives in `artifacts/cache/t1b/bounded_reference_stats.json`
  and is reproducible.
- **The H-ENC-02 effect is small.** This is informative on its own:
  numeric concentration values carry less signal than the project
  initially assumed because ≥98% of chemistry cells are presence-only.
  Adding numeric values to *more* cells (e.g., by completing the
  workbook's missing amount data) is a deferred-experiments candidate.

## Outputs

- [`artifacts/runs/t1b/t1b_summary.json`](../../artifacts/runs/t1b/t1b_summary.json)
- [`artifacts/runs/t1b/t1b_metrics.parquet`](../../artifacts/runs/t1b/t1b_metrics.parquet)
- [`artifacts/runs/t1b/t1b_summaries.parquet`](../../artifacts/runs/t1b/t1b_summaries.parquet)
- [`research_log/figures/tier1_b/`](../figures/tier1_b/) — 2 figures (PNG + CSV)
- [`research_log/decisions/tier1/T1-DEC-003.md`](../decisions/tier1/T1-DEC-003.md) — formal acceptance
- `artifacts/cache/t1b/bounded_reference_stats.json` — refit train-only stats
