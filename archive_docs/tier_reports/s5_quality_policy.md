# Stage 5 Report — Training Recipe Lock

**Status:** approved (2026-04-29). Decision: `research_log/decisions/stage5/S5-DEC-001.md`.

## TL;DR

- Winner: **`weighted_full`** row-quality policy (`organism_pool=full`).
- Co-primary metrics (mean ± std across seeds):
  - `weighted_full`: RMSE **0.5151 ± 0.0017**, MAE **0.2957 ± 0.0014**
  - `strict_slice`: RMSE **0.5193 ± 0.0012**, MAE **0.2997 ± 0.0007**
- H-BASE-01 gate: both arms beat additive baseline RMSE on the same val rowset; winner still `weighted_full`.
- Frozen policy emitted to `data_contract/policy/quality_policy.yaml`.

## Experimental setup

- Split protocol: `multi_org_balanced` from `data_contract/splits/locked_protocol.yaml`.
- Feature substrate: S4 Option D `artifact_id=de21504134c84a6c`.
- Model vehicle (held fixed across arms): shallow concat-linear MLP
  (`gene_emb + role-blind chemistry multihot[425]`, no metadata).
- Seeds: `[0, 1, 2]`; epochs: `8` (no early stop) to inspect train-vs-val behavior.
- Row-quality policies compared:
  - `weighted_full`: all rows kept, weight = `clamp(cor12/median_cor12,0,1) * clamp(|t|/median_|t|,0,1)`.
  - `strict_slice`: keep only rows with `cor12 >= q25_cor12` and `|t| >= q25_|t|`.
- H-POLICY-02 (`curated` pool) skipped with rationale from S1 support gating.

## Frozen thresholds and effective sample size

- `cor12_median_train`: `0.213965`
- `abs_t_median_train`: `0.701673`
- `cor12_q25_train`: `0.162083`
- `abs_t_q25_train`: `0.319171`
- Effective train sample size:
  - `weighted_full`: `15,800,521`
  - `strict_slice`: `13,878,149`
- Strict-slice drop attribution:
  - dropped only by `cor12`: `4,524,885`
  - dropped only by `|t|`: `4,506,870`
  - dropped by both: `1,622,835`

## Arm-level results

| Arm | RMSE mean | RMSE std | MAE mean | MAE std | Best epoch mean |
|---|---:|---:|---:|---:|---:|
| weighted_full | 0.515147 | 0.001741 | 0.295729 | 0.001393 | 5.67 |
| strict_slice | 0.519310 | 0.001162 | 0.299679 | 0.000668 | 5.33 |

Train/val behavior:
- `weighted_full` shows negative train-val gap on RMSE/MAE (val slightly better than train under weighted objective), consistent across seeds.
- `strict_slice` shows positive train-val gap, indicating stronger overfit under the stricter retained subset.

Spearman (secondary, S2 eligibility policy):
- Mean within-gene Spearman over all epochs:
  - `weighted_full`: `0.0386`
  - `strict_slice`: `0.0353`
- Eligible genes per epoch remained high (min over run: `13,572` weighted, `13,720` strict).

## Additive baseline gate (H-BASE-01)

On the same val denominator per arm:

| Arm | Additive RMSE | Additive MAE | Model RMSE mean | Pass RMSE gate |
|---|---:|---:|---:|---|
| weighted_full | 0.631979 | 0.315920 | 0.515147 | true |
| strict_slice | 0.636887 | 0.338548 | 0.519310 | true |

## Threshold sensitivity

Winner was `weighted_full`, so sensitivity used a weight-floor stress test:

| Setting | val RMSE | val MAE |
|---|---:|---:|
| base | 0.515147 | 0.295729 |
| weight_floor_p5 | 0.512388 | 0.293897 |
| weight_floor_p15 | 0.516200 | 0.296215 |

Conclusion: winner remains stable under moderate weight-floor perturbation.

## Artifacts

- Policy contract: `data_contract/policy/quality_policy.yaml`
- Run summaries: `s5_metrics.parquet`, `s5_summary.parquet`, `s5_additive_gate.parquet`
- Figures (+ CSV sidecars):
  - `research_log/figures/stage5/01_arm_train_val_curves.png`
  - `research_log/figures/stage5/02_arm_summary_bar.png`
  - `research_log/figures/stage5/03_per_org_val_rmse.png`
  - `research_log/figures/stage5/04_residual_quantiles.png`
  - `research_log/figures/stage5/05_train_dynamics.png`
  - `research_log/figures/stage5/06_threshold_sensitivity.png`
