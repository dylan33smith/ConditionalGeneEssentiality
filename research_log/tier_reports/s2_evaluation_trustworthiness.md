# Stage 2 Report — Evaluation Trustworthiness

**Status:** approved (2026-04-28). See [`decisions/stage2/S2-DEC-001.md`](../decisions/stage2/S2-DEC-001.md).

---

## TL;DR

- **Under organism-holdout, four of the five required null baselines collapse
  to global_train_mean.** `per_condition_mean`, `per_organism_mean`, and
  `additive_baseline` all reduce to (essentially) the global mean for held-out
  organisms — val orgs/genes/conditions are all cold-start by construction,
  so each baseline's per-group estimate falls back to global. Only
  `embedding_nn` produces meaningfully different predictions.
- **`embedding_nn` is the only "gene-aware" null we have.** It beats global
  mean on `multi_org_balanced` (RMSE 0.588 vs 0.631) where val media are 100%
  in train; it loses to global mean on the other 3 protocols where ≥33% of
  val rows fall back to global-NN (because val media are not in train).
- **Bootstrap CIs on the additive baseline's within-gene Spearman are very
  tight (half-width < 0.01)**, which means we have ample power to detect a
  real model achieving Spearman ≥0.05 on any of the 4 protocols. **All 4
  protocols approved as `primary` for Spearman.**
- **`H-BASE-01` reinterpreted.** The original H-BASE-01 framing ("model must
  beat additive baseline → proves it learns gene×condition interactions")
  loses its bite under organism-holdout because additive collapses to a
  constant. A revised gate is locked in S2-DEC-001: a model must beat
  **embedding_nn** by RMSE+MAE thresholds (or global_mean when NN is also
  cold-start). H-BASE-01 retained as a sanity check.
- **Gain-threshold rule revised.** The original "0.5 × (global − additive)"
  rule produces non-positive thresholds under organism-holdout. New rule:
  `max(0.005, 0.5 × |global_mean − best_non_global_baseline|)`. Result:
  3 of 4 protocols hit the 0.005 floor; `multi_org_balanced` gets a
  threshold of 0.021 RMSE (real gap to NN baseline).

## Per-protocol baseline metrics

| Protocol | global RMSE | per_cond RMSE | per_org RMSE | additive RMSE | NN RMSE | NN fallback | n_eligible | Spearman role |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `largest_by_rows` | 0.5668 | 0.5742 | 0.5668 | 0.5675 | 0.6060 | 100% | 3041 | primary |
| `high_overlap_easy` | 0.5866 | 0.5875 | 0.5866 | 0.5873 | 0.6251 | 33% | 6095 | primary |
| `low_overlap_stress` | 0.9370 | 0.9386 | 0.9370 | 0.9388 | 0.9793 | 100% | 1424 | primary |
| `multi_org_balanced` | 0.6307 | 0.6319 | 0.6307 | 0.6320 | **0.5884** | 0% | 13804 | primary |

(MAE and full numbers in `artifacts/baselines/baselines_per_protocol.json`.)

`multi_org_balanced` is the only protocol where NN beats global mean — and it's
the only protocol with full chemistry+media overlap on val. The other three have
val media that are organism-unique strings, so their NN baseline falls back to
"nearest train gene globally → that gene's mean fit," which is too noisy to
beat a constant predictor.

## Power analysis

For each protocol, we ran:
1. **Eligibility curve** (n_genes_eligible vs m): all 4 protocols have ≥1,400
   genes at m=5, ≥800 at m=12. Power is not the bottleneck.
2. **Bootstrap-by-gene 95% CI** for the additive-baseline mean within-gene
   Spearman: half-width 0.003–0.007 across protocols. We can confidently detect
   a real model gain ≥0.02 Spearman.
3. **Permutation null** (200 perms): null mean Spearman ≈ 0; p95 < 0.01.
   Real-model results above this threshold are above-chance.

## Heteroscedastic-noise diagnostics

The additive-baseline residual quantile profile (fig 06) confirms the heavy-tailed
behavior we already saw in S1 fig 13:
- Residual P5 ≈ −1.4 to −2.4 across protocols
- Residual P95 ≈ +0.7 to +1.0
- Asymmetric: left tail much fatter than right. Tn-seq fitness has more very-bad
  observations than very-good ones.

This **confirms RMSE+MAE co-primary policy is the right call**. RMSE alone
overweights the left tail. A Huber-loss model (T4) is well-motivated.

Per-organism residual std (fig 07) varies 2–5× across organisms within a single
protocol → noise is heteroscedastic across organisms. Per-organism reporting
required for any tier-level claim.

## Required figures

| # | Figure | Decision it informs |
|---|---|---|
| 01 | `01_baseline_metrics_per_protocol.png` | which baselines are required (all 5 retained per H-BASE-01) |
| 02 | `02_difficulty_ladder.png` | per-protocol difficulty ranking (low_overlap_stress is hardest, additive ≈ global elsewhere) |
| 03 | `03_spearman_eligibility_at_m.png` | confirms m=5 lock; eligibility comfortably exceeds 200 in every protocol |
| 04 | `04_bootstrap_ci_per_protocol.png` | Spearman role decision (all 4 → primary) |
| 05 | `05_permutation_null_spearman.png` | null distribution well-separated from any real signal |
| 06 | `06_residual_quantile_profile.png` | heteroscedastic noise → RMSE+MAE co-primary, Huber motivated for T4 |
| 07 | `07_per_organism_residual_spread.png` | per-organism residual std varies 2–5× → per-org reporting mandatory |

## Hard-gate decisions

| Decision | Outcome | Reference |
|---|---|---|
| Spearman role per protocol | All 4 → `primary` | bootstrap-CI half-width < 0.01 across protocols, n_eligible ≥1,400 |
| Final null-baseline set per protocol | All 5 retained: global_train_mean, per_condition_mean, per_organism_mean, additive_baseline, embedding_nn | `eval_policy.yaml` |
| Gain-threshold rule | revised to `max(0.005, 0.5 × \|global − best_non_global\|)` | S2-DEC-001 |
| H-BASE-01 reinterpretation | "must beat additive" → "must beat best-non-global baseline (NN where applicable, else global mean)" | S2-DEC-001 |
| `v_min` per protocol | Locked at the 25th percentile of cross-gene IQR on each candidate's val set; values 0.186–0.238 | `eval_policy.yaml::v_min_value_per_protocol` |

## Outputs

- [`artifacts/baselines/baselines_per_protocol.json`](../../artifacts/baselines/baselines_per_protocol.json) — full numerical results
- [`data_contract/policy/eval_policy.yaml`](../../data_contract/policy/eval_policy.yaml) — locked policy
- [`research_log/figures/stage2/`](../figures/stage2/) — 7 figures (PNG + sibling CSV)
- [`research_log/decisions/stage2/S2-DEC-001.md`](../decisions/stage2/S2-DEC-001.md) — formal acceptance + rule revisions

## Open items carried into S3

- The auto-decision logic flagged all 4 protocols as `primary` for Spearman.
  But `high_overlap_easy`'s additive-baseline mean Spearman straddles zero
  ([−0.003, 0.000]). It's *power-fine* (we'd detect a real model achieving 0.05),
  it's just that additive itself learns nothing on this protocol. S3 should be
  aware: a model that produces non-zero Spearman is the meaningful signal,
  not a model that beats the additive baseline's Spearman per se.
- The 100% NN-fallback rate on `largest_by_rows` and `low_overlap_stress` is
  a direct consequence of the media-name uniqueness finding from S1. S3 should
  consider whether the locked split should prefer protocols with lower fallback.
- `multi_org_balanced` has the highest n_eligible (13,804) and the only
  non-trivial NN baseline (RMSE 0.588 < global 0.631). It's structurally the
  most informative protocol for tier comparisons. Strong S3 candidate for
  primary-promotion role.
