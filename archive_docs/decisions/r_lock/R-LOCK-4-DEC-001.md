## Decision: R-LOCK-4-DEC-001

### Header
- decision_id: R-LOCK-4-DEC-001
- stage_or_tier: R-LOCK-4
- regime: R
- date: 2026-05-25
- owner: project lead
- status: approved
- related_experiments: [R0-A_data_characterization]
- related_hypotheses: [H-RANK-01]

### Assumption Under Test
- assumption_statement: The R-regime primary metric is **within-gene Spearman**
  with **bootstrap CI over genes (n=1000, 95% percentile)**, the secondary
  co-primary is **within-gene Kendall's τ**, the H-RANK-01 baseline is
  **per-condition train mean**, and the primary noise floor is the
  **task-relevant** per-gene cross-condition Spearman across replicate pairs
  (NOT the cross-gene within-condition proxy).
- assumption_type: evaluation
- why_it_matters: The metric contract is the lens through which every
  R-tier promotion is judged. Getting it wrong means we either ship a
  system that doesn't generalize the way we claim, or we kill promising
  work because the metric is mis-targeted. R-LOCK-4 is the analog of S2
  for the ranking regime.

### Pre-Registered Test Plan
- comparison: n/a — engineering + analysis lock, not an A/B test.
- promotion_rule:
  - All metric helpers unit-tested with known answers.
  - H-RANK-01 baseline reproduces the "per-condition train mean" formula
    as stated in the contract.
  - Task-relevant noise floor function returns 1.0 when replicates are
    identical and `nan`/zero-genes when no gene has ≥ 5 paired conditions.
  - Bootstrap CI brackets the mean.
- failure_guardrails:
  - Every R-tier figure that shows a Spearman value MUST also show:
    bootstrap CI, H-RANK-01 baseline line, noise floor line, cross-org
    drift line (per `metric_contract.yaml::figure_requirements`).

### Evidence Summary

**Contract:** `data_contract/ranking/metric_contract.yaml`.

| Field | Locked value |
|---|---|
| `primary_metric` | `within_gene_spearman_mean` |
| `secondary_primary` | `within_gene_kendall_mean` |
| `aggregation` | `mean` over eligible val genes (NOT per-org-balanced default) |
| `tie_handling` | scipy default (`average_ranks`) |
| `bootstrap_n` | 1000 |
| `bootstrap_unit` | gene |
| `ci_level` | 0.95 |
| `ranking_baseline.id` | `per_condition_train_mean` (H-RANK-01) |
| `noise_floor.primary.id` | `per_gene_cross_condition_replicate_spearman` |
| `noise_floor.secondary_proxy.id` | `cross_gene_within_condition_replicate_spearman` |
| `cross_org_drift.source` | `data_contract/splits/multi_org_balanced.yaml` |
| `promotion_delta.within_gene_spearman` | 0.01 (absolute) |
| `promotion_delta.within_gene_kendall` | 0.008 (absolute) |
| `promotion_delta.ci_must_be_disjoint` | true |

**Implementation:** `src/evaluation/ranking_metrics.py` (301 lines).

- `BootstrapMetric` dataclass: `mean, ci_low, ci_high, n_genes_used, n_bootstrap`.
- `within_gene_rank_metric(df, metric="spearman"|"kendall", ...)`:
  per-gene correlation (skipping constants and short genes), mean across
  eligible val genes, bootstrap CI via percentile method over genes with
  replacement. Backed by `_per_gene_correlations` + `_bootstrap_ci`.
- `per_condition_train_mean_predictions(train_df, val_df)`: H-RANK-01
  baseline predictions per val row. Maps each val `condition_key` to the
  mean fit of train genes at that condition (NaN for unseen conditions).
- `ranking_baseline_metrics(...)`: full baseline run — predictions +
  Spearman/Kendall + bootstrap CIs, packaged for the run manifest.
- `model_beats_baseline(model_metric, baseline_metric)`: H-RANK-01 gate;
  returns True iff model mean > baseline mean AND model `ci_low` >
  baseline `ci_high`.
- `task_relevant_noise_floor(val_rows_pre_pool, ...)`: PRIMARY noise floor.
  For each val gene with ≥ 2 replicate `expName`s at ≥ 5 distinct
  `condition_key`s: pair first two replicates per condition, compute
  per-gene Spearman across conditions (rep_A vs rep_B). Return median
  across eligible genes.
- `cross_gene_within_condition_noise_proxy(...)`: SECONDARY noise floor.
  R0 fig 04 definition retained as diagnostic for cross-org comparison
  (computable for more orgs than the task-relevant version).

**Tests:** `tests/unit/test_ranking_metrics.py` (17 tests, all pass).

- Within-gene Spearman/Kendall: perfect correlation → 1.0; anti-correlation
  → -1.0; constant predictions excluded; min_n filter; mean across genes;
  bootstrap CI brackets mean; eligible_mask filters correctly.
- H-RANK-01 baseline: per-condition mean correctness; unknown conditions
  return NaN; full pipeline runs end-to-end.
- Promotion gate: disjoint CIs → True; overlapping → False; NaN-safe.
- Task-relevant noise floor: identical replicates → 1.0; fewer than
  min_conditions paired → gene skipped; counters returned.
- Proxy noise floor: smoke test passes; returns Spearman ∈ (0.5, 1) for
  near-identical replicates.

### Decision
- decision_outcome: **lock** the metric contract above as the R-regime
  evaluation lens.
- rationale:
  1. **Within-gene Spearman + Kendall as co-primary** — Spearman captures
     ordinal agreement; Kendall captures pair-by-pair agreement and is
     more robust to outliers. Both must be reported and both must
     improve for promotion, preventing a model from gaming one metric at
     the other's expense.
  2. **H-RANK-01 baseline = per-condition train mean.** This is the
     "predict the population condition profile for every gene" null. All
     genes share the same prediction vector, but the per-gene Spearman
     correlates that shared profile against each gene's OWN observed fit,
     so it VARIES across genes (NOT constant — corrected 2026-05-25 audit):
     genes tracking the bulk stress response score high, idiosyncratic genes
     low. A model that loses to this baseline has learned no gene-specific
     deviation from the population profile — the entire point of the task.
  3. **Task-relevant noise floor as primary** (per project-lead decision
     2026-05-24). The R0-fig-04 proxy measures cross-gene reproducibility
     within a condition; the primary task is per-gene ranking across
     conditions. The two are correlated but orthogonal: per-condition
     noise (every gene shifts by the same amount) disappears from the
     proxy entirely but degrades within-gene rankings. The proxy is
     retained as a secondary because it's computable for orgs whose
     replicate design doesn't support the task-relevant version
     (e.g., orgs with replicates at < 5 conditions per gene).
  4. **Bootstrap over genes** (not bootstrap-over-conditions, not
     analytical SE) gives a CI that honors the actual unit of analysis
     (each gene contributes one number to the mean). With n=1000 the
     CI half-width on a typical n_genes ≈ 3000 sample is very tight.
  5. **Promotion deltas Δ-Spearman ≥ 0.01 AND Δ-Kendall ≥ 0.008 AND CIs
     disjoint.** The absolute deltas are calibrated to be ~3× the
     bootstrap CI half-width we observed in T7-prep diagnostics; the
     "CIs disjoint" rule is the standard for non-overlapping evidence.
     R1's first run will reveal whether these thresholds are too loose
     or too strict.
- risks_remaining:
  - **Aggregation = global mean (not per-org-balanced)** may let large
    orgs dominate. R-LOCK-4 leaves the per-org-balanced switch on the
    table; if R1's first run reveals one org carrying most of the
    Spearman, R-LOCK-4 can be revised. Sample size at lock time: across
    38 promotion-target orgs, no single org contributes >10% of total
    eligible val genes (estimated from R0 fig 06).
  - **Task-relevant noise floor only computable for orgs with replicate
    designs that cover ≥ 5 distinct conditions per gene with ≥ 2 reps
    each.** From R0 fig 08 + fig 10: ~14 orgs qualify (DvH, Btheta,
    Caulo, etc. — the ones with `frac_cond_with_replicates ≥ 0.7` AND
    enough conditions). For the rest, noise floor is reported as NaN
    in the manifest and the proxy is the only available reference line.
  - **The 0.01 / 0.008 promotion deltas are educated guesses.** They
    should be revisited after R1 lands and we see realistic spread.
  - **Tie handling = average ranks** matters when many genes have
    identical fit values (e.g., flat-fit genes that snuck through the
    filter). The hard val-eligibility on `tail_g ≥ tail_min_org` should
    have already removed those; if not, this is the next failure mode.
- next_action: R1 can start immediately. The first R1 run will populate
  the v2 manifest with real values; R-LOCK-4 may be revised based on
  observed CI widths.

### Reproducibility Attachments
- code_sha: <fill on commit>
- artifacts:
  - `data_contract/ranking/metric_contract.yaml`
  - `src/evaluation/ranking_metrics.py`
  - `tests/unit/test_ranking_metrics.py` (17 tests)
- companion locks:
  - R-LOCK-1-DEC-001 (eligibility — defines what "eligible val genes" means here)
  - R-LOCK-2-DEC-001 (split — defines what "val rows" mean here)
  - R-LOCK-3-DEC-001 (RankingBatch + run manifest v2)
- primary_metric_name: within_gene_spearman_mean
- bootstrap_n: 1000
- ci_level: 0.95
