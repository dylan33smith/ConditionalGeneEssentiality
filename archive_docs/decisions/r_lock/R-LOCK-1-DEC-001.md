## Decision: R-LOCK-1-DEC-001

### Header
- decision_id: R-LOCK-1-DEC-001
- stage_or_tier: R-LOCK-1
- regime: R
- date: 2026-05-24
- owner: project lead
- status: approved
- related_experiments: [R0-A_data_characterization]
- related_hypotheses: [H-R0-01, H-R0-04]
- deferred_hypotheses:
    [H-R0-03 (requires homology mapping; out of R0 scope)]

### Assumption Under Test
- assumption_statement: Eligibility for the ranking regime must (a) use
  **`tail_g = p95 − p5`** of per-gene `fit` as the spread metric — NOT IQR —
  because IQR throws away sparse-but-strong conditional structure (a gene
  essential in 3–5 of 100 conditions has `IQR ≈ 0` by construction but is
  highly rankable); and (b) use a **per-organism** threshold scaled to that
  org's replicate noise floor, because R0 fig 04 showed the noise floor
  varies 5.4× across organisms (median Spearman 0.16–0.89).
- assumption_type: data
- why_it_matters: The eligibility filter and its weighting scheme are the
  first contract the entire R-regime hangs on. Every R-tier inherits them;
  getting them wrong silently biases every downstream metric. The
  tail-vs-IQR distinction in particular is a major correction: under IQR
  we would have *excluded the most rankable genes in the dataset*.

### Pre-Registered Test Plan
- comparison: R0 figures 02 (per-org tail_g + IQR_g histograms), 04 (per-org
  replicate noise floor), 05 (per-org tail-based SNR), 06 (eligibility
  frontier on tail_g) inform a single parameter choice. R-LOCK-1 is a
  data-driven lock, not an A/B promotion.
- fixed_controls: n/a (no model trained)
- metrics_primary: n/a (lock decision)
- promotion_rule:
  - eligibility_policy_passes_h_r0_01: ≥ 30% genes per organism eligible in
    ≥ 80% of organisms at the locked threshold.
  - eligibility_policy_consistent_with_noise: per-org `tail_min` tracks
    per-org noise floor (qualitative check via SNR fig 05).
- failure_guardrails:
  - eligibility_filter_hash_logged: required in run_manifest v2.
  - val_eligibility_independent_of_train_iqr: required (no leakage).

### Evidence Summary

**From R0-A (cf. `research_log/tier_reports/r0_data_characterization.md`):**

- **N = 48 organisms, 182,447 (orgId, gene) pairs, 7,500 expNames, ~4,200
  distinct conditions `(expDesc, media, temperature)`** (R0 figs 01, 08).
- **`tail_g` population distribution** (across all (org, gene), R0 fig 02):
  p25 = 0.376, **p50 = 0.599**, p75 = 1.002.
- **`IQR_g` population distribution** (diagnostic only):
  p25 = 0.142, p50 = 0.223, p75 = 0.363. About 2.7× smaller than tail_g.
- **`m_g` is essentially saturated for promotion-target orgs** (R0 fig 06).
  Frontier shows m_min ∈ {3, 5, 10, 15, 20, 30, 50} produces *identical*
  eligible-gene counts for well-characterized orgs. m_min has real effect
  only in tiny diagnostic-only orgs (Burk376: 10 train conditions; Ralstonia*:
  <10; Magneto; SyringaeB728a). `m_min=10` effectively excludes tiny-org
  genes from training, consistent with R-LOCK-2's diagnostic-only status
  for those orgs.
- **Per-org cross-replicate Spearman varies 5.4×** (R0 fig 04, n=23 orgs):
  - cleanest: Caulo 0.89, Methanococcus_JJ 0.87, ANA3 0.80
  - mid: DvH 0.59, MR1 0.59, Keio 0.57
  - noisiest: Cup4G11 0.16, Koxy 0.34, Btheta 0.39
- **CAVEAT on the noise proxy:** `r_replicate_org_median` is the *cross-gene
  Spearman within a single condition* between replicate assays. This is a
  proxy for the *task-relevant* noise floor (per-gene cross-condition
  Spearman across replicate pairs), which is computed in R-LOCK-4. The two
  are correlated but not identical; the proxy may underestimate
  noise if per-condition noise dominates. R-LOCK-1's coefficient is
  calibrated to the proxy and may be revised after R-LOCK-4 lands.
- **`tail_g` distribution is well-separated from population median** at
  per-org thresholds. At Keio (r=0.57, tail_min_org=0.34), 92% of genes
  are eligible. At Cup4G11 (r=0.16, tail_min_org=0.67), ~50% are eligible
  — which is correct behavior for the noisiest org.
- **H-R0-03 DEFERRED:** R0 fig 07 (low-spread gene Jaccard across orgs) was
  dropped because `gene_key` is org-prefixed in the canonical data
  (e.g. `Keio:14146`), making the Jaccard zero off-diagonal by construction.
  Testing requires the S3-style cross-org homology mapping (0.85 cosine
  cutoff), out of R0 scope. R-LOCK-1's weighting choice does not depend on
  H-R0-03 — it rests on the model-stability argument (see rationale 3 below).

### Decision

- decision_outcome: **lock** eligibility policy below.

```yaml
# data_contract/ranking/eligibility_policy.yaml (proposed)
policy_id: r_lock_1_v1
schema_version: 1

# PRIMARY spread metric — NOT IQR
spread_metric: tail_g                # = p95 − p5 of fit across conditions
spread_metric_diagnostic: iqr_g      # = q75 − q25, retained for analysis only

# Training-target eligibility
m_min: 10                            # defensive floor; no-op for promotion orgs
tail_min_scheme: per_org             # NOT global (R-LOCK-1-DEC-001 rationale)
tail_min_floor: 0.20
tail_min_noise_coef: 0.80
# Per-org formula:
#   tail_min_org = max(tail_min_floor, tail_min_noise_coef × (1 − r_replicate_org_median))
# Examples (from R0 fig 04):
#   Caulo  (r=0.89) → 0.20 (floor)
#   ANA3   (r=0.80) → 0.20 (floor)
#   DvH    (r=0.59) → 0.33
#   Keio   (r=0.57) → 0.34
#   Cola   (r=0.40) → 0.48
#   Cup4G11 (r=0.16) → 0.67
# Orgs without replicate data → use floor 0.20 (caveat in risks below)
tail_min_computed_on: train_rows_only  # no leakage from val

# Weighting (applies on top of filtering)
weighting_scheme: tail_x_m
# w_g = clip((tail_g_train − tail_min_org) / (tail_ref_org − tail_min_org), 0, 1)
#       × min(m_g_train / m_min, 1)
# where tail_ref_org = per-org p75 of tail_g_train

train_eligibility: weighted_all      # all train genes contribute, weighted by w_g
val_eligibility: hard                # hard filter; need stable per-gene Spearman
m_min_val: 5
tail_min_val_scheme: per_org         # same formula, on val rows only

# eligibility_filter_hash = sha256(yaml body above) — written by R-LOCK-3
```

- rationale:
  1. **`tail_g` instead of IQR captures the most rankable genes.** A gene
     essential in 3–5 of 100 conditions has IQR ≈ 0 (the middle 50% of
     conditions is flat) but a tail of ≥ 0.5 in absolute fit. Excluding
     these genes by IQR would discard exactly the targets the per-gene
     ranking task most benefits from. tail_g catches both broadly-conditional
     and sparsely-conditional genes; IQR catches only the broadly-conditional
     subset. Per-gene diagnostic interpretation: high tail + high IQR
     → broadly conditional; high tail + low IQR → sparsely conditional;
     low tail → genuinely non-conditional.
  2. **Per-org thresholds are required by the 5.4× noise spread.** A
     global threshold over-includes Cup4G11 (where 0.30 < replicate noise
     scale, so we'd train on coin-flips) and over-excludes Caulo (where
     0.30 throws away real signal). The `0.80 × (1 − r)` coefficient
     keeps the IQR-equivalent "30% of noise budget" intuition, just
     rescaled by the tail/IQR ratio (~2.7×).
  3. **Weighting train + hard-filtering val** is the right asymmetry. Two
     reasons (not biology — H-R0-03 is unmeasured):
     (a) **Model stability.** A loss that only sees high-spread genes can
     learn to artificially inflate predicted variance. Including low-spread
     genes at low weight teaches a "predict near-flat for non-conditional
     genes" prior, anchoring the variance scale.
     (b) **Val Spearman is undefined when fit is constant**, so val MUST
     be hard-filtered regardless of train policy.
- risks_remaining:
  - **Orgs without replicate data** (25 of 48) get the floor `tail_min=0.20`.
    If any of these are noisy in ways we can't measure, we'll over-include
    their genes. Mitigation: R-LOCK-1 revision after R1's first run reveals
    per-org Spearman; orgs with degenerate metrics get manual threshold bumps.
  - **`r_replicate` is a proxy for the task-relevant noise floor**
    (per-gene cross-condition Spearman across replicate pairs). The proxy
    measures within-condition cross-gene Spearman instead. The two are
    correlated but not identical. The shape of per-org variation is
    preserved; the absolute coefficient (0.80) may need recalibration when
    R-LOCK-4's task-relevant floor lands.
  - **`0.80` and `0.20` are judgment calls** calibrated to a noise proxy.
    Defensible range: coef 0.60–1.00, floor 0.15–0.25. Locking the midpoints;
    R-LOCK-1 revision if R1 first run reveals issues.
  - **`tail_ref_org = p75`** is arbitrary; could be p80 or p90. p75 chosen
    to anchor at "well-above-typical" genes without being dominated by
    outliers. Documented; not blocking.
  - **⚠ TRAIN-ONLY LEAKAGE (audit 2026-05-25, BLOCKER for R1):** the per-org
    `r_replicate` values materialized in `eligibility_policy.yaml` were
    computed on the FULL dataset (R0 ran before the split existed), not
    train-only — contradicting `tail_min_computed_on: train_rows_only` and the
    charter. They drive every per-org `tail_min`. Magnitude is small
    (`r_replicate` is a median over thousands of pairs; val's marginal effect
    is negligible), but it is a real locked-artifact violation. **R1 MUST**
    either (a) recompute `tail_g`/`m_g`/`r_replicate` train-only per fold, or
    (b) accept an explicit waiver with a sensitivity check showing the per-org
    threshold is insensitive to the train/full distinction. Until then the
    eligibility filter is documented but uncoded, so no run has consumed the
    leaky values yet.
- next_action: R-LOCK-3 implements the eligibility-filter hashing +
  `RankingBatch` integration with these weights. R-LOCK-2 was approved in
  parallel. **R1 must resolve the train-only `r_replicate` blocker above.**

### Reproducibility Attachments
- config_snapshot: `configs/experiment/R0-A_data_characterization.yaml`
- split_manifest_id: `r_lock_2_v1_within_org_condition_holdout` (R-LOCK-2-DEC-001)
- preprocessing_artifact_id: `de21504134c84a6c` (T-regime feature contract)
- code_sha: <fill on commit>
- report_path: `research_log/tier_reports/r0_data_characterization.md`
- eligibility_filter_hash: <computed when YAML is materialized in R-LOCK-3>
- sampler_mode: n/a (R-LOCK-3)
- primary_metric_name: n/a (R-LOCK-4)
