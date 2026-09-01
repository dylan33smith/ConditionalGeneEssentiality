# R0 — Data Characterization Report

Source: `R0-A_data_characterization`. See `docs/RPLAN.md §5` for spec.

## Per-organism summary

- Orgs analyzed: 48
- Total (org, gene) pairs: 182,447
- Total experiments: 7491

## IQR summary

- Median IQR_g across all (org, gene): 0.223
- 25th percentile IQR_g: 0.143
- 75th percentile IQR_g: 0.363
- Fraction of (org, gene) with IQR < 0.10: 0.123 (distribution is moderately centered, not a bimodal housekeeping spike — truly non-conditional genes are a small minority)

## Replicate noise floor

- Median cross-replicate Spearman across orgs: 0.506
- Range: [0.171, 0.885]
- Orgs with replicate data: **23 of 48**. The other 25 orgs ran each (expDesc, media) once (no biological replicates) and default to the IQR floor (0.10) in R-LOCK-1.

## Experiment structure (why and what)

**Why this matters.** The ranking task ranks distinct *conditions* within a gene,
but the raw data is at the `expName` (assay) level — and most conditions were
run multiple times as biological replicates (different `expName`s sharing the same
`(expDesc, media)`). Three downstream decisions depend on this structure:

1. **Split protocol (R-LOCK-2):** holding out random `expName`s would silently leak
   conditions into train via their replicate siblings. The split MUST hold out at
   the condition level. The DvH numbers (757 assays → 248 conditions; 99% of
   conditions have replicates) show how severe this would be.
2. **Replicate handling (R-LOCK-3):** train can treat each replicate as an
   independent noisy observation of the same target; val must mean-pool
   replicates before per-gene Spearman or the metric is inflated by intra-condition
   noise.
3. **Eligibility (R-LOCK-1):** `m_g` (conditions per gene) is at the condition
   level, not the `expName` level — otherwise replicate count would inflate
   it and under-filter low-condition genes.

**What's reported (per org, in `08_experiment_structure.csv`):**

- `n_experiments` — distinct assays.
- `n_conditions` — distinct `(expDesc, media)` pairs (the ranking unit).
- `n_media` — distinct media base names (coarser than condition).
- `exp_per_cond_p50, p90, max` — replicate-depth distribution. p50≥2 means
  replication is the norm; p50=1 means most conditions are singletons.
- `frac_cond_with_replicates` — fraction with `exp_per_cond ≥ 2`. High values
  (DvH 0.99, Caulo 0.98, Pedo557 0.93) make the leakage risk above critical;
  low values (Miya 0.22) make the constraint mostly a no-op.

## Deferred

- **H-R0-03** (low-IQR gene overlap across orgs): figure 07 was dropped because `gene_key` is org-prefixed in the canonical data, making the Jaccard zero off-diagonal by construction. Testing this hypothesis requires a cross-org homology mapping (analog of S3's 0.85 cosine cutoff). Not blocking R-LOCK decisions; rationale for `weighted_all` training rests on the model-stability argument (predict-flat genes anchor the loss).

## Candidate eligibility protocols

| metric | m_min | threshold | coverage | median eligible/org |
|---|---|---|---|---|
| tail_g | 3 | 0.10 | 0.92 | 3680 |
| tail_g | 5 | 0.10 | 0.92 | 3680 |
| tail_g | 3 | 0.20 | 0.92 | 3630 |
| tail_g | 5 | 0.20 | 0.92 | 3630 |
| tail_g | 3 | 0.30 | 0.92 | 3323 |
| tail_g | 5 | 0.30 | 0.92 | 3323 |
| tail_g | 3 | 0.40 | 0.92 | 2874 |
| tail_g | 5 | 0.40 | 0.92 | 2874 |
| tail_g | 3 | 0.60 | 0.92 | 2048 |
| tail_g | 5 | 0.60 | 0.92 | 2048 |

## Candidate split protocols (fraction-based holdout)

| split_id | frac | coverage | median val genes/org | median val conditions/org |
|---|---|---|---|---|
| within_org_holdout_frac0.25 | 0.25 | 0.91 | 3676 | 23 |
| within_org_holdout_frac0.30 | 0.30 | 0.91 | 3676 | 27 |
| within_org_holdout_frac0.15 | 0.15 | 0.86 | 3638 | 14 |
| within_org_holdout_frac0.20 | 0.20 | 0.86 | 3638 | 18 |
| within_org_holdout_frac0.10 | 0.10 | 0.82 | 3638 | 9 |

## Figures

See `research_log/figures/r0_data/` for the produced plots:

- 01 m_g per gene per org · 02 IQR_g per gene per org · 03 joint (m_g, IQR_g)
- 04 replicate noise floor · 05 SNR · 06 eligibility frontier
- (07 dropped — see Deferred section)
- 08 experiment structure · 09 condition discriminability
- 10 split feasibility · 11 expGroup coverage

## Next

- R-LOCK-1 selects an eligibility candidate (or proposes a new one).
- R-LOCK-2 selects a split candidate (likely refined with stratification).
- R-LOCK-3 implements the RankingBatch contract.
- R-LOCK-4 finalizes metric + baseline + noise-floor reporting.