## Decision: R-LOCK-4-DEC-002

### Header
- decision_id: R-LOCK-4-DEC-002
- stage_or_tier: R-LOCK-4
- regime: R
- date: 2026-05-25
- owner: project lead
- status: proposed
- supersedes: parts of R-LOCK-4-DEC-001 (baseline, metric set, promotion delta, bootstrap)
- related_experiments: [materialized split + noise-floor computation 2026-05-25]
- related_hypotheses: [H-RANK-01 (revised)]

### Assumption Under Test
- assumption_statement: The R-LOCK-4-DEC-001 metric contract needs revision on
  five points surfaced by (a) the publication-readiness audit and (b) computing
  the baseline/noise-floor on the real materialized split: (1) baselines must be
  SPLIT-SPECIFIC because the primary condition-holdout split has COLD columns;
  (2) NDCG@k/precision@k retrieval metrics are valued above full-list Spearman
  for the "top stressors" use case; (3) the promotion delta must be anchored to
  the improvable gap, not an arbitrary absolute; (4) the bootstrap must be
  hierarchical (org→gene); (5) a multiple-comparison (FDR) budget and required
  per-organism breakdown are added.
- assumption_type: evaluation
- why_it_matters: The DEC-001 H-RANK-01 baseline is empirically BROKEN for the
  primary split (returned NaN for all 28,005 val genes). Promotion decisions
  built on a broken baseline and an over-confident flat bootstrap would be
  invalid. This must be fixed before any R-tier promotion.

### Evidence Summary

**THE COLD-COLUMN DISCOVERY (empirical, 2026-05-25):**
- Materialized the locked R-LOCK-2 condition-holdout split on real data.
- **Verified: val conditions are 100% disjoint from train** (0 of 78 val
  conditions appear in train, on DvH/Btheta/Caulo). This is by construction —
  condition-holdout removes whole condition columns.
- **Consequence 1:** `per_condition_train_mean` (DEC-001 H-RANK-01) is undefined
  for the primary split — confirmed NaN for all val genes.
- **Consequence 2:** pure matrix factorization is invalid for the primary split
  (cannot place a cold column with zero observed entries).
- **Consequence 3:** the "transductive matrix completion" critique applies to
  the cell_holdout DIAGNOSTIC (warm columns), not the primary. The primary is
  cold-start / inductive matrix completion — it genuinely requires chemistry
  side-features, which is a more defensible setup.

**Task-relevant noise floor (measured):** median per-gene cross-condition
replicate Spearman = **0.358** (8 high-replicate orgs, 24,337 val genes;
mean 0.361). Supersedes the 0.43/0.506 cross-gene proxy as the true ceiling.

### Decision

**1. SPLIT-SPECIFIC BASELINES** (see `metric_contract.yaml` v2):
- Primary (condition-holdout, cold columns): gate = `chemistry_nearest_condition_profile`
  (population profile transferred via chemistry); competitive = `chemistry_knn`.
  MF and per-condition-mean are INVALID here.
- cell_holdout diagnostic (warm columns): MF + per-condition-mean valid — this
  is where the matrix-completion baseline lives.
- cold_gene diagnostic (warm columns, cold rows): per-condition-mean valid;
  additive two-factor valid.

**2. METRICS** — retrieval family valued above full-list correlation:
- retrieval-primary: NDCG@k, precision@k for k ∈ {1,3,5}; relevance =
  max(0, −fit) (stressors reduce fitness). Headline for "find the top stressors."
- full-list co-primary (completeness/anti-gaming): within-gene Spearman + Kendall.
- Promotion on NDCG@5 + Spearman; must not regress Kendall.

**3. EFFECT-SIZE-ANCHORED PROMOTION DELTA:**
- Δ ≥ 0.15 × (noise_floor − baseline) on the locked val, NOT an absolute 0.01.
- **Blocked:** requires the (corrected, chemistry-based) baseline value, which
  could not be computed until the baseline definition was fixed (point 1).
  Concrete delta set once `chemistry_nearest_condition_profile` is implemented
  and scored. Noise floor side is known (0.358).

**4. HIERARCHICAL BOOTSTRAP** (org→gene): resample orgs with replacement, then
genes within. Replaces the flat gene bootstrap (over-confident — genes within an
org are correlated). Implemented in `ranking_metrics.py` (`within_gene_rank_metric`
gains a `hierarchical` mode + an `orgId` column).

**5. FDR BUDGET + PER-ORG BREAKDOWN:**
- Benjamini-Hochberg across a tier's arms; pre-registered α-budget across the
  full tier sequence; the per-lock "revise after R1" clauses collapse to ONE
  recalibration checkpoint.
- Per-organism breakdown table required every val scoring (cheap — group
  per-gene scores by org before the global mean).

- decision_outcome: **proposed** — adopt the v2 metric contract; resolve the
  blocked promotion-delta number once the chemistry baseline is implemented.
- risks_remaining:
  - The chemistry-based null + chemistry-kNN require a chemistry distance matrix
    (fingerprint Tanimoto / multihot Jaccard) — new code for R1.
  - Implementations of hierarchical bootstrap, NDCG@k, FDR, per-org table are
    declared but not yet coded (R-LOCK-4 follow-up before R1 scores anything).
- next_action: implement (a) chemistry distance + the two chemistry baselines,
  (b) NDCG@k/precision@k helpers, (c) hierarchical bootstrap, (d) per-org table,
  (e) FDR helper — then compute the baseline value and set the concrete delta.

### Reproducibility Attachments
- code_sha: <fill on commit>
- artifacts: `data_contract/ranking/metric_contract.yaml` (v2),
  `src/data/datasets/build_ranking_split.py`, noise-floor computation 2026-05-25.
- primary_metric_name: ndcg_at_5 (retrieval) + within_gene_spearman_mean (completeness)
