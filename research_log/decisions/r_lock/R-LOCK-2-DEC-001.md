## Decision: R-LOCK-2-DEC-001

### Header
- decision_id: R-LOCK-2-DEC-001
- stage_or_tier: R-LOCK-2
- regime: R
- date: 2026-05-24
- owner: project lead
- status: approved
- related_experiments: [R0-A_data_characterization]
- related_hypotheses: [H-R0-02]

### Assumption Under Test
- assumption_statement: A **fraction-based, within-organism, condition-level**
  holdout (with replicate-group-together constraint and `expGroup`
  stratification where feasible) preserves enough val signal in ≥86% of
  promotion-target organisms to compute stable per-gene Spearman, AND avoids
  replicate-induced leakage that would silently inflate scores.

  **Primary generalization claim (PROJECT-LEAD DECISION 2026-05-24):**
  "novel condition combinations within a known organism." Holdout conditions
  share chemistries with train conditions; the model must compose. This is
  the deployment-relevant claim. Cell-holdout is added as a diagnostic to
  bound how much of the gap is from condition-novelty vs raw signal limits;
  stressor-class-holdout is the stress diagnostic.
- assumption_type: split
- why_it_matters: The split protocol defines what the R-regime *claims*. Get
  the unit (`condition` vs `expName`) or the grouping (`with-replicate-leak`
  vs `replicate-group-together`) wrong and every R-tier metric is meaningless.
  This is the analog of S3 for the ranking regime.

### Pre-Registered Test Plan
- comparison: R0 analyses J (fig 10, fraction-based split feasibility) and K
  (fig 11, expGroup stratification feasibility) — both are data-driven locks,
  not model A/B tests.
- fixed_controls: n/a (no model)
- promotion_rule:
  - n_orgs_pass_at_chosen_frac: ≥ 38/44 (≥ 86%) with ≥ 100 val-eligible genes.
  - n_val_conditions_per_org_p25: ≥ 10 (so per-gene Spearman has ≥10 data
    points in most orgs).
  - stratification_feasible_in_promotion_orgs: required for orgs with
    `n_expgroups ≥ 2`.
  - replicate_leakage: forbidden — every replicate of a held-out condition
    must be in val (the constraint that *fraction-based* alone doesn't enforce).
- failure_guardrails:
  - cross_org_drift_diagnostic: required (run T-regime `multi_org_balanced` on
    same checkpoint to detect R-regime regressions of T-regime generalization).
  - stressor_class_holdout_diagnostic: required (one held-out chemical class
    per org, evaluated separately, not gated).

### Evidence Summary

**From R0-A (figures 08, 10, 11; CSVs in `research_log/figures/r0_data/`):**

1. **Org-size spread is 17×** (R0 fig 08 updated):
   - DvH: 757 expNames → 248 conditions (replicate factor 3.05)
   - Btheta: 499 → 273 (rf 1.83)
   - Keio: 168 → 111 (rf 1.51)
   - Burk376: 30 → 13 (rf 2.3)
   - RalstoniaGMI1000: 7 → ? (likely ≤5 conditions)
   - Fraction-based holdout (vs fixed-k) is required to handle this spread.

2. **Replicate factor is large and uneven.** Most orgs have replicate factor
   1.5–3 (each condition observed ~2–3 times as different expNames). DvH has
   3× and ~98% of conditions have ≥2 replicates. Random expName-level holdout
   would leak ~half the val conditions into train via their replicate
   siblings. **Condition-level (`expDesc, media`) holdout closes this leak.**

3. **Split feasibility at fraction-based holdouts** (R0 fig 10, post-fix):

   | frac | n_orgs_pass (≥100 val genes) | median val conditions/org | median val genes/org |
   |---|---|---|---|
   | 0.10 | <38/44 | 9 | sub-threshold |
   | 0.15 | 38/44 | 14 | 3,638 |
   | **0.20** | **38/44** | **18** | **3,638** |
   | 0.25 | 40/44 | 23 | 3,677 |
   | 0.30 | 40/44 | 27 | 3,677 |

   - **0.20 hits the promotion gate** (38/44 = 86%; ≥10 val conditions in
     every passing org).
   - The 4–6 failing orgs are tiny (Burk376: 13 conditions total — even 50%
     holdout doesn't help; Ralstonias: 5–8 conditions; Magneto: 18). These
     would be excluded from primary scoring regardless; they remain
     diagnostic-only.
   - 0.25 buys 2 more orgs at the cost of 5 fewer train conditions per org —
     marginal. User pre-locked 0.20 as the value to use.

4. **Stratification feasibility** (R0 fig 11, new):
   - 41/47 orgs have `stratification_feasible = True` at 20% holdout with
     2-per-group minimum.
   - Failing orgs (`stratification_feasible = False`):
     RalstoniaGMI1000, RalstoniaBSBF1503, RalstoniaPSI07, RalstoniaUW163
     (each: 3 expGroups but smallest group <10), SyringaeB728a_mexBdelta,
     Magneto, Burk376 (1 group). All are diagnostic-only orgs by point 3.
   - **For promotion-target orgs, stratification is always feasible.**
     Largest orgs: DvH (5 groups, smallest 17), Btheta (8 groups, smallest 1
     but 4 pass), Keio (6 groups, smallest 2 but 3 pass), MR1 (7 groups,
     smallest 3).

### Decision

- decision_outcome: **lock** split protocol below.

```yaml
# data_contract/ranking/split_protocol.yaml (proposed)
protocol_id: r_lock_2_v1_within_org_condition_holdout
schema_version: 1

holdout_unit: condition_key       # = (expDesc, media, temperature), NOT expName
                                  # Temperature added 2026-05-24 after R0 audit
                                  # found 53 of 2246 (2.4%) (expDesc, media) groups
                                  # contained assays at distinct temperatures
                                  # that we were incorrectly grouping as replicates.
holdout_scope: within_organism

# Fraction-based, with floor and ceiling
holdout_fraction: 0.20
min_holdout: 3                    # at least 3 val conditions even for tiny orgs
max_holdout: 30                   # cap for very large orgs

# Replicate handling — already enforced by holdout_unit=condition_key, but documented:
replicate_group_constraint: hold_out_together
# Operationally: choose 20% of distinct (expDesc, media) keys per org;
# ALL expNames with each chosen key go to val. No replicate leakage.

# Stratification
stratify_by: expGroup
stratification_min_groups: 2
stratification_min_group_size: 10  # per R0 fig 11: 2-per-group at 20% needs ≥10 in source
stratification_fallback: random_within_org

# Diagnostic splits (run alongside primary; NOT gated for promotion)
diagnostic_splits:
  - id: cell_holdout
    description: random (gene, condition) cells held out from known conditions
    note: |
      Easy-ceiling diagnostic. Held-out cells' conditions are present in
      train (for other genes), so the model can look up condition behavior
      from siblings. Bounds "how much of the gap is condition-novelty
      vs raw signal limits." If primary (condition_holdout) Spearman
      is far below cell_holdout Spearman, condition-novelty is the bottleneck;
      if they're similar, signal limits dominate.
    holdout_fraction: 0.20             # match primary; reuse the gene set
    seed: 0
  - id: stressor_class_holdout
    description: hold out one entire chemical class per org per fold
    n_folds: 1
    note: hard-floor diagnostic; tests generalization to novel chemical families
  - id: cross_org_balanced_drift
    description: T-regime multi_org_balanced as drift monitor
    source: data_contract/splits/multi_org_balanced.yaml
    note: predicted, not trained on — same R-regime checkpoint scored cross-org

# Seeds
primary_seed: 0
multi_seed_check: [0, 1, 2]

# Per-org eligibility for PRIMARY (vs diagnostic) scoring
primary_orgs: <all orgs passing split feasibility AND stratification AND R-LOCK-1>
diagnostic_orgs: <complement: Burk376, Ralstonia*, Magneto, SyringaeB728a_mexBdelta, ...>

# Dev subset (RPLAN §6)
dev_subset_org: Keio
dev_subset_seed: 0
# Selection rationale: median data volume (3789 genes × 168 expNames = 637k;
# rank 12 of 23 orgs with replicate data); mid-range replicate noise
# (r=0.57); stratification feasible (6 expGroups, 3 pass threshold);
# biologically iconic (E. coli K-12 BW25113) for results readability.
```

- rationale:
  1. **Condition-level holdout closes the replicate-leakage hole.** Per fig 08,
     DvH has ~98% of conditions with replicates; random expName-level holdout
     would silently inflate Spearman by leaking val conditions into train.
     Locking `holdout_unit: condition_key` makes this impossible by construction.
  2. **20% fraction was user-pre-locked**; R0 fig 10 confirms it meets the
     promotion gate (38/44 = 86%) and delivers ≥18 median val conditions per
     promotion-target org — comfortably above the m_val_min=5 from R-LOCK-1.
  3. **Stratification by `expGroup`** is the standard ML hedge against
     condition-class skew in the val set. R0 fig 11 confirms it's feasible
     for every promotion-target org (largest orgs are richest in expGroups);
     orgs that fail are already excluded from primary by org-size criteria.
  4. **Dev-subset = Keio.** Picked because:
     - **Median data volume** (rank 12 of 23 orgs with replicate data) →
       results generalize across the org-size spectrum.
     - **Mid-range replicate noise** (r=0.57, vs floor 0.16 and ceiling 0.89)
       → not an outlier on the noise dimension either.
     - **Stratification feasible** (6 expGroups, 3 pass the threshold) so any
       stratification bug surfaces in dev, not in promotion runs.
     - **Biologically iconic**: E. coli K-12 BW25113 (the Keio collection
       reference) is the most-studied bacterial genome; results are
       interpretable to a broad audience and any pathology will be visible
       to anyone reading them.
     - **Not too big** (168 expNames; iteration cost is bounded) and **not too
       small** (enough conditions to compute Spearman meaningfully).
- risks_remaining:
  - **Min-holdout floor of 3** for tiny orgs may not be enough to compute
    meaningful val Spearman in those orgs, but they're diagnostic-only by
    design.
  - **Stratification could oversample very small expGroups** in some orgs;
    fallback to random within-org is implemented. R0 didn't measure the
    distortion magnitude — if R1 first run shows degenerate val behavior in
    any stratified org, R-LOCK-2 revision swaps it to random.
  - **Cross-replicate noise floor (R-LOCK-4 deliverable)** computed on the
    final locked val rows — not on a generic baseline. Until R-LOCK-4 lands,
    R0 fig 04's *per-org cross-experiment* Spearman is the proxy.
- next_action: R-LOCK-3 (RankingBatch + sampler default + manifest v2) and
  R-LOCK-4 (metric + ranking baseline + noise-floor protocol) can both
  proceed; they consume this protocol.

### Reproducibility Attachments
- config_snapshot: `configs/experiment/R0-A_data_characterization.yaml`
- split_manifest_id: <generated when YAML is materialized>
- preprocessing_artifact_id: `de21504134c84a6c`
- code_sha: <fill on commit>
- report_path: `research_log/tier_reports/r0_data_characterization.md`
- eligibility_filter_hash: <R-LOCK-1 hash, once finalized>
- sampler_mode: n/a (R-LOCK-3)
- primary_metric_name: n/a (R-LOCK-4)
