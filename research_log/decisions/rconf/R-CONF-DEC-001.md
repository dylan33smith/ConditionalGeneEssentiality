## Decision: R-CONF-DEC-001

### Header
- decision_id: R-CONF-DEC-001
- stage_or_tier: R-CONF
- regime: R
- date: 2026-06-06
- owner: project lead
- status: approved
- approved_date: 2026-06-06
- related_experiments: [R-CONF_confidence_strat]
- related_hypotheses: [H-R-CONF-01]
- type: CHARACTERIZATION (not a promotion gate)
- metrics_primary: [ndcg_at_5, within_gene_spearman] (R-LOCK-4 v2)
- baseline_gate: chemistry_knn (reference only; characterization)
- eligibility_filter_hash: r_lock_1_v1

### Assumption Under Test
- assumption_statement: H-R-CONF-01 — measurement label NOISE (not structure)
  explains the deep model's deficit relative to the chem-kNN gate. If true, then
  (a) on well-measured rows the model should catch kNN, and/or (b) confidence-
  weighting the training labels should help. Confidence signal = `abs_t`, the
  Wetmore-et-al-2015 moderated t already in the canonical fitness table
  (|t|>4 = the field-standard "real effect" threshold).
- assumption_type: data-quality / evaluation
- why_it_matters: The central project finding is a NEGATIVE (no global/hybrid model
  beats local chem-kNN). A reviewer's first objection is "your labels are noisy —
  the model is fine, the metric is just measuring noise." R-CONF directly tests
  whether the negative is a noise artifact or structural, and bounds how much of
  the model–ceiling gap is irreducible biological noise.

### Method
Full eval: 23 replicate orgs, 3 seeds, locked base (AdapterResidualMLP + weighted
Huber + multihot_425), eligible val gene set, denominator parity vs chem-kNN.
1. **Confidence-stratified eval (primary).** Bin eligible val GENES into quartiles
   by median per-cell `abs_t` (Q1 low → Q4 high confidence). Per stratum: model /
   chem-kNN / chem-NULL within-gene Spearman + NDCG@5, AND the biological-replicate
   noise-floor CEILING restricted to that stratum's genes.
2. **Cell-level high-|t| filter (companion).** Restrict each gene's ranked
   conditions to cells with abs_t ≥ {0, 2, 4}; recompute on surviving (≥5-cell)
   genes. NOTE: selection-biased (well-measured cells are big-effect) — reported
   as a companion, not the headline.
3. **t-weighted training (secondary lever).** Retrain the base with per-row loss
   weight w_g · f(abs_t), f a saturating ramp to 1 at |t|=4; compare to w_g-only.

### Evidence (full val, 23 orgs, 3 seeds, seed-averaged)

**1. Confidence-stratified — kNN beats the model in EVERY stratum; gap narrows but never closes.**

| stratum | median \|t\| | chem-NULL | deep model | chem-kNN | replicate ceiling | gap (kNN−model) |
|---|---|---|---|---|---|---|
| Q1 (low)  | 0.52 | 0.294 | 0.359 | 0.414 | 0.586 | **0.055** |
| Q2        | 0.70 | 0.299 | 0.375 | 0.431 | 0.562 | **0.056** |
| Q3        | 0.93 | 0.321 | 0.423 | 0.473 | 0.626 | **0.050** |
| Q4 (high) | 2.06 | 0.467 | 0.593 | 0.633 | 0.835 | **0.040** |
*(NDCG@5; ~9.8k genes/stratum.)* Spearman tells the same story (Q4: model 0.300 vs kNN 0.345).

**2. Cell-level high-|t| filter — kNN stays ahead even on the cleanest cells.**

| abs_t threshold | n_genes | model NDCG@5 | kNN NDCG@5 | gap |
|---|---|---|---|---|
| ≥ 0 (all) | 39,356 | 0.435 | 0.485 | 0.050 |
| ≥ 2 | 8,427 | 0.768 | 0.789 | 0.021 |
| ≥ 4 | 3,276 | 0.825 | 0.835 | 0.010 |

**3. t-weighted training — does not help.** Mean Δ(conf−base): NDCG@5 −0.0036,
Spearman −0.0051 (slightly hurts), consistent across all 3 seeds.

Figure: `research_log/figures/r_conf/01_confidence_stratified.png`.

### Decision
- decision_outcome: **CHARACTERIZATION FINDING — the negative is structural, not a
  label-noise artifact. H-R-CONF-01 NOT supported.**
- findings:
  1. **kNN > model in every confidence stratum and at every cell-filter threshold.**
     Even on the highest-confidence genes (Q4, |t|~2) and the cleanest cells
     (|t|≥4), chem-kNN still wins. Denoising does not flip the ranking → the
     deep model's deficit is structural (local-vs-global), as established across
     R1 / capacity / R-LOSS / R-HYBRID.
  2. **Label noise DOES depress measured performance — and is partly the metric.**
     Every method and the replicate ceiling rise monotonically with confidence
     (ceiling 0.59→0.83). The model–kNN gap narrows with confidence (0.055→0.040
     stratified; 0.050→0.010 cell-filtered), so noise explains PART of the model's
     deficit — but never enough to close it.
  3. **A large share of the model–ceiling gap is irreducible biological noise.**
     In Q1 the ceiling is only 0.59; the task is intrinsically noisy at the
     low-confidence end. The achievable target rises with measurement quality.
  4. **Confidence-weighted training is not a lever** (Δ≈−0.004) — consistent with
     R1/R-LOSS: the bottleneck is not what the global model fits, it is that a
     global model cannot reproduce each gene's local idiosyncratic history.
  5. **Selection-bias caveat is real and logged.** The cell-filter's headline
     gains come almost entirely from discarding genes (39k→3.3k) and isolating
     big-effect conditions, NOT from the model improving. (Keio-only dev had the
     model edge ahead at |t|≥4; this did NOT survive multi-org — another single-
     org-ceiling artifact.)
- rationale: This is the rigor the characterization paper needs. It pre-empts the
  "your labels are just noisy" objection with data: the negative holds at every
  measurement-confidence level, while honestly quantifying that noise sets a
  modest, confidence-dependent ceiling and accounts for part (not all) of the gap.
- next_action:
  1. Fold the finding into SCIENTIFIC_SYNTHESIS (§6 durable findings + a noise
     subsection). DONE in this commit.
  2. Use the stratified figure + table as a characterization-paper exhibit
     ("performance vs measurement confidence; the negative is noise-robust").
  3. Remaining optional characterization measurement: the cold-gene diagnostic
     (how far kNN degrades on unseen genes — the only regime a global model could
     help). Not a gate.

### Reproducibility
- code: src/experiments/rconf/_rconf_common.py, src/experiments/rconf/run.py;
  abs_t carried via src/experiments/r1/_r1_common.py (prepare_r1_data).
- tests: tests/unit/test_rconf.py (6 tests); full unit suite green (161 passed).
- config: configs/experiment/R-CONF_confidence_strat.yaml (23 orgs, seeds 0/1/2,
  8 epochs, 4 strata, |t| cap 4.0).
- artifacts: artifacts/runs/rconf/{stratified.csv, cell_filter.csv,
  tweight_overall.csv, full_run.log}; figure research_log/figures/r_conf/01_*.png.
- gate reference: chem-kNN (k=5), denominator parity on the eligible val gene set.
