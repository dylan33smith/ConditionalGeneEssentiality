## Decision: <decision_id>

### Header
- decision_id: <e.g., T1-DEC-001 or S3-DEC-001>
- stage_or_tier: <S0|S1|S2|S3|S4|S5|T1|T2|T3|T4|T5|T6|R0|R-LOCK-1|R-LOCK-2|R-LOCK-3|R-LOCK-4|R1|R2|R3>
- regime: <T|R>     # T = pointwise MSE/MAE (REFACTORPLAN); R = ranking (RPLAN)
- date: <YYYY-MM-DD>
- owner: <name>
- status: <proposed|approved|rejected|superseded>
- related_experiments: [<T1-A_s0>, <T1-A_s1>, <T1-A_s2>]
- related_hypotheses: [<H-ENC-01>]

### Assumption Under Test
- assumption_statement: <single falsifiable statement>
- assumption_type: <data|split|representation|fusion|architecture|optimization|evaluation>
- why_it_matters: <1-2 lines tied to conditional essentiality goal>

### Pre-Registered Test Plan
- comparison: <exact A vs B (vs C)>
- fixed_controls:
  - split_protocol_id: <from data_contract/splits/locked_protocol.yaml>
  - feature_contract_id: <from data_contract/feature_contract.yaml>
  - quality_policy_id: <from data_contract/policy/quality_policy.yaml>
  - seed_set: [0, 1, 2]
  - training_budget: <epochs / steps>
  - code_sha: <commit hash>
- metrics_primary:
    # T-regime: [rmse, mae]
    # R-regime: [within_gene_spearman, within_gene_kendall]
    <list>
- metrics_secondary:
    # T-regime: [within_gene_spearman, per_organism_rmse_spread]
    # R-regime: [rmse, mae, cross_org_within_gene_spearman, per_organism_spearman_spread]
    <list>
- promotion_rule:
    # T-regime fields:
    # - rmse_improvement: <delta>
    # - mae_improvement: <delta>
    # - additive_baseline_gate: required           # H-BASE-01
    # R-regime fields:
    # - spearman_improvement: <delta>
    # - kendall_improvement: <delta>
    # - ranking_baseline_gate: required            # H-RANK-01 (per-cond mean)
    # - noise_floor_reported: required
    <fields>
- failure_guardrails:
  - leakage_checks_pass: required
  - split_overlap_pass: required
  - manifest_validation_pass: required
  - reproducibility_check_pass: required

### Evidence Summary
- run_manifest_ids: [<run1>, <run2>, <run3>]
- sample_sizes:
  - rows_scored: <int>
  - genes_eligible: <int>
  - organisms_scored: <int>
- result_summary (mean ± std across seeds):
  - A: rmse=<>, mae=<>, spearman=<>, beats_additive=<bool>
  - B: rmse=<>, mae=<>, spearman=<>, beats_additive=<bool>
- null_baseline_deltas (model − baseline, negative = model better):
  - global_train_mean:  <delta_rmse>, <delta_mae>
  - per_condition_mean: <delta_rmse>, <delta_mae>
  - per_organism_mean:  <delta_rmse>, <delta_mae>
  - additive_baseline:  <delta_rmse>, <delta_mae>
  - embedding_nn:       <delta_rmse>, <delta_mae>
- statistical_check: <CI or seed-std summary>
- quality_checks:
  - unknown_category_rate: <value>
  - leakage_checks: <pass/fail>
  - split_overlap_checks: <pass/fail>

### Decision
- decision_outcome: <promote A | promote B | no winner>
- rationale: <evidence-based short explanation>
- risks_remaining: <open risks>
- next_action: <exact follow-up carried into next stage/tier>

### Reproducibility Attachments
- config_snapshot: <path>
- split_manifest_id: <id>
- preprocessing_artifact_id: <id>
- code_sha: <sha>
- report_path: <path>
- # R-regime only:
- # eligibility_filter_hash: <hash>
- # sampler_mode: <pointwise|pairwise|listwise>
- # primary_metric_name: <within_gene_spearman_mean|within_gene_spearman_per_org_balanced>
