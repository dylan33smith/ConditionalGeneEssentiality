## Decision: <decision_id>

### Header
- decision_id: <e.g., T1-DEC-003>
- tier: <PreTier|T1|T2|T3|T4>
- date: <YYYY-MM-DD>
- owner: <name>
- status: <proposed|approved|rejected|superseded>
- related_experiments: [<T1-E1A>, <T1-E1B>]

### Assumption Under Test
- assumption_statement: <single falsifiable statement>
- assumption_type: <data|split|representation|fusion|architecture|optimization|evaluation>
- why_it_matters: <1-2 lines tied to conditional essentiality goal>

### Pre-Registered Test Plan
- comparison: <exact A vs B, or A/B/C>
- fixed_controls:
  - split protocol id
  - seed set
  - training budget
  - loss/eval code version
- metrics_primary: [RMSE]
- metrics_secondary: [within-gene Spearman, MAE]
- promotion_rule:
  - primary threshold: <explicit>
  - secondary non-degradation tolerance: <explicit>
- failure_guardrails:
  - leakage test pass required
  - split integrity pass required
  - data contract consistency pass required

### Evidence Summary
- run_manifest_ids: [<run1>, <run2>, <run3>]
- sample_sizes:
  - rows_scored: <int>
  - genes_scored: <int>
  - organisms_scored: <int>
- result_summary:
  - RMSE: <A value> vs <B value>
  - Spearman: <A value> vs <B value>
  - MAE: <A value> vs <B value>
- statistical_check: <mean±std or CI across seeds>
- quality_checks:
  - unknown_category_rate: <value>
  - leakage_checks: <pass/fail>
  - split_overlap_checks: <pass/fail>

### Decision
- decision_outcome: <promote A|promote B|no winner>
- rationale: <evidence-based short explanation>
- risks_remaining: <open risks>
- next_action: <exact follow-up carried into next tier>

### Reproducibility Attachments
- config_snapshot: <path_or_id>
- split_manifest_id: <id>
- preprocessing_artifact_id: <id>
- code_sha: <sha>
- report_path: <path>
