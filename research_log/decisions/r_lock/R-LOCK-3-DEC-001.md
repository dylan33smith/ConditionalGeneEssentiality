## Decision: R-LOCK-3-DEC-001

### Header
- decision_id: R-LOCK-3-DEC-001
- stage_or_tier: R-LOCK-3
- regime: R
- date: 2026-05-25
- owner: project lead
- status: approved
- related_experiments: []
- related_hypotheses: []

### Assumption Under Test
- assumption_statement: A single `RankingBatch` dataclass with three sampler
  modes (pointwise / pairwise / listwise) sharing one model forward signature
  `(gene_emb, cond_feat) -> scalar` is sufficient for every R-tier we plan to
  run. Pointwise is the default for R1, R2 (matches T-regime training loop;
  zero code-path change for existing model). Pairwise and listwise are
  reserved for R-LOSS where the sampler mode is itself the test axis.
- assumption_type: data
- why_it_matters: Without a single batch contract, every R-tier would
  reinvent its own data layout and metric code paths, making cross-tier
  comparison impossible. R-LOCK-3 is the analog of S4 (feature contract)
  for the ranking regime.

### Pre-Registered Test Plan
- comparison: n/a — this is an engineering lock, not an A/B test.
- promotion_rule:
  - `RankingBatch` dataclass + 3 samplers + 3 collate functions all unit-tested.
  - `policy_hash` is deterministic across runs and changes with content.
  - Run-manifest v2 schema is JSON-Schema-valid and includes all
    R-regime-only fields listed in RPLAN §2.6.
- failure_guardrails:
  - All R-regime runs must log `regime: "R"` and validate against v2.
  - All R-regime runs must log `eligibility_filter_hash` matching the
    sha256 of the materialized R-LOCK-1 policy YAML.

### Evidence Summary

**Implementation:** `src/data/datasets/ranking_batch.py` (265 lines).

- `RankingBatch` dataclass with mode-dependent validators (pairwise requires
  `sign`, listwise requires `mask`).
- `PointwiseSampler` — yields row indices in shuffled order; epoch-varying
  seed; supports the existing pointwise MSE/Huber training loop with zero
  code change.
- `PairwiseSampler` — per gene with ≥ 2 conditions, samples
  `pairs_per_gene` ordered condition pairs per epoch. Sign convention:
  `+1` if `fit_i > fit_j` (item i is less essential than j); locked here
  so margin losses don't need to re-derive direction.
- `ListwiseSampler` — per gene, yields its full set of row indices (padded
  to max list length); collate produces (B, L) tensors with mask.
- `collate_*` functions transform sampler output into `RankingBatch`
  instances; all return CPU tensors (train loop handles device move).
- `policy_hash(path)` returns sha256 of YAML body bytes — used as
  `eligibility_filter_hash` field in run manifest v2.

**Schema:** `data_contract/schemas/run_manifest_v2.schema.json` (168 lines).

Extends v1. New required fields (per RPLAN §2.6):

- `regime: "R"`
- `sampler_mode: pointwise | pairwise | listwise`
- `ranking_loss: pointwise_mse | pointwise_huber | pairwise_margin | listmle | softrank | approxndcg`
- `eligibility_filter_hash` (sha256, required)
- `split_protocol_sha256` and `metric_contract_sha256` (under `data_contract_checksums`)
- `primary_metric_name = "within_gene_spearman_mean"`
- `metrics_ranking.within_gene_spearman` and `.within_gene_kendall` (bootstrap_metric objects)
- `ranking_baseline` with `baseline_id`, `spearman_mean`, `kendall_mean`, `model_beats`
- `noise_floor` with `primary_value`, `n_genes_used`, optional `proxy_value`
- `cross_org_drift` (optional, non-gated)
- `leakage_checks` extended with `eligibility_train_only_pass` and `replicate_group_intact_pass`

**Tests:** `tests/unit/test_ranking_batch.py` (15 tests, all pass).

- Dataclass validators reject malformed batches.
- Samplers respect gene grouping (pairwise pairs within same gene; listwise
  yields one list per gene); shuffle changes across epochs; build_sampler
  factory routes correctly.
- Collate functions produce correctly-shaped tensors, correct sign
  convention for pairwise, and correct mask for listwise padding.
- `policy_hash` is deterministic and content-sensitive.

**Materialized policy artifacts:**

- `data_contract/ranking/eligibility_policy.yaml` (R-LOCK-1 in YAML form)
- `data_contract/ranking/split_protocol.yaml` (R-LOCK-2 in YAML form)
- `data_contract/ranking/metric_contract.yaml` (R-LOCK-4 in YAML form;
  approved alongside this in R-LOCK-4-DEC-001)

### Decision
- decision_outcome: **lock** the RankingBatch contract + run manifest v2 + the
  three materialized policy YAMLs as the data contract for the R-regime.
- rationale:
  1. **One model forward signature across modes** keeps R1/R2 (chemistry,
     fusion) trivially compatible with the T-regime training loop. The
     sampler change happens above the model.
  2. **Pointwise default** preserves all of T5-A's training behavior, so R1
     can immediately reuse the existing optimizer/LR/loss code with only
     the loss-target switching from `MSE(pred, fit)` to `MSE(pred, fit)`
     (literally identical — pointwise IS MSE in the ranking context, with
     the gene-weight `w_g` from R-LOCK-1 multiplied in).
  3. **Pairwise sign convention locked here** so R-LOSS doesn't need to
     rederive whether `+1` means "i less essential" or "i more essential".
     The choice (`sign = sign(fit_i − fit_j)`) follows from the R-LOCK-2
     convention "low fit = essential = should rank low."
  4. **Eligibility-filter hashing in the manifest** makes cross-tier metric
     comparability auditable. If the eligibility YAML changes (intentionally
     or accidentally), the hash changes and downstream comparisons surface
     the discrepancy.
- risks_remaining:
  - **Listwise variable-length batches** will need a model-side
    `MaskedLinear` or attention-style aggregation if R-LOSS picks ListMLE
    or SoftRank. Out of scope for R1/R2; flagged for R-LOSS implementation.
  - **`policy_hash` is whitespace-sensitive**. Editing the YAML to add a
    comment changes the hash. Mitigation: in R-LOCK-3 we hash the raw
    bytes (simplest, no parser dependency); R-LOSS can switch to a
    canonical-JSON hash if reformatting becomes a hassle.
  - **Sampler `pairs_per_gene` and `max_list_len` are tunable**, not locked
    in this decision. They're hyperparameters of R-LOSS, not of the
    contract.
- next_action: R1 can start. Pointwise sampler integrates with the T5-A
  training loop via a thin wrapper. R-LOSS uses pairwise/listwise when it
  runs (R3+).

### Reproducibility Attachments
- config_snapshot: n/a (no Hydra run)
- code_sha: <fill on commit>
- artifacts:
  - `src/data/datasets/ranking_batch.py`
  - `src/evaluation/ranking_metrics.py` (R-LOCK-4 partner)
  - `data_contract/ranking/eligibility_policy.yaml`
  - `data_contract/ranking/split_protocol.yaml`
  - `data_contract/ranking/metric_contract.yaml`
  - `data_contract/schemas/run_manifest_v2.schema.json`
  - `tests/unit/test_ranking_batch.py` (15 tests)
- sampler_mode: pointwise (default for R1, R2)
- primary_metric_name: within_gene_spearman_mean
