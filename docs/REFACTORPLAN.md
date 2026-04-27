# Tiered Refactor Plan v2 — Conditional Gene Essentiality Prediction

**Status:** plan revision (v2). Supersedes the v1 plan archived at
`archive/docs/REFACTORPLAN_v1.md`.

**Scope:** clean-room rebuild of a continuous-fitness regression system for
`(gene, condition)` pairs from Tn-seq fitness data, using frozen ProteomeLM gene
embeddings and chemistry features from `media_composition_v4.xlsx` →
`Media_Components_ML`.

**Reading order:** §1 (charter) → §2 (locked decisions) → §3 (data scope) →
§7 (stage pipeline) → §8 (tiers). §4 (hypothesis registry) and §5 (decision
ledger) are referenced but not read end-to-end.

---

## 1. Clean-Room Charter (Authoritative Rules)

- This plan is the **single source of truth** for the restart.
- Existing code, metrics, and prior conclusions in `archive/` are **untrusted inputs**.
  Prior work may only appear as explicit hypotheses to re-test; nothing is auto-promoted.
- **No tier or stage may advance without** a pre-declared success criterion and a written
  decision-ledger entry.
- **Train-only preprocessing.** Vocab/scalers fit on train rows only per split protocol,
  with explicit `UNK` handling at val/test. Unknown-category rate logged every run.
- **Denominator parity.** Model and any baseline being compared against must be
  evaluated on the *exact same scored row set* (same row ids, same count).
- **Co-primary metrics.** RMSE and MAE are co-primary; neither may be omitted from a
  promotion decision. Within-gene Spearman is reported alongside but its gating role is
  decided in Stage 2 based on power.

## 2. Locked Decisions

| # | Decision | Reason it's locked |
|---|---|---|
| L1 | Primary task is continuous regression on raw `fit`. | Conditional essentiality is a continuous quantitative phenotype; thresholding throws away signal. |
| L2 | Authoritative condition source is `data/media_composition_v4.xlsx`, sheet `Media_Components_ML`. | v4 is the only workbook with `Include_in_ml`, `Canonical_ID`, and a `Decomposition_type` column suitable for ML preprocessing. v1–v3 archived. |
| L3 | Gene embeddings: frozen ProteomeLM layer-8 vectors at `data/processed/ProtLM_embeddings_layer8/*.pt`. | Compute budget; fine-tuning is testable as `H-EMB-01` if a tier plateaus. |
| L4 | Co-primary metrics: RMSE + MAE. Within-gene Spearman is conditional on Stage-2 power evidence. | Tn-seq fit residuals are heavy-tailed; either-or reporting hides regimes. |
| L5 | Hydra is the config framework. | Composable YAML is required; ad-hoc loaders fragment quickly across stages and tiers. |
| L6 | Project structure follows §9. Logic lives only in `src/`. | Prevents code drift across tiers. |

## 3. Authoritative Data Scope

### Required inputs

| Artifact | Path | Notes |
|---|---|---|
| Raw fitness DB | `data/raw/feba.db` | Never modified |
| Condition workbook | `data/media_composition_v4.xlsx`, sheet `Media_Components_ML` | Schema verified in S0 |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` | Frozen |
| Canonical fitness | `data/derived/canonical/v0/fitness_experiment_long.parquet` | Inner join `GeneFitness ⋈ Experiment` |
| Canonical experiments | `data/derived/canonical/v0/experiments.parquet` | |
| Media master | `data/derived/canonical/v0/media_master.parquet` | |
| Media components | `data/derived/canonical/v0/media_components_long.parquet` | |

### Explicitly out of scope

- Older workbooks: `data/media_composition.xlsx`, `media_composition_v2.xlsx`,
  `media_composition_v3.xlsx`. Diagnostic comparison only.
- All `archive/` code, prior labeled-embedding artifacts, prior threshold-derived
  label bundles. Forbidden as supervision sources.
- `data/derived/condition_encoding/v0/` — diagnostic only.
- `data/materialized/` — candidate caches, never source of truth for promotion metrics.

### Data-contract hard gate (per run)

Every reported run must log:
- `feba_db_sha256`
- `workbook_v4_sha256` + sheet id (`Media_Components_ML`)
- `embedding_manifest_id` (sha256 of the bundle manifest)
- `canonical_manifest_id` (sha256 of the canonical-build manifest)

Any run using non-authoritative inputs is `exploratory` and ineligible for promotion.

---

## 4. Hypothesis Registry & Experiment Matrix

Every hypothesis below has exactly one "owner" — the stage or tier that tests it.
Hypotheses without a clear owner are dropped.

| ID | Hypothesis | Owner | Concrete test |
|---|---|---|---|
| H-DATA-01 | Explicit mapped/unmapped chemistry handling improves robustness vs silent drop. | S1 | Audit unmapped rows; pre-register handling policy. |
| H-DATA-02 | Organism support thresholds materially affect metric stability. | S1 → S3 | Measure RMSE/Spearman variance across candidate val orgs at varying support. |
| H-EMB-01 | Frozen ProteomeLM embeddings carry sufficient gene-level signal for fitness regression. | T3 (re-test only if T2 winner plateaus). | Fine-tune top-N layers vs frozen, controlled by null-baseline delta. |
| H-EVAL-01 | Within-gene Spearman is sufficiently powered for promotion decisions. | S2 | Bootstrap CI on locked val; report `n_genes_eligible`. |
| H-EVAL-02 | RMSE gains must be interpreted relative to protocol-specific nulls. | S2 (policy) | Always-on policy; not a separately-tested hypothesis. |
| H-BASE-01 | A model that does not beat the additive baseline is not learning gene×condition interactions and is ineligible for promotion. | S2 → all tiers | Fit `fit ~ a + α[gene] + β[condition]`; record additive RMSE/MAE; gate every tier promotion. |
| H-SPLIT-01 | Split outcomes depend on chemistry overlap between train and val. | S1 → S3 | Per candidate protocol: report val/test chemistry seen-rate vs train. |
| H-SPLIT-02 | Stratified val-org selection produces tighter cross-seed metric stability than random selection. | S3 | Run k=5 random vs k=5 stratified; compare metric std across seeds. |
| H-POLICY-01 | Weighted-full retains more supervision mass than strict-slice without harming primary metrics. | S5 | Same split, seeds, budget; compare RMSE+MAE + null delta. |
| H-POLICY-02 | Curated organism pools improve robustness only if primary metrics improve and generalization diagnostics do not degrade. | S5 (deferred if S1 finds support too uneven for curation to matter). | Full vs curated org pool, same protocol. |
| H-ENC-01 | Canonical-ID chemistry multihot beats media-name-only encoding. | T1-A | Media-id encoder vs multihot encoder. |
| H-ENC-02 | Numeric concentration transforms (log1p / bounded) outperform raw amounts. | T1-B | raw vs log1p vs bounded. |
| H-ENC-03 | Explicit UNK + mask indicators improve robustness on novel conditions vs zero-fill. | T1-C | Zero-fill vs explicit UNK. |
| H-ENC-04 | Adding selected experiment metadata (oxygen, growth phase, temperature) improves conditional prediction over chemistry-only features. | T1-D | Chemistry-only vs chemistry+metadata. |
| H-ENC-05 | Explicit decomposition-mode indicators (extract/in-silico flags) mitigate over-coupling vs untagged chemistry vectors. | T1-E | Chemistry vs chemistry+mode flags. |
| H-FUSE-01 | Shallow nonlinear fusion beats linear fusion. | T2-A | Linear head vs 1-hidden MLP head, same encoder. |
| H-FUSE-02 | Two-tower (separate encoders → late merge) beats early concat on novelty subsets. | T2-B | Early concat vs two-tower merge. |
| H-FUSE-03 | Condition-gated gene features (FiLM-like) improve ranking on high-condition-variance genes vs un-gated fusion. | T2-C | Un-gated vs FiLM/gating. |
| H-CAP-01 | Adding depth + residual links to the locked fusion improves RMSE without seed instability. | T3-A | 1-layer vs 2-layer vs 4-layer residual MLP. |
| H-CAP-02 | A smaller model can match a larger model when input representation is well-designed. | T3-B | Param/runtime vs RMSE frontier. |
| H-LOSS-01 | Huber objective improves robustness to extreme rows vs MSE without degrading central-mass metrics. | T4 | MSE vs Huber, all else fixed. |
| H-TARGET-01 | Per-experiment z-score normalization of `fit` aids optimization but should not be promoted unless gains persist on raw-scale metrics. | T4 | Raw vs normalized target. Promotion gated on raw-scale RMSE+MAE. |
| H-HOMO-01 | Model performance is partially explained by train-val sequence similarity. | S1 (diagnostic) | Embedding cosine similarity bins; metric stratification. |
| H-HOMO-02 | Homology-aware masking reduces optimistic bias vs pure organism holdout. | S3 (conditional, triggered if H-HOMO-01 effect size > 0.5σ on val Spearman). | Add homology-masked diagnostic protocol. |
| H-METRIC-01 | RMSE and MAE may disagree under heavy-tailed noise. | Always-on policy (per L4). | Report both for every comparison. |
| H-SPR-01 | Variability-gated Spearman eligibility better reflects conditional sensitivity than count-only eligibility. | S2 | Pre-register threshold; report eligible-gene counts. |

**Dropped from v1 (orphaned, untestable, or scope creep):**
- v1 H-TRAIN-01 (balanced sampling): no concrete experiment owner; re-introducible as
  T4 ablation if T1–T3 plateau.
- v1 H-TRAIN-02 (curriculum): same reason.
- v1 H-PAIR-01 (paired-dropout): supervision-paradigm scope creep. Stage-1 audit
  retained as diagnostic only; no promotion track.
- v1 H-UQ-01 (coverage probability): defer to a future post-T4 calibration project.

---

## 5. Decision Ledger Protocol

For every promotion decision (stage gate or tier gate):

1. **Pre-register** the comparison and promotion rule **before** running the experiment.
   Record in `research_log/decisions/<stage_or_tier>/<decision_id>.md` using the template.
2. **Run** the comparison; persist all artifacts to `artifacts/runs/<run_id>/`.
3. **Decide.** Update the ledger entry with evidence summary, decision outcome, rationale.
4. **Promote** by emitting the appropriate handoff artifact (§6).

### Promotion rubric (hard gate — all must pass)

- Co-primary metric thresholds met (both RMSE and MAE).
- Secondary non-degradation rule met (Spearman, per-organism spread).
- **Beats additive baseline** (per H-BASE-01).
- Leakage tests pass.
- Split-integrity tests pass.
- Reproducibility check passes (fixed-seed rerun within tolerance).

`No winner` is a permitted outcome; underpowered or unstable comparisons must not produce promotions.

### Decision-ledger entry template

Located at `research_log/decisions/decision_template.md`. Required header fields:
`decision_id`, `stage_or_tier`, `date`, `owner`, `status`, `related_experiments`,
`assumption_under_test`, `pre_registered_comparison`, `metrics_primary` (RMSE+MAE),
`metrics_secondary`, `promotion_rule`, `evidence_summary`, `decision_outcome`,
`reproducibility_attachments`.

---

## 6. Artifact Handoff Contracts

Every stage and tier consumes upstream artifacts and emits downstream artifacts.
The handoff is by file, not by prose.

| Stage | Consumes | Emits |
|---|---|---|
| S0 | — | `data_contract/run_manifest_v1.schema.json`, `data_contract/data_contract_v1.md`, `data_contract/v4_schema_verification.json` |
| S1 | S0 | `data_contract/splits/candidate_protocols.yaml`, `research_log/tier_reports/s1_data_characterization.md` |
| S2 | S1 | `data_contract/policy/eval_policy.yaml`, `artifacts/baselines/baselines_per_protocol.json` |
| S3 | S1 + S2 | `data_contract/splits/locked_protocol.yaml`, `data_contract/splits/diagnostic_protocols.yaml` |
| S4 | S3 | `data_contract/feature_contract.yaml`, `data_contract/preprocessing/<artifact_id>/` |
| S5 | S4 | `data_contract/policy/quality_policy.yaml` |
| T1 | S0–S5 | `research_log/decisions/tier1/<decision_id>.md`, `data_contract/representation_winner.yaml` |
| T2 | T1 | `research_log/decisions/tier2/<decision_id>.md`, `data_contract/fusion_winner.yaml` |
| T3 | T2 | `research_log/decisions/tier3/<decision_id>.md`, `data_contract/architecture_winner.yaml` |
| T4 | T3 | `research_log/decisions/tier4/<decision_id>.md`, final policy locks |

Each handoff file has a JSON Schema in `data_contract/schemas/`.

---

## 7. Stage Pipeline (S0 → S5)

### S0 — Reproducibility & Governance

**Goal:** infrastructure that makes every later result reproducible and auditable.

**Required outputs**
1. **v4 schema verification** — script reads `Media_Components_ML`, asserts the expected
   columns (`Media`, `Canonical_ID`, `Compound_name`, `Chemical_form`,
   `Source_row_component`, `Decomposition_type`, `Ingredient_source`, `Include_in_ml`,
   `Source_dataset`, `Source_url`), records `workbook_v4_sha256`, emits
   `data_contract/v4_schema_verification.json`. **First action of the refactor.**
2. **Run manifest schema** at `data_contract/schemas/run_manifest_v1.schema.json` —
   defines the JSON shape every run output must conform to (data-contract checksums,
   git SHA, config snapshot, seed, scored_rowset_hash, n_rows_scored,
   inclusion/exclusion counters, unknown_category_rate, null-baseline deltas).
3. **End-to-end smoke pipeline** — Hydra config → split → train-only preprocessing →
   no-op model (predicts global mean) → metrics → manifest. Proves the wiring works.
4. **Test harness** — leakage, split-integrity, metric-correctness, manifest-validation,
   smoke-reproducibility tests, all green.
5. **Decision-ledger template** at `research_log/decisions/decision_template.md`.

**Acceptance gate**
- Smoke run reproduces metrics within tolerance across two fixed-seed reruns.
- All tests green.
- v4 schema verification artifact committed.

### S1 — Data Characterization

**Goal:** describe the data well enough to choose split protocols rationally.
Pure description; no parameter fitting.

**Required analyses**
- Organism-to-organism overlap: media-name overlap, component-level chemistry overlap
  (Canonical_ID intersection), stressor/condition overlap.
- Support and sparsity: conditions-per-gene distributions per organism; row-count per
  organism; row-count per (org, media).
- Quality and noise: `fit`, `t`, `cor12` distributions globally and per organism;
  per-organism residual-spread proxies.
- Modality coverage: mapped vs unmapped chemistry coverage; embedding coverage by organism.
- OOD diagnostics: candidate val/test chemistry unseen-rate vs train; embedding cosine
  similarity profile vs train genes (homology proxy for `H-HOMO-01`).
- LB representation-mode audit: every Canonical_ID gets a `representation_mode` tag
  (`physical` | `mix` | `in_silico`) derived from `Decomposition_type`. Fraction of
  rows per mode, per organism, per candidate protocol.

**Hard-gate decisions**
- Mapped/unmapped chemistry handling policy (`H-DATA-01`).
- Minimum support threshold for candidate val/test organisms (`H-DATA-02`).
- Whether `H-HOMO-01` evidence is strong enough (effect size > 0.5σ) to require a
  homology diagnostic in S3.

**Emits:** `data_contract/splits/candidate_protocols.yaml` listing 3–5 candidate
protocols with documented (val_orgs, test_orgs, chemistry-overlap, support) tuples.

### S2 — Evaluation Trustworthiness

**Goal:** lock evaluation policy and compute baselines for each candidate protocol.
Must complete *before* S3 can choose a protocol.

**Required baselines (computed per candidate protocol)**
- **Global train mean** — predict train mean for every val row.
- **Per-condition mean** with safe fallback to global.
- **Per-organism mean** with safe fallback to global (cold-start documented).
- **Additive baseline** — fit `fit ~ a + α[gene] + β[condition]` by least squares on
  train; predict on val. Required by `H-BASE-01`.
- **Embedding nearest-neighbour baseline** — for each val row, find the most similar
  train gene by embedding cosine within the same condition; copy its fit. Stress test
  for homology leakage.

**Required reports**
- **Metric power report**: bootstrap 95% CI for Spearman on each candidate val set;
  permutation null Spearman; per-protocol `n_genes_eligible` at candidate `m` values.
- **Spearman eligibility policy**: pre-register `m = 5` and `v_min = 25th percentile
  of cross-gene IQR on the candidate val set`. Frozen before S3.
- **Heteroscedastic noise diagnostics**: residual quantile profile per candidate
  protocol; per-organism residual spread.

**Hard-gate decisions**
- Whether within-gene Spearman is a primary gate metric or diagnostic-only — decided
  per protocol from the power report (`H-EVAL-01`).
- Protocol-specific "meaningful gain" thresholds for RMSE and MAE (e.g., 1× vs 0.5×
  the gap between additive baseline and global-mean baseline).
- Final null-baseline set reused by all later models under that protocol.

**Emits:** `data_contract/policy/eval_policy.yaml` (Spearman eligibility, gain
thresholds, primary/secondary metric assignments per protocol),
`artifacts/baselines/baselines_per_protocol.json` (the actual numbers).

### S3 — Split Protocol Lock

**Goal:** select one primary protocol from S1 candidates using S1 + S2 evidence.
Pick by transparent rule, not preference.

**Selection rule**
- Primary: protocol with non-degenerate chemistry overlap (val/test chemistry seen-rate
  in 30–80% range) AND val support sufficient for stable metrics (per S2 power report).
- Secondary: lower-overlap stress-test protocol; reported but not gating.
- Conditional: if `H-HOMO-01` triggered in S1, add a homology-masked diagnostic protocol.

**Stratified vs random selection** (`H-SPLIT-02`): when multiple candidates qualify,
prefer overlap/support-stratified over random; record cross-seed stability as evidence.

**Emits:** `data_contract/splits/locked_protocol.yaml` (one protocol, with seeds and
manifest hash) + `data_contract/splits/diagnostic_protocols.yaml`.

### S4 — Feature Contract

**Goal:** define the chemistry feature schema as an immutable contract. The split
is locked, so train-only preprocessing has a well-defined target row set.

**Required outputs**
- `canonical_id_vocab` — ordered list of Canonical_IDs present in train rows with
  `Include_in_ml = True`.
- `canonical_id_to_index` — dict; index 0 reserved for `<UNK>`.
- `media_to_multihot` — `(n_media, len(vocab))` binary contract; idempotent on
  duplicate `(Media, Canonical_ID)` rows.
- `representation_mode` per medium (`physical` | `in_silico` | `mix`).
- Numeric-field registry — `Amount`, optional `temperature`, optional `concentration`,
  with declared transforms (raw / log1p / bounded).
- Metadata-field registry — `oxygen`, `growth_phase`, `genotype`, with encoding
  method and missingness policy each.

**Hard-gate decisions**
- Mapped/unmapped policy from S1 instantiated as concrete `<UNK>` vs explicit drop
  per family.
- Train-only feature trimming policy: prevalence threshold, fit on train only.

**Emits:** `data_contract/feature_contract.yaml`,
`data_contract/preprocessing/<artifact_id>/` containing fitted vocab/scalers and
their checksums.

### S5 — Training-Recipe Lock

**Goal:** lock the optimization-irrelevant training-data choices that, if left as
sweep dimensions in T1, would confound all tier comparisons.

**Required controlled comparisons** (run with the locked feature contract, no model
architecture changes; use `concat_linear` shallow MLP fixed):
- Weighted-full vs strict-slice (`H-POLICY-01`).
- Curated vs full organism pool, only if S1 found pool variance large enough to matter
  (`H-POLICY-02`); else explicitly skipped with rationale.

**Hard-gate decisions**
- Default row-quality policy (`weighted_full` | `strict_slice`).
- Default organism pool (`full` | `curated`).
- Frozen weighting/filtering thresholds.

**Deliberately out of scope at S5** (moved to T4):
- Loss family (MSE vs Huber). Locked in T4.
- Target normalization. Locked in T4.

**Emits:** `data_contract/policy/quality_policy.yaml`.

---

## 8. Tiered Modeling Roadmap (T1 → T4)

Each tier has a single concern. No tier may revisit decisions made by an earlier tier
without an explicit ledger entry justifying the re-open.

### Tier 1 — Representation

**Sole concern:** which condition feature schema instance gives the best regression?

**Fixed controls:** locked split (S3), locked feature contract (S4), locked
training recipe (S5), shallow MLP fusion + concat head (T2/T3 architectures NOT yet
chosen — use the simplest viable head). Same seeds across arms.

**Experiments**
| ID | Hypothesis | Comparison |
|---|---|---|
| T1-A | H-ENC-01 | media-id only vs canonical-ID multihot |
| T1-B | H-ENC-02 | raw vs log1p vs bounded numeric transform |
| T1-C | H-ENC-03 | zero-fill UNK vs explicit-UNK + mask indicators |
| T1-D | H-ENC-04 | chemistry-only vs chemistry + experiment metadata |
| T1-E | H-ENC-05 | chemistry vs chemistry + decomposition-mode indicators |

**Promotion rule:** one schema wins. Beats every null baseline and the additive
baseline (H-BASE-01) on RMSE+MAE. Spearman non-degradation. Reject schemas where
gains are explained by representation-mode confound (LB risk policy).

**Emits:** `data_contract/representation_winner.yaml`.

### Tier 2 — Fusion

**Sole concern:** how do gene and condition vectors combine? **All fusion topology
decisions live here.** No new fusion topologies introduced in T3.

**Fixed controls:** T1 winner. Same shallow head capacity as T1.

**Experiments**
| ID | Hypothesis | Comparison |
|---|---|---|
| T2-A | H-FUSE-01 | linear head vs 1-hidden-layer MLP head |
| T2-B | H-FUSE-02 | early concat vs two-tower merge |
| T2-C | H-FUSE-03 | un-gated fusion vs FiLM/condition-gated gene features |

**Promotion rule:** one fusion topology wins; beats T1-winner on co-primary metrics
plus additive baseline. T2-C wins only if it improves ranking on high-condition-variance
genes (H-FUSE-03 specific).

**Emits:** `data_contract/fusion_winner.yaml`.

### Tier 3 — Capacity

**Sole concern:** given the locked fusion, what additional capacity helps? **Depth,
residuals, regularization only — no new fusion topologies.**

**Fixed controls:** T1 + T2 winners.

**Experiments**
| ID | Hypothesis | Comparison |
|---|---|---|
| T3-A | H-CAP-01 | 1-layer vs 2-layer vs 4-layer residual MLP head |
| T3-B | H-CAP-02 | param/runtime vs RMSE+MAE frontier sweep; pick smallest within ε of best |
| T3-C (conditional) | H-EMB-01 | frozen ProteomeLM vs fine-tune top-N layers — only if T3-A/B plateau against null delta. |

**Promotion rule:** smallest model class within tolerance of the best.

**Emits:** `data_contract/architecture_winner.yaml`.

### Tier 4 — Optimization & Final Policy Locks

**Concerns** (in order):
1. Loss family (`H-LOSS-01`): MSE vs Huber, with delta search.
2. Target normalization (`H-TARGET-01`): raw vs per-experiment z-score. Promotion
   gated on raw-scale metric improvement.
3. LR schedule, batch size, weight decay, dropout, early stopping.
4. Multi-seed confidence intervals on the locked architecture.

**Promotion rule:** Pareto-improving combinations only; no regression vs T3 winner
on co-primary metrics.

---

## 9. Project Structure & Tooling

### Layout

```
project_root/
  README.md  pyproject.toml  .gitignore  CLAUDE.md

  data_contract/
    data_contract_v1.md
    v4_schema_verification.json    # S0
    feature_contract.yaml          # S4
    representation_winner.yaml     # T1
    fusion_winner.yaml             # T2
    architecture_winner.yaml       # T3
    schemas/
      run_manifest_v1.schema.json
      canonical_tables.schema.json
      condition_features.schema.json
    splits/
      candidate_protocols.yaml     # S1
      locked_protocol.yaml         # S3
      diagnostic_protocols.yaml    # S3
    policy/
      eval_policy.yaml             # S2
      quality_policy.yaml          # S5
    preprocessing/
      <artifact_id>/               # S4 fitted vocab + scalers

  src/
    domain/         # entities, contracts, metrics_contract, split_contract
    data/
      ingestion/    # load_v4_media_components_ml, load_embeddings, load_fitness_tables
      preprocessing/# fit_condition_vocab, fit_condition_scalers, transform_conditions, unknown_category_policy
      datasets/     # build_model_dataset, dataset_audits
    models/
      representations/  # condition_encoders
      fusion/           # concat_linear, towers_merge, gated_fusion (FiLM)
      architectures/    # shallow_mlp, residual_mlp
    train/          # loop, losses, optimizer_factory, evaluator, checkpointing
    evaluation/     # null_baselines, additive_baseline, nn_baseline, metrics, reporting
    experiments/    # registry + per-stage/tier runners
      stage0/  stage1/  stage2/  stage3/  stage4/  stage5/
      tier1/   tier2/   tier3/   tier4/
    cli/            # run_experiment.py — Hydra entrypoint

  configs/                   # Hydra config tree
    config.yaml              # root; defines defaults list
    data/                    # data input groups
    model/
    train/
    eval/
    stage/                   # one yaml per stage
    tier/                    # one yaml per tier
    experiment/              # one yaml per experiment (T1-A, T2-B, etc.)

  tests/
    unit/  integration/  fixtures/

  research_log/
    decisions/decision_template.md
    decisions/{stage0..stage5,tier1..tier4}/
    tier_reports/

  artifacts/
    runs/<run_id>/
    baselines/
    indexes/

  archive/    # legacy pre-refactor code (reference only)
```

### Configuration: Hydra

- **Framework:** `hydra-core>=1.3` (added to `pyproject.toml`).
- **Composition:** root `configs/config.yaml` has a `defaults` list pulling from
  `data/`, `model/`, `train/`, `eval/`, `stage/` or `tier/`, `experiment/`.
- **Override style:** experiments are run as
  `python -m src.cli.run_experiment +experiment=tier1/T1-A`.
- **Output dirs:** Hydra writes per-run dirs under `artifacts/runs/${now:%Y%m%d_%H%M%S}_${experiment_id}/`.
- **Sweep:** Hydra's multirun (`-m`) handles seed sweeps and arm sweeps. No hand-rolled
  sweep loops in `src/experiments/*`.

### Tracking policy

Persist for every run (in the run's manifest, conforming to `run_manifest_v1.schema.json`):
- git SHA, hydra config snapshot (`.hydra/config.yaml`)
- split protocol id, preprocessing artifact id
- seed, code SHA
- `feba_db_sha256`, `workbook_v4_sha256`, `embedding_manifest_id`, `canonical_manifest_id`
- metrics (RMSE, MAE, Spearman) with `n_rows_scored`, `n_genes_eligible`
- null-baseline deltas (one per baseline)
- `unknown_category_rate`
- `scored_rowset_hash` for denominator parity

### Structure rules (non-negotiable)

- Logic only in `src/`. Configs are data, never code.
- Tier-specific differences live in `configs/experiment/` and thin wrappers in
  `src/experiments/<tier>/`. Core data/model/train modules are tier-agnostic.
- Every promoted decision has a file in `research_log/decisions/<stage_or_tier>/`.
- Failed experiments stay in `artifacts/runs/<run_id>/` with diagnostics; never in `src/`.
- Any change to `data_contract/` triggers a version bump and a decision-ledger entry.

---

## 10. Test Plan (Required Before Any Tier)

**Leakage tests**
- Vocab/scalers fit only on train rows; assertion check.
- Val/test unseen categories map to `<UNK>`; explicit policy test.
- No transform statistics derived from val/test.

**Split tests**
- No organism overlap across train/val/test partitions.
- Deterministic split reproduction from manifest + seed.

**Data integrity tests**
- Join-key cardinality and duplicate checks for `orgId`, `locusId`, `expName`, `gene_key`.
- Canonical-table schema conforms to `canonical_tables.schema.json`.
- v4 workbook conforms to `condition_features.schema.json`.

**Metric tests**
- RMSE / MAE / within-gene Spearman correctness on synthetic fixtures.
- Eligibility filter behaves as specified (`m`, `v_min`).

**Manifest tests**
- Every run output validates against `run_manifest_v1.schema.json`.
- Required checksum fields are non-empty.

**Smoke reproducibility tests**
- Fixed-seed rerun of S0 smoke pipeline reproduces metrics within `1e-6` tolerance.

`pytest tests/` must stay green before any commit. Stage gates require this plus
the stage-specific acceptance criteria.
