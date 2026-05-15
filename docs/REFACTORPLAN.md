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
| L7 | **Scope of generalization claim:** "*Given a gene and a growth medium **and applied stressor chemistry** drawn from a known chemistry vocabulary (with explicit `<UNK>` / `<UNK_STRESSOR>` handling), our model predicts conditional gene essentiality — including for organisms not seen during training, and conditions structured differently from those the gene appeared in during training.*" Locked 2026-04-27; stressor clause ratified 2026-04-29 (S4-DEC-002). | S1 figure 17 confirmed all 4 candidate protocols are ≥95% **medium** chemistry-seen at the Canonical_ID level. S4 Option D extends the locked contract to stressors (`condition_1..4`) as chemistry rows, with train-only prevalence and resolution YAML. v4 still cannot support a "generalizes to any chemistry" claim. Going beyond requires fingerprint encoders or Canonical_ID-level holdouts (see §12 Deferred Experiments). |

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
| H-ENC-01 | Decomposed chemistry encoding — a multihot over the locked 425-slot Canonical_ID vocab covering **both medium and stressor chemistry per experiment** (per S4-DEC-002 Option D) — beats coarse medium-name-only encoding. | T1-A | Media-name embedding (medium string only) vs canonical-ID multihot from `experiment_chemistry.parquet` (which already includes resolved stressors). T1-A's gap therefore conflates two effects: (a) decomposed vs coarse representation, and (b) chemistry scope (medium-only vs medium+stressor). The conflation is intentional — it measures the headline encoder vs the strawman baseline. A strict-stressor ablation that disentangles (a) from (b) is in §12 Deferred Experiments. |
| H-ENC-02 | Numeric concentration transforms (log1p / bounded) outperform raw amounts. | T1-B | raw vs log1p vs bounded. |
| H-ENC-03 | Explicit UNK + mask indicators improve robustness on novel conditions vs zero-fill. | T1-C | **Skipped** (T1-DEC-005): zero chemistry unknown rate on locked split + vocab makes both arms identical. Deferred, not rejected — reactivate if prevalence trimming or Canonical_ID holdout introduces nonzero unknowns. |
| H-ENC-04 | Adding selected experiment metadata (oxygen, growth phase, temperature) improves conditional prediction over chemistry-only features. | T1-D | Chemistry-only vs chemistry+metadata. |
| H-ENC-05 | Explicit decomposition-mode indicators (extract/in-silico flags) mitigate over-coupling vs untagged chemistry vectors. | T1-E | Chemistry vs chemistry+mode flags. |
| H-FUSE-01 | Shallow nonlinear fusion beats linear fusion. | T2-A | Linear head vs 1-hidden MLP head, same encoder. |
| H-FUSE-02 | Two-tower (separate encoders → late merge) beats early concat on novelty subsets. | T2-B | Early concat vs two-tower merge. |
| H-FUSE-03 | Condition-gated gene features (FiLM-like) improve ranking on high-condition-variance genes vs un-gated fusion. | T2-C | Un-gated vs FiLM/gating. |
| H-CAP-01 | Adding depth + residual links to the locked fusion improves RMSE without seed instability. | T3-A | 1-layer vs 2-layer vs 4-layer residual MLP. |
| H-CAP-02 | Wider hidden layers improve performance by retaining more gene×condition interaction capacity through the projection bottleneck. | T3-B | Width sweep {128, 256, 512, 1024} at T3-A winning depth. |
| H-CAP-03 | FiLM gating's near-threshold advantage (T2-C) amplifies at deeper capacity. | T3-D | Concat vs FiLM at T3-A winning depth. |
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
  (`physical` | `mix` | `extract` | `in_silico`) per
  `data_contract/representation_mode_mapping.yaml`. Fraction of rows per mode,
  per organism, per candidate protocol.

**Hard-gate decisions**
- Mapped/unmapped chemistry handling policy (`H-DATA-01`).
- Minimum support threshold for candidate val/test organisms (`H-DATA-02`).
- Whether `H-HOMO-01` evidence is strong enough (effect size > 0.5σ) to require a
  homology diagnostic in S3.
- Ratification or revision of the proposed `representation_mode_mapping.yaml`
  (S1-DEC-001).

**Required figures (per §11 Visualization Standard).** All saved to
`research_log/figures/stage1/` as PNG + sibling CSV. Each figure informs at
least one hard-gate decision listed above.

| # | Figure | Decision it informs |
|---|---|---|
| 01 | `01_org_media_overlap_heatmap.png` (48×48 shared-media count) | candidate protocol selection |
| 02 | `02_org_canonical_id_overlap_heatmap.png` (48×48 chemistry overlap) | H-SPLIT-01, candidate selection |
| 03 | `03_org_pair_jaccard_distribution.png` (pairwise Jaccard histogram) | candidate selection |
| 04 | `04_bipartite_org_media_top.png` (top-degree bipartite graph) | qualitative organism-clustering |
| 05 | `05_rows_per_organism_bar.png` (sorted, log-scale) | min support threshold |
| 06 | `06_conditions_per_gene_cdf.png` (empirical CDF, faceted) | Spearman eligibility `m` (H-EVAL-01, H-SPR-01) |
| 07 | `07_conditions_per_gene_violin.png` (per-organism distribution) | min support threshold |
| 08 | `08_org_media_row_count_heatmap.png` (org × media row counts) | candidate selection support analysis |
| 09 | `09_genes_per_organism_bar.png` | min support threshold |
| 10 | `10_fit_distribution_per_org_violin.png` | heteroscedastic-noise policy |
| 11 | `11_t_stat_distribution_per_org.png` | quality filter policy |
| 12 | `12_cor12_distribution_per_experiment.png` | quality filter policy |
| 13 | `13_fit_qq_plot_global.png` (tail-behavior QQ plot) | T4 loss-family choice (H-LOSS-01) |
| 14 | `14_chemistry_mapped_unmapped_by_org.png` | mapped/unmapped policy (H-DATA-01) |
| 15 | `15_embedding_coverage_by_org.png` | candidate organism eligibility |
| 16 | `16_canonical_id_prevalence_distribution.png` | feature trimming policy (S4) |
| 17 | `17_chemistry_seen_unseen_rate_per_protocol.png` (faceted) | candidate selection (H-SPLIT-01) |
| 18 | `18_embedding_cosine_to_nearest_train_per_protocol.png` | H-HOMO-01 trigger evaluation |
| 19 | `19_homology_similarity_by_org.png` | H-HOMO-01 trigger evaluation |
| 20 | `20_representation_mode_proportions_per_org.png` (stacked bar) | representation_mode mapping ratification |
| 21 | `21_representation_mode_per_protocol.png` (train/val/test stacks per candidate) | LB risk policy diagnostics |
| 22 | `22_chemical_ubiquity_histogram.png` (cross-organism reuse distribution) | mapped/unmapped policy, feature trimming |
| 23 | `23_organism_topN_chemical_heatmap.png` (top-100 chemicals × 48 orgs, log experiment counts) | cross-organism chemistry coverage |
| 24 | `24_chemical_coverage_curve.png` (chemicals sorted by descending #orgs; dual-axis with experiment count) | feature trimming policy (S4) |

**Optional / exploratory figure**
| # | Figure | Note |
|---|---|---|
| 25 | `25_media_chemistry_umap.png` (UMAP of media multihot vectors, colored by #organisms using each medium) | exploratory; flagged as "not used to gate any decision" in the report. |

**Emits:**
- `data_contract/splits/candidate_protocols.yaml` listing 3–5 candidate
  protocols with documented (val_orgs, test_orgs, chemistry-overlap, support) tuples.
- `research_log/tier_reports/s1_data_characterization.md` embedding all figures
  with captions and the hard-gate decision rationale.
- `research_log/figures/stage1/` containing 24 required + 1 optional figures
  (PNG + sibling CSV each).
- `research_log/decisions/stage1/S1-DEC-001.md` ratifying or revising
  `representation_mode_mapping.yaml`.

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
- Conditional: if `H-HOMO-01` triggered in S1, add homology-aware diagnostics. **Locked policy** (bin stratification as the primary diagnostic; masking for a secondary homology-clean RMSE/MAE; cosine cutoff 0.85; evidence tied to S1 fig 18) is pre-registered in `research_log/decisions/stage3/S3-DEC-001.md`.

**Stratified vs random selection** (`H-SPLIT-02`): when multiple candidates qualify,
prefer overlap/support-stratified over random; record cross-seed stability as evidence.

**Emits:** `data_contract/splits/locked_protocol.yaml` (one protocol, with seeds and
manifest hash) + `data_contract/splits/diagnostic_protocols.yaml`.

### S4 — Feature Contract

**Goal:** define the chemistry + condition metadata schema as an immutable contract
(Option D, S4-DEC-002). The split is locked, so train-only preprocessing has a
well-defined target experiment set.

**Required outputs**
- `canonical_id_vocab` — unified vocabulary of workbook `Canonical_ID` values **plus**
  stressor-derived canonical slots (normalized names for compounds absent from the
  workbook). **Chemistry strings in `experiment_chemistry.parquet` are not rewritten**
  for prevalence; train-only **distinct-experiment** counts drive the vocab registry
  only (threshold 0 ⇒ every train-seen `canonical_id` is listed). Index 0 = `<UNK>`,
  index 1 = `<UNK_STRESSOR>` (reserved for **T1** OOV mapping).
- `canonical_id_to_index` — dict aligned to the ordered vocab.
- `experiment_chemistry.parquet` — long table per experiment:
  `(experiment_id, canonical_id, role ∈ {medium, stressor}, amount, log1p_amount)`.
- `stressor_to_canonical_id.yaml` — ratified string map from Phase 0 matcher
  (`scripts/build_stressor_match_report.py`); copied into the artifact bundle.
- `representation_mode` per medium (`physical` | `mix` | `extract` | `in_silico`)
  unchanged from S1 mapping.
- Numeric registry — `Amount` / stressor concentrations via shared `log1p_amount`
  scaler; `bounded_reference_stats` from workbook `Amount`; optional numeric
  metadata scalers in `numeric_metadata_scalers.json`.
- `experiment_metadata.parquet` — wide encoded table: `oxygen`, `experiment_group`,
  `liquid_state` + parsed `temperature_c`, `pH`, `shaking_rpm` (no genotype /
  `mutantLibrary` in the locked contract).

**Hard-gate decisions**
- `<UNK>` / `<MISSING>` index policy on categoricals; `<UNK>` / `<UNK_STRESSOR>` on
  unified chemistry.
- Train-only **metadata** prevalence + stressor resolution hygiene (no duplicate
  surface strings where the matcher maps to an existing canonical). Chemistry
  prevalence is a vocab-listing gate only (see `chemistry_prevalence_threshold` in
  `configs/stage/s4_feature_contract.yaml`).

**Emits:** `data_contract/feature_contract.yaml` (`schema_version: s4_option_d_v1`),
`data_contract/preprocessing/<artifact_id>/` (checksum-pinned tree including
`artifact_manifest.json`).

### S5 — Training-Recipe Lock

**Goal:** lock the optimization-irrelevant training-data choices that, if left as
sweep dimensions in T1, would confound all tier comparisons.

**Feature contract pointer:** `configs/stage/s5_quality_policy.yaml` references
`data_contract/feature_contract.yaml` — that file must stay aligned with the active
S4 artifact id (**post–S4-DEC-002, Option D**). No code change required unless the
path is duplicated elsewhere with a stale id.

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

**S4 Option D data handoff (do not resurrect pre–S4-DEC-002 paths)**  
Implementers **must** load condition features from the checksum-pinned tree under
`data_contract/preprocessing/<artifact_id>/` as declared in
`data_contract/feature_contract.yaml` (`schema_version: s4_option_d_v1`):
`experiment_chemistry.parquet` (long: `experiment_id`, `canonical_id`, `role`,
`amount`, `log1p_amount`), `experiment_metadata.parquet` (wide encoded metadata),
`representation_mode_per_media.parquet`, **`representation_mode_per_canonical.parquet`**
(one row per `canonical_id` in the chemistry union; stressor-only slots use
`dominant_mode=stressor`), `canonical_id_vocab.json`, `amount_scaler.json`,
`numeric_metadata_scalers.json`, `stressor_to_canonical_id.yaml`. Join experiments
to chemistry/metadata on **`experiment_id`** = SHA256-64 of `(orgId, setName,
seqindex, media)` with **missing `media` hashed as empty string**. There is **no**
`media_to_multihot.parquet` in the locked contract anymore; any multihot baseline
must be **derived in T1** from `experiment_chemistry.parquet`. **Concentration policy:**
`units_1..units_4` are not applied in S4 — do not treat pooled `concentration_*` as
cross-comparable molarities until T1 normalizes units (see
`research_log/notes/T1_option_d_encoding_handoff.md`). **`mutantLibrary` / genotype is
not part of the locked S4 metadata table** (organism-holdout made it non-informative);
do not wire a genotype channel from S4 artifacts.

**Run manifests (when wired):** log `unknown_category_rates.chemistry_eval_rows.fraction_canonical_id_not_in_train_vocab`
plus metadata unknowns, analogous to existing unknown-category logging.

**Experiments**
| ID | Hypothesis | Comparison |
|---|---|---|
| T1-A | H-ENC-01 | `media`-name embedding only vs canonical-ID multihot (medium + stressor) from `experiment_chemistry.parquet`. Strawman-vs-useful-encoder framing; conflates representation effect with chemistry-scope effect. A strict stressor ablation (concatenated stressor strings as the media-id baseline, isolating the representation effect) is in §12 Deferred Experiments. |
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

**Condition side input:** fusion consumes **encoder outputs** on
`experiment_chemistry.parquet` / `experiment_metadata.parquet` (and optional mode
tables), **not** a frozen multihot matrix from S4.

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
| T3-B | H-CAP-02 | Width sweep {128, 256, 512, 1024} at T3-A winning depth; promote if wider hidden dim improves co-primary metrics |
| T3-C (conditional) | H-EMB-01 | frozen ProteomeLM vs fine-tune top-N layers — only if T3-A/B plateau against null delta. |
| T3-D | H-CAP-03 | FiLM gating at T3-A winning depth; re-tests T2-C's near-threshold signal with more capacity |

**Promotion rule:** best performing arm on co-primary metrics. Parsimony applies only
as a tiebreaker when arms are within threshold of each other.

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

---

## 11. Visualization Standard

Every stage and tier that emits a tier report must include figures that directly
inform its hard-gate decisions. Visualizations are not optional decoration —
they are part of the deliverable.

### Per-stage / per-tier figure deliverables

Each stage's spec in §7 (and each tier's spec in §8) lists the **required figures**
inline as a numbered table with one column for "decision it informs." Stages
and tiers may also list **optional / exploratory figures** flagged separately.

| Phase | Required figures | Optional |
|---|---|---|
| S0 | none (no analysis) | none |
| S1 | 24 (see §7 S1) | 1 |
| S2 | TBD (defined when S2 is reached) | — |
| S3 | TBD | — |
| S4 | TBD | — |
| S5 | TBD | — |
| T1+ | TBD per tier | — |

### Layout

```
research_log/figures/<stage_or_tier>/
  NN_<descriptive_snake_case_name>.png   # the figure
  NN_<descriptive_snake_case_name>.csv   # underlying data, one row per plotted element
```

- **Numbering** is global within a stage (`01`, `02`, ..., `25`); stable across
  reruns (do not renumber).
- **Naming** uses snake_case and describes what is plotted, not the conclusion.
  Bad: `21_overlap_is_high.png`. Good: `21_representation_mode_per_protocol.png`.
- **Sibling CSV** contains the data behind the figure so the plot can be
  reproduced without re-running the full analysis. Same basename as the PNG.
- **Captions** live in the tier report (`research_log/tier_reports/<phase>_report.md`),
  not in the PNG. One paragraph per figure: "what to look at, what decision it
  informs, what we concluded."

### Code

Plotting helpers live in `src/evaluation/reporting.py`. Each stage's runner
calls them; no stage reimplements matplotlib boilerplate. Required helpers:

| Helper | Use |
|---|---|
| `save_heatmap(matrix, row_labels, col_labels, *, title, path)` | overlap matrices, org × chemical |
| `save_distribution_per_group(df, value_col, group_col, *, kind, path)` | violin / box plots faceted by organism or condition |
| `save_ecdf(df, value_col, *, group_col, path)` | empirical CDF (e.g. conditions per gene) |
| `save_stacked_bar(df, group_col, stack_col, *, path)` | representation mode proportions |
| `save_bipartite(edges, *, max_nodes_per_side, path)` | org × media bipartite graphs |
| `save_qq(values, *, dist, path)` | tail diagnostic |
| `save_similarity_bin_scatter(...)` | homology-stratified metric plots |
| `save_coverage_curve(...)` | dual-axis chemical-coverage curves |
| `save_umap_scatter(...)` | exploratory media-chemistry UMAP |

Each helper writes both the PNG and its sibling CSV.

### Promotion rule

A stage gate cannot pass with figures missing. Tier reports without figures
fail the promotion rubric §5. The decision-ledger entry for a stage gate must
list each required figure path and confirm it exists.

---

## 12. Deferred Experiments (Future Work, Not Currently Scheduled)

These ideas are tracked here so they are not forgotten. None are scheduled.
Each has a one-line trigger that would justify scheduling it.

### Chemistry-axis generalization

| Experiment | What it tests | Trigger to schedule |
|---|---|---|
| **Chemical-fingerprint encoders** | Replace multihot Canonical_ID with structural fingerprints (Morgan / RDKit / RDF). Lets the model represent unseen chemicals via shared substructure with seen chemicals — i.e., enables "generalize to novel chemistry" claims that L7 currently locks out. | A reviewer asks for novel-chemistry generalization, OR T1+ results plateau and we suspect multihot is the bottleneck. |
| **Canonical_ID-level holdouts** | Hold out specific Canonical_IDs in addition to organisms. Forces the model to predict for chemistry it has never seen. Requires careful construction because holding out water/glucose breaks every medium. | Same triggers as above, especially if fingerprint encoders are added (the two pair naturally). |
| **External validation set** | Evaluate on a Tn-seq dataset from a separate publication that uses media not in v4. Ground-truth test of L7's "known media" boundary. | Once locked architecture is published-ready and we want to bound external validity. |
| **T1-A.1: strict stressor ablation** | Disentangles T1-A's two confounded effects. A1' = single token = concatenation of `(media, condition_1, condition_2, condition_3, condition_4)` as a string id (so the baseline has *access* to the same stressor scope as A2, but still no decomposition). A2 = the locked multihot. Difference isolates the **representation effect** (decomposed vs coarse) from the **scope effect** (with vs without stressors). | T1-A measured gap is large AND we want to publish a causal "decomposed chemistry helps" claim. Skip if T1-A gap is small (effect size doesn't justify decomposition). |
| ~~**T1-A.2: stress-test H-ENC-01 on a protocol with unseen val media**~~ **Completed 2026-05-12** (T1-DEC-002). Result: H-ENC-01 strongly supported. Multihot RMSE 0.5897 vs media_id 3.8451 — 3.26 RMSE gap, 650× threshold. `largest_by_rows` now mandatory diagnostic for all future T1+ promotion reports. | — | Completed. |

### Training-paradigm extensions

| Experiment | What it tests | Trigger to schedule |
|---|---|---|
| **Balanced sampling** (`H-TRAIN-01`, dropped from v2) | Up- or down-weight rows so each organism / each condition contributes equal mass to gradient updates. | T1–T3 plateau and per-organism analysis shows the model is memorizing high-mass orgs. |
| **Curriculum training** (`H-TRAIN-02`, dropped from v2) | Train on quality-filtered or "easy" rows first, then add harder rows. | Same trigger as H-TRAIN-01. |
| **Paired-dropout supervision** (`H-PAIR-01`, dropped from v2) | Use `_no_X` / `_minus_X` media pairs as causal-delta supervision (Δfit per dropped component). | Stage 1 paired-dropout audit (currently optional) finds enough valid within-organism pairs to support this paradigm. |
| **Predictive uncertainty** (`H-UQ-01`, dropped from v2) | Quantile regression / conformal prediction for per-row prediction intervals. | Post-T4, if downstream users (biologists) explicitly need calibrated uncertainty. |

### Embedding fine-tune

| Experiment | What it tests | Trigger to schedule |
|---|---|---|
| **Re-embed the 2,484 missing genes** | Obtain a more complete `aaseqs` (currently has zero coverage of these genes — confirmed 2026-04-27, not just an embedding gap). | New aaseqs dump becomes available; or fallback (mean-of-org) is empirically tested and rejected. |
| **Conditional embedding fine-tune** (`H-EMB-01`, conditional T3-C) | Already scheduled as conditional T3-C; included here for visibility. Fine-tune top-N ProteomeLM layers if T3-A/B plateau. | T3-A/B near-tie with no clear capacity winner. |

### Process

To schedule a deferred experiment: open a new decision-ledger entry under
`research_log/decisions/<owning_stage_or_tier>/`, copy the row from this section
into the entry's "assumption under test," remove it from this list, and slot
the experiment into the appropriate stage/tier spec.


