---

# Tiered Refactor Proposal

overview: Define a rigorous, restart-style roadmap for conditional gene essentiality prediction with regression-first objectives, strict anti-leakage preprocessing, and a tiered experimental decision process.
todos:

- id: freeze-data-contract
content: Finalize and document v4 Media_Components_ML data contract + train-only preprocessing artifacts
status: pending
- id: freeze-eval-protocol
content: Select and lock official organism-holdout split protocol and baseline metric thresholds
status: pending
- id: tier1-experiments
content: Design and run Tier 1 representation experiments with strict anti-leakage preprocessing
status: pending
- id: tier2-experiments
content: Run minimal fusion ablations and lock fusion MVP winner
status: pending
- id: tier3-experiments
content: Evaluate justified architecture complexity and choose final model family
status: pending
- id: mlops-foundation
content: Implement clean project structure, config management, tracking, and required tests before broad sweeps
status: pending
isProject: false

---

# Tiered Refactor Proposal

## Clean-Room Restart Charter (Authoritative Rules)

- This plan is the **single source of truth** for the restart.
- Existing code, metrics, and prior conclusions are treated as **untrusted inputs**.
- Prior work may only appear as explicit hypotheses to re-test; nothing is auto-promoted.
- No tier may advance without pre-declared success criteria and an auditable decision log entry.

## Locked Decisions

- Primary task: **continuous fitness regression** (`fit`) for `(gene, condition)` pairs.
- Authoritative condition source: `[media_composition_v4.xlsx](/home/ds85/projects/ConditionalGeneEssentiality/data/media_composition_v4.xlsx)`, sheet `Media_Components_ML`.
- Leakage policy: **train-only** condition vocab/scalers per split protocol, with explicit `UNK` handling at val/test.

## Authoritative Data Scope (Use vs Do-Not-Use)

This section defines exactly which local data artifacts are in scope for the restart. Any deviation requires a decision-ledger entry and version bump.

### Required inputs (authoritative)

- Raw fitness source of truth:
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/raw/feba.db`
- Condition composition source of truth:
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/media_composition_v4.xlsx`
  - required sheet: `Media_Components_ML`
- Frozen gene embedding bundle:
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/processed/ProtLM_embeddings_layer8/*.pt`
- Canonical modeling tables (required operational artifacts for reproducible runs):
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/derived/canonical/v0/fitness_experiment_long.parquet`
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/derived/canonical/v0/experiments.parquet`
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/derived/canonical/v0/media_components_long.parquet`
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/derived/canonical/v0/media_master.parquet`

### Optional inputs (explicitly non-authoritative defaults)

- `/home/ds85/projects/ConditionalGeneEssentiality/data/raw/aaseqs` (needed only if regenerating embeddings)
- `/home/ds85/projects/ConditionalGeneEssentiality/data/derived/condition_encoding/v0/experiments_condition.parquet` (diagnostic/legacy compatibility only unless promoted by test)
- `/home/ds85/projects/ConditionalGeneEssentiality/data/materialized/`* (candidate caches only; never source of truth for published metrics)

### Out-of-scope by default

- older workbook versions:
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/media_composition.xlsx`
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/media_composition_v2.xlsx`
  - `/home/ds85/projects/ConditionalGeneEssentiality/data/media_composition_v3.xlsx`
- any prior labeled-embedding artifacts or threshold-derived label bundles (if present elsewhere in repo) are forbidden as supervision sources.

### Data-contract hard gate

- All reported experiments must log:
  - raw source id (`feba.db` checksum),
  - workbook checksum + sheet id (`Media_Components_ML`),
  - embedding manifest/checksum,
  - canonical-table manifest/checksum.
- If any run uses non-authoritative inputs, mark as exploratory and block promotion decisions.

## Program-Level Decision Ledger (Required)

- For each assumption, record:
  - assumption id,
  - tier tested,
  - experiment ids,
  - evidence summary,
  - pass/fail decision,
  - promoted default (or rejection).
- Required assumption classes:
  - data contract assumptions,
  - split and leakage assumptions,
  - representation assumptions,
  - fusion assumptions,
  - architecture assumptions,
  - optimization assumptions.

### Decision Ledger Entry Template (Use For Every Promotion Decision)

```markdown
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
- metrics_primary: [<RMSE>]
- metrics_secondary: [<within-gene Spearman>, <MAE>]
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
- statistical_check: <mean+-std or CI across seeds>
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
```

### Promotion Rubric (Hard Gate)

- **Pass** only if:
  - primary metric threshold is met,
  - secondary non-degradation rule is met,
  - leakage and split-integrity tests pass,
  - reproducibility checks pass.
- **Fail** if any hard gate fails, even if primary metric improves.
- **No winner** is allowed when results are inconclusive, unstable, or underpowered.

## Stage 0: Reproducibility and Governance (Before Any Modeling Tier)

**Goal:** establish infrastructure so every downstream result is reproducible and attributable.

### Required outputs

- frozen data contract document (v4 + `Media_Components_ML` parsing rules),
- immutable split manifest(s) with seeds,
- run manifest schema,
- decision-ledger template,
- minimal test harness for data/split/metric invariants.

### Stage-0 acceptance gate

- A run can be recreated from artifacts + config + seed with equivalent metrics inside tolerance.
- Leakage checks pass by test, not by manual inspection.
- If reproducibility is not proven, Tier 1 cannot start.

## Pre-Tier Baseline: Evaluate the Measurement System First

**Goal:** validate that evaluation is trustworthy before comparing model choices.

### Data-splitting strategy (anti-leakage)

- Use organism-level holdout as primary benchmark axis.
- Select one official protocol for tier decisions; keep others as secondary diagnostics.
- Add novelty buckets for condition/media exposure where possible.
- Freeze split manifest and disallow split edits during a tier.

### Baseline hierarchy (must be re-established from scratch)

- Null baseline set (required):
  - global mean predictor,
  - per-condition/group mean with safe fallback,
  - per-organism mean with safe fallback (document cold-start behavior).
- Legacy baseline ideas are treated as hypotheses, not truths.
- Compute baselines on the exact scored row set used by each model comparison.

### Metrics and decision criteria

- Primary metric: **RMSE** on val/test for `fit`.
- Secondary metrics:
  - within-gene condition ranking Spearman,
  - MAE,
  - per-organism spread,
  - residual slicing by condition categories.
- A candidate wins only if it beats current MVP on RMSE and does not violate ranking stability tolerance.

### Pre-Tier acceptance gate

- Baseline numbers are stable across reruns with fixed seeds.
- Metric code and row-set alignment are verified by tests.
- Decision ledger contains baseline selection rationale.

## Pre-Tier Addendum: Data and Evaluation Trustworthiness (Required Before Tier 1)

This addendum promotes previously observed risks to explicit gates. Tier-1 work cannot start until these stages are complete and logged in the decision ledger.

### Revised early-stage flow (authoritative)

1. **Stage 0 - Reproducibility and governance lock** (existing section)
2. **Stage 0.5 - Evaluation trustworthiness lock** (new hard gate)
3. **Stage 1 - Deep data-sheet analysis lock** (new hard gate)
4. **Stage 2 - Split protocol lock (chemistry-aware organism holdout)** (new hard gate)
5. **Stage 2.5 - Data-quality policy lock** (weighted-full vs strict-slice + organism-tier policy)
6. **Tier 1-4** proceed only after Stages 0-2.5 are approved

### Stage 0.5 - Evaluation trustworthiness lock

**Goal:** define what "good" means without external benchmark papers by anchoring model results to protocol-specific nulls and estimated noise floors.

#### Required outputs

- null-baseline suite for each protocol:
  - global train mean,
  - per-condition/per-experiment mean with safe fallback,
  - per-organism mean with safe fallback (document cold-start behavior),
  - embedding nearest-neighbor baseline once embedding lookup is active.
- optional stronger checks (if feasible):
  - component-vector chemistry mean baseline,
  - additive baseline (`embedding prior + condition prior`) as interaction sanity check.
- metric power report:
  - within-gene Spearman eligibility counts for candidate `m`,
  - bootstrap CI for Spearman on locked val set,
  - permutation/null Spearman check.
- estimated irreducible-noise reference from replicate-quality proxies (document method and assumptions).

#### Hard-gate decisions

- whether within-gene Spearman is primary gate metric or diagnostic-only (based on power, not preference),
- protocol-specific "meaningful gain" thresholds for RMSE and ranking metrics,
- final null-baseline set reused by all models under that protocol.

#### Heteroscedastic-noise metric policy (required)

- Treat Tn-seq `fit` residuals as potentially heavy-tailed/heteroscedastic unless disproven by diagnostics.
- Report **RMSE and MAE as co-primary metrics** for promotion decisions; neither metric may be omitted.
- Add residual diagnostics to every protocol report:
  - residual quantiles (including tail quantiles),
  - per-organism residual spread,
  - residual slicing by fit-magnitude buckets.
- Any improvement claim must disclose whether gains occur in central-mass errors, tail errors, or both.

#### Evaluation integrity checks (hard gate)

- **Variance decomposition status control**
  - treat any sum-of-squares decomposition run on convenience subsamples as exploratory-only evidence,
  - do not use exploratory decomposition outputs as sole justification for split-policy choices,
  - require at least one stronger confirmatory analysis (mixed-effects or documented robustness checks across sampling schemes) before promoting variance-based conclusions to policy.
- **LOOO null-definition audit**
  - for each LOOO fold, explicitly record null definition and fallback path,
  - verify whether the effective null is truly global train mean (excluding held-out organism) or includes any organism-conditioned information,
  - block cross-protocol interpretation if null definitions differ without explicit disclosure.
- **Denominator parity requirement**
  - model and baseline comparisons must use the exact same scored row set (identical row ids/counts),
  - embedding-join filters, quality filters, and split masks must be applied identically before metric computation,
  - fail the comparison if row-set parity checks fail; no promotion decision allowed on mismatched denominators.
- **Evaluation-manifest minimum fields**
  - scored_rowset_id/hash,
  - n_rows_scored for model and each baseline,
  - null_definition_id (per protocol/fold),
  - inclusion/exclusion counters (pre-join, post-join, post-filter, post-split),
  - spearman_eligibility_policy_id (`m`, variability threshold, exclusion counters).

### Stage 1 - Deep data-sheet analysis lock

**Goal:** decide split feasibility and evaluation limits from empirical overlap and support, not assumptions.

#### Required analysis questions

- organism-to-organism overlap:
  - media-name overlap,
  - component-level chemistry overlap,
  - stressor/condition overlap.
- support and sparsity:
  - conditions-per-gene distributions per organism,
  - genes eligible for Spearman at candidate `m` thresholds,
  - minimum support thresholds for valid val/test organisms.
- quality and noise structure:
  - `fit`, `t`, `cor12` distributions globally and per organism,
  - impact of candidate quality filters on retained supervision mass.
- modality coverage:
  - mapped vs unmapped chemistry coverage and policy implications,
  - embedding coverage/drop rates by organism and by split candidate.
- OOD diagnostics:
  - candidate val/test chemistry unseen-rate versus train,
  - optional embedding-distance OOD profile versus train genes.
- homology-leakage diagnostics:
  - nearest-train sequence-similarity proxy for each val/test gene (embedding cosine required baseline; alignment/MMseqs optional upgrade),
  - metric stratification by similarity bins (high/medium/low),
  - error-versus-similarity trend checks to detect memorization-like behavior.

#### Condition encoding specification (required before Tier 1)

- Build chemistry vocabulary from `Media_Components_ML` using rows with `Include_in_ml == True`.
- Define and freeze deterministic mapping artifacts:
  - `canonical_id_vocab` (ordered list),
  - `canonical_id_to_index`,
  - `media_to_multihot` matrix contract,
  - duplicate handling rule for repeated `(Media, Canonical_ID)` rows (must be idempotent).
- Keep feature-family registry explicit:
  - chemistry multihot features,
  - condition/stressor categorical features,
  - optional numeric fields (concentration, temperature) with documented transforms.
- Any prevalence/variance-based feature trimming must be fit on **train only** per split and persisted as an artifact.

#### Experiment metadata feature policy (required)

- Audit and document candidate metadata fields for predictive use:
  - organism id,
  - genotype/background,
  - oxygen/aerobic state,
  - growth phase / harvest state,
  - temperature,
  - stressor/concentration/unit fields where present.
- For each field, pre-register:
  - encoding method,
  - missingness policy,
  - leakage risk check,
  - ablation priority (include vs exclude decision test).

#### Paired-dropout feasibility audit (required)

- Detect candidate medium pairs via explicit naming/set-difference rules (for example `_no_`*, `_minus_`*, or controlled component-delta patterns).
- Hard validity requirement: a usable pair must be within the **same organism** and compatible protocol context; cross-organism pairings are diagnostic-only and cannot define causal delta targets.
- Emit a pair manifest with:
  - pair ids,
  - dropped component(s),
  - organisms represented,
  - number of experiments and genes eligible per pair.
- If pair coverage is insufficient or heavily confounded, paired-dropout remains secondary/diagnostic only.

#### Target-scale and normalization policy (required)

- Authoritative default target remains raw `fit` regression.
- Any target normalization (for example per-experiment z-score) is allowed only as a controlled ablation with train-only fitted parameters where applicable.
- Promotion decisions must report whether gains persist on raw-scale metrics versus null baselines.

#### Spearman eligibility refinement (required)

- Within-gene Spearman is computed only for genes that satisfy both:
  - minimum validation condition count `m`,
  - minimum observed validation variability threshold `v_min` (pre-registered robust spread rule, such as IQR/std threshold).
- Genes failing eligibility are excluded from Spearman aggregation and counted explicitly in reports.
- Eligibility thresholds must be fixed before tier comparisons and recorded in the decision ledger.

#### Biological sanity-check set (required)

- Define a small set of known biological expectation checks (for example, nutrient-dropout/auxotrophy-style plausibility cases) used as non-promotion guardrails.
- A model that fails these checks cannot be promoted without documented exception rationale.

#### Hard-gate decisions

- explicit mapped/unmapped chemistry policy,
- minimum support policy for organisms used in val/test,
- approved overlap diagnostics that must be reported with every split protocol.
- homology-diagnostic readiness policy:
  - whether Stage-1 similarity evidence requires a dedicated homology-aware diagnostic protocol for promotion reporting.

#### LB mixed-representation risk policy (required)

**Risk statement:** a large fraction of rows use LB-family media, and historical workbook versions mixed physical-ingredient semantics with in-silico metabolite semantics for LB-like entries. This can create spurious overlap, unstable vocabularies, and misleading chemistry generalization claims if not controlled.

**Required controls**

- normalize LB naming to a single canonical alias policy before feature construction (for example, `LB`, `LB (Miller)`, and related synonyms must map deterministically);
- assign and persist a `representation_mode` tag per medium (`physical`, `in_silico`, or `mixed`) in the condition data contract;
- forbid silent merging of physical and in-silico component vectors into one untagged feature space;
- require split diagnostics to report representation-mode proportions for train/val/test and within novelty buckets;
- require at least one ablation that compares:
  - strict physical-only chemistry encoding,
  - strict in-silico-aware encoding with explicit mode indicators,
  - fallback policy for media lacking one representation mode.

**Promotion guardrail**

- no representation/fusion promotion is allowed if gains are explained only by representation-mode leakage (for example, train/val differences in LB encoding mode that are not controlled by explicit features and diagnostics).

### Stage 2 - Split protocol lock (chemistry-aware organism holdout)

**Goal:** lock split definitions that test scientifically meaningful generalization instead of accidental impossibility.

#### Split-design policy

- Organism holdout remains primary axis.
- Chemistry overlap is a required companion axis; each protocol must report val/test chemistry seen-rate and unseen-rate versus train.
- At least one protocol must avoid near-zero chemistry overlap scenarios that make success unattainable regardless of architecture.
- If using multiple held-out organisms, select them by pre-declared overlap/support strata, not random choice.
- If paired-dropout protocols are used, pair assignments must satisfy within-organism validity requirements and be reported separately from standard organism-holdout metrics.
- Homology-aware evaluation is a required secondary axis when Stage-1 diagnostics show strong similarity dependence:
  - keep organism-holdout as primary promotion protocol,
  - add homology-aware masking/partition as a diagnostic protocol (do not silently replace the primary axis).

#### Recommended protocol set

- **Primary promotion protocol:** organism holdout with non-trivial but not degenerate chemistry overlap.
- **Secondary diagnostic protocol:** lower-overlap stress test (reported, not necessarily promotion-gating).
- **Optional robustness protocol:** LOOO or leave-k-organisms-out summary once candidate defaults are near locked.
- **Homology diagnostic protocol (conditional required):** if enabled by Stage-1 evidence, evaluate with train-time homology masking/partition or clustering constraints and report similarity-stratified outcomes.

#### Hard-gate decisions

- official primary protocol id for tier promotions,
- official secondary diagnostic protocol(s),
- allowed val/test organism pool with documented support and overlap rationale.
- paired-dropout protocol status (`not_used|diagnostic_only|promotion_eligible`) with explicit justification.
- homology protocol status (`not_used|diagnostic_only|required_diagnostic`) and thresholding method (`embedding-sim bins|MMseqs/cluster`) with rationale.

### Stage 2.5 - Data-quality policy lock

**Goal:** resolve data-policy questions before architecture expansion.

#### Required controlled comparisons

- weighted-full vs strict-slice (same split, seeds, budget),
- full vs curated vs curated-strict organism pool (if curation policy is under consideration),
- all comparisons include null deltas and ranking diagnostics.
- extract-handling policy comparison:
  - keep decomposition-derived component features as-is,
  - add explicit extract-indicator features (for example yeast extract/tryptone-like flags),
  - optional restrictive filter policy for decomposition-derived features if justified.
- metadata contribution ablations:
  - chemistry-only,
  - chemistry + selected experiment metadata,
  - chemistry + metadata + extract-indicator features.
- target normalization ablation:
  - raw `fit` target baseline,
  - normalized-target variant (if tested), both scored with denominator parity and protocol-specific null context.
- loss-family policy comparison (before architecture expansion):
  - MSE objective baseline,
  - Huber objective baseline with pre-registered delta candidates,
  - same split, seeds, budget, and denominator parity.
- optional uncertainty diagnostic comparison (if enabled):
  - interval estimation method(s),
  - empirical coverage probability by organism and fit-magnitude bucket,
  - interval sharpness/width summary.

#### Hard-gate decisions

- default row-quality policy,
- default organism-pool policy,
- frozen thresholds/functions for quality weighting or filtering.
- default condition-feature bundle (chemistry-only vs chemistry+metadata vs chemistry+metadata+extract flags),
- default target-scale policy (raw only vs raw+normalized dual reporting),
- paired-dropout track disposition (deferred vs active secondary track),
- default loss family for Tier 3+ (`MSE|Huber`) and, if Huber, default delta policy,
- uncertainty diagnostics status (`off|diagnostic_only|required_diagnostic`) for downstream tiers.

### Added cross-tier hypotheses (must be pre-registered in ledger)

- **H-DATA-01:** Explicit mapped/unmapped chemistry handling improves robustness versus silent drop or naive fallback.
- **H-DATA-02:** Organism support thresholds materially affect metric stability and decision confidence.
- **H-EVAL-01:** Within-gene Spearman is sufficiently powered for promotion decisions under the locked protocol.
- **H-EVAL-02:** RMSE gains should be interpreted relative to protocol-specific null and documented noise references.
- **H-SPLIT-01:** Split outcomes depend strongly on chemistry overlap; protocol must report this dependence explicitly.
- **H-SPLIT-02:** Random held-out organism choice is inferior to overlap/support-stratified selection for fair comparison.
- **H-POLICY-01:** Weighted-full can match or beat strict-slice while preserving more supervision mass.
- **H-POLICY-02:** Curated organism pools improve robustness only if they improve primary metrics without harming generalization diagnostics.
- **H-TRAIN-01:** Balanced sampling across organisms/conditions may outperform naive row-proportional sampling.
- **H-TRAIN-02:** Curriculum-style training (quality-first or simpler-to-harder) may improve stability and ranking consistency; treat as controlled Tier-4-style optimization hypotheses unless promoted earlier by evidence.
- **H-ENC-01:** Canonical-ID chemistry multihot with train-only trimming outperforms media-name-only encoding without causing instability.
- **H-ENC-02:** Adding experiment metadata (oxygen/temperature/genotype/growth state where available) improves conditional prediction over chemistry-only features.
- **H-ENC-03:** Explicit extract-indicator features mitigate decomposition-induced over-coupling better than chemistry-only vectors.
- **H-PAIR-01:** Within-organism paired-dropout deltas provide stronger conditional signal than unpaired medium-level supervision for affected nutrient families.
- **H-TARGET-01:** Normalized targets may aid optimization but should not be promoted unless gains persist on raw-scale protocol metrics.
- **H-HOMO-01:** Model performance is partially explained by train-val sequence similarity; low-similarity bins must be reported to avoid overclaiming generalization.
- **H-HOMO-02:** Homology-aware masking/clustering reduces optimistic bias versus pure organism holdout while preserving enough supervision mass for stable evaluation.
- **H-METRIC-01:** MAE and RMSE may disagree under heavy-tailed noise; robust improvements should improve or hold both.
- **H-LOSS-01:** Huber objective improves robustness to extreme noisy rows versus MSE without degrading central-mass performance.
- **H-UQ-01:** Coverage probability diagnostics reveal whether predictive intervals are calibrated across organisms and fit-magnitude regimes.
- **H-SPR-01:** Spearman computed with variability-gated eligibility better reflects conditional sensitivity than count-only eligibility.

## Tiered Experimental Roadmap

## Tier 1: Input Representation

**Core question:** what condition encoding from `Media_Components_ML` best supports conditional ranking and error reduction?

### Tier-1 MVP baseline

- Gene input: frozen ProteomeLM embeddings.
- Condition input: simple categorical + numeric encoding built from train-only vocab/scalers.
- Model head: shallow regression MLP.

### Experiments (2-3)

1. **Exp 1A - Granularity Test**
  - Hypothesis: component-level composition beats coarse media-id encoding.
  - Implementation: compare media-id/category-only vs component presence/amount vectors.
  - Success criteria: RMSE decrease plus Spearman non-degradation.
2. **Exp 1B - Numeric Transform Test**
  - Hypothesis: transformed concentrations (log/scaled) outperform raw values.
  - Implementation: compare raw vs log1p vs bounded normalization.
  - Success criteria: consistent gains across seeds with stable variance.
3. **Exp 1C - Unknown Handling Test**
  - Hypothesis: explicit `UNK` + unknown-rate diagnostics improves robustness on novel conditions.
  - Implementation: compare naive drop/zero-fill vs explicit UNK + mask indicators.
  - Success criteria: stronger novelty-bucket performance and fewer brittle failures.

### Tier-1 required representation ablations (must run before promotion)

- chemistry-only vs chemistry+experiment-metadata feature bundles,
- chemistry-only vs chemistry+extract-indicator feature bundles,
- train-only feature trimming policy sensitivity (no trim vs conservative trim),
- if paired-dropout is promotion-eligible at Stage 2, include at least one paired-delta representation comparison as a secondary diagnostic.

### Tier-1 promotion rule

- Promote exactly one representation schema and artifact contract to Tier 2.
- Record rejected alternatives and reasons in decision ledger.

## Tier 2: Fusion MVP

**Core question:** what is the simplest reliable fusion strategy once representation is fixed?

### Tier-2 MVP baseline

- Tier-1 winning representation frozen.
- Initial fusion: `concat([gene_emb, condition_vec]) -> regression head`.

### Experiments (2-3)

1. **Exp 2A - Linear vs Shallow MLP Fusion**
  - Hypothesis: shallow nonlinearity captures useful interaction signal.
  - Implementation: linear head vs one-hidden-layer head.
  - Success criteria: better RMSE/Spearman at similar runtime budget.
2. **Exp 2B - Early vs Late Fusion**
  - Hypothesis: separate towers then merge improves compositional robustness.
  - Implementation: direct concat vs two-tower merge.
  - Success criteria: gains on novelty subsets with acceptable complexity.
3. **Exp 2C - Minimal Gating**
  - Hypothesis: condition-gated gene features improve conditional sensitivity.
  - Implementation: small gating module over gene embedding.
  - Success criteria: improved ranking on high-condition-variance genes.

### Tier-2 promotion rule

- Promote one fusion method only.
- Freeze fusion defaults before architecture scaling.

## Tier 3: Architectural Complexity

**Core question:** which complexity is justified after representation and fusion are locked?

### Tier-3 MVP baseline

- Tier-1 and Tier-2 winners fixed.
- Training recipe fixed to isolate architecture effects.

### Experiments (2-3)

1. **Exp 3A - Depth/Residual Test**
  - Hypothesis: deeper residual MLP improves performance without instability.
  - Implementation: 2-4 layer MLP, with/without residual links.
  - Success criteria: consistent gain and acceptable seed variance.
2. **Exp 3B - Interaction Block Test**
  - Hypothesis: lightweight interaction block improves gene-condition coupling.
  - Implementation: small FiLM-like or attention-like interaction block.
  - Success criteria: meaningful gain over Tier-2 winner at bounded compute.
3. **Exp 3C - Efficiency Frontier Test**
  - Hypothesis: smaller models can match larger models when inputs are well designed.
  - Implementation: parameter/runtime vs RMSE/Spearman frontier sweep.
  - Success criteria: choose smallest model within predefined delta of best score.

### Tier-3 promotion rule

- Promote single model class and default size.
- Archive full comparison evidence for reproducibility.

## Tier 4: Optimization (Brief, Post-Lock)

- Tune only after Tier 1-3 are frozen:
  - LR schedule, batch size, weight decay, dropout, Huber delta, early stopping.
- Re-test data quality policy choices (weighted vs strict filtering) as controlled experiments.
- Finalize multi-seed confidence intervals and robustness report for locked architecture.

## Canonical Starter Structure (Authoritative)

```text
project_root/
  README.md
  pyproject.toml
  .gitignore

  data_contract/
    data_contract_v1.md
    schemas/
      canonical_tables.schema.json
      condition_features.schema.json
    splits/
      split_protocol_primary.yaml
      split_protocol_diagnostic.yaml
    preprocessing/
      vocab_policy.md
      scaler_policy.md

  src/
    domain/
      entities.py
      contracts.py
      metrics_contract.py
      split_contract.py
    data/
      ingestion/
        load_v4_media_components_ml.py
        load_embeddings.py
        load_fitness_tables.py
      preprocessing/
        fit_condition_vocab.py
        fit_condition_scalers.py
        transform_conditions.py
        unknown_category_policy.py
      datasets/
        build_model_dataset.py
        dataset_audits.py
    models/
      representations/
        condition_encoders.py
      fusion/
        concat_linear.py
        towers_merge.py
        gated_fusion.py
      architectures/
        shallow_mlp.py
        residual_mlp.py
        interaction_block.py
    train/
      loop.py
      losses.py
      optimizer_factory.py
      evaluator.py
      checkpointing.py
    evaluation/
      null_baselines.py
      metrics.py
      reporting.py
    experiments/
      registry.py
      tier0/
        run_stage0_reproducibility.py
      pretier/
        run_baselines.py
      tier1/
        run_representation_ablation.py
      tier2/
        run_fusion_ablation.py
      tier3/
        run_architecture_ablation.py
      tier4/
        run_optimization_sweeps.py
    cli/
      run_experiment.py

  configs/
    base/
      data.yaml
      model.yaml
      train.yaml
      eval.yaml
    stage0/
      reproducibility.yaml
    pretier/
      baseline_eval.yaml
    tier1/
      exp_1a_granularity.yaml
      exp_1b_numeric_transform.yaml
      exp_1c_unknown_handling.yaml
    tier2/
      exp_2a_linear_vs_mlp.yaml
      exp_2b_early_vs_late.yaml
      exp_2c_gated.yaml
    tier3/
      exp_3a_depth_residual.yaml
      exp_3b_interaction_block.yaml
      exp_3c_efficiency_frontier.yaml
    tier4/
      optimization_sweep.yaml

  tests/
    unit/
      test_metrics.py
      test_unknown_category_policy.py
      test_scaler_fit_train_only.py
    integration/
      test_split_integrity.py
      test_no_feature_leakage.py
      test_dataset_build_determinism.py
      test_baseline_reproducibility.py
    fixtures/
      mini_canonical.parquet
      mini_conditions.parquet

  research_log/
    decisions/
      decision_template.md
      pretier/
      tier1/
      tier2/
      tier3/
      tier4/
    tier_reports/
      stage0_report.md
      pretier_report.md
      tier1_report.md
      tier2_report.md
      tier3_report.md
      tier4_report.md

  artifacts/
    runs/
      <run_id>/
        config_snapshot.yaml
        split_manifest.yaml
        preprocessing_artifacts/
        metrics.json
        baseline_comparison.json
        diagnostics.json
        model_checkpoint.pt
    indexes/
      run_index.csv
```

### Structure Rules (Non-Negotiable)

- Core implementation logic lives only in `src/` (single source of truth).
- Tier-specific differences are expressed in `configs/tier*/` and `src/experiments/tier*/`, not duplicated code trees.
- Every promoted decision must have a corresponding file in `research_log/decisions/<tier>/`.
- Failed experiments remain in `artifacts/runs/<run_id>/` with diagnostics; do not keep dead experimental branches in core modules.
- Any change to dataset schema, split policy, or preprocessing policy must update `data_contract/` first.

### Why This Is Preferred Over Full Per-Tier Code Directories

- Preserves readability by tier (configs, runners, logs are tier-scoped).
- Prevents code drift and copy-paste divergence in critical data/model/train logic.
- Makes bug fixes and reproducibility checks centralized and auditable.
- Keeps failed ideas discoverable via immutable artifacts instead of polluting production code.

### Configuration and tracking policy

- Use YAML-based composed configs (Hydra-style or equivalent).
- Persist for every run:
  - git SHA,
  - split id,
  - preprocessing artifact id,
  - config snapshot,
  - seed,
  - metrics,
  - null-baseline deltas,
  - unknown-category rates.
- Track experiments with consistent tags:
  - `stage`,
  - `tier`,
  - `experiment_id`,
  - `split_protocol`,
  - `data_contract_version`.

## Minimal Test Plan Required Before Tier 1

- Leakage tests:
  - vocab/scalers fit only on train rows,
  - val/test unseen categories map to `UNK`,
  - no transform statistics derived from val/test.
- Split tests:
  - no organism overlap across split partitions,
  - deterministic split reproduction from manifest/seed.
- Data integrity tests:
  - join key cardinality and duplicate checks for `orgId`, `locusId`, `expName`, `gene_key`.
- Metric tests:
  - RMSE/Spearman correctness on synthetic fixtures.
- Smoke reproducibility tests:
  - fixed-seed rerun reproduces metrics inside tolerance.

## Prompt Refinement (Authoritative Restart Prompt)

"Start a clean-room refactor of this project into a strict, tiered ML research system for conditional gene essentiality prediction from Tn-seq.

Hard constraints:

1. Primary objective is regression on continuous fitness (`fit`).
2. Condition source is `media_composition_v4.xlsx`, sheet `Media_Components_ML`.
3. Preprocessing is split-aware and train-only: build condition vocab/scalers on train data only; map unseen val/test categories to `UNK`; log unknown-category rates.
4. Existing code and results are untrusted by default and may only be used as hypotheses to re-test.
5. No tier promotion without pre-declared success criteria and a written decision-ledger entry.

Required deliverables:

- Stage 0 reproducibility/governance setup.
- Pre-Tier evaluation baseline re-established from scratch (including null baselines).
- Tier 1-4 experiment plans with hypothesis, implementation, and success criteria.
- Strictly separated project structure for data, models, and training.
- Config/experiment tracking design for full reproducibility.
- Test suite focused on leakage prevention, split integrity, and metric correctness.

Process policy:

- Re-learn every major assumption through controlled tier experiments.
- Promote only one winner per tier.
- Record all accepted/rejected decisions with evidence."

