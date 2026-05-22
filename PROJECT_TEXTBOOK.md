# PROJECT TEXTBOOK: Conditional Gene Essentiality Prediction

**Audience:** Incoming PhD students at the intersection of computational biology and machine learning.  
**Purpose:** Rigorous cross-domain bridge — ML concepts explained for biologists, biological constraints explained for computer scientists.  
**Codebase state as of 2026-05-22:** Stages S0–S5 complete; Tier 1 (T1-A through T1-B.3) complete; T1-C through T4 pending.

---

## 1. Executive Summary & Glossary

### 1.1 Executive Summary

The central question of this project is: **given a bacterial gene and a precise growth condition (a specific medium plus any applied chemical stressors), how essential is that gene for survival?** Essentiality is not binary — a gene that is merely "present" in the genome may be strongly, weakly, or not at all essential depending on what nutrients are present, what chemicals are applied, and which metabolic pathways are redundant under that condition. This project builds a machine learning regression model to predict that continuous essentiality score for any `(gene, condition)` pair.

The raw signal comes from **Tn-seq** (transposon insertion sequencing), a high-throughput functional genomics technique that measures how much each gene is required for growth under a given condition. The model takes as input two modalities: (1) a frozen pre-trained protein language model embedding of the gene's amino acid sequence, and (2) a chemically decomposed encoding of the growth condition derived from a curated chemistry workbook. The model predicts the Tn-seq **fitness score** — a log-ratio that is zero when a gene is dispensable and highly negative when the gene is essential.

**Generalization scope (locked as REFACTORPLAN L7):** The model predicts conditional gene essentiality for `(gene, condition)` pairs where the condition's chemistry is drawn from the locked v4 vocabulary, including for organisms never seen during training. This is a real, testable, publishable claim. Claims of generalization to arbitrary novel chemistry require fingerprint encoders or Canonical_ID-level holdouts, which are deferred to future work.

**Current state:** The pipeline has cleared S0 (infrastructure), S1 (data characterization), S2 (evaluation lock), S3 (split protocol lock), S4 (feature contract), and S5 (training recipe lock). Tier 1 has established the winning condition encoder (multihot over Canonical_IDs with binary presence/absence). The model architecture, fusion topology, capacity, and optimization are decided in Tiers 2–4, which are not yet started.

**Key strategic pivot documented in S2:** The original intuition that a model trained on one organism would rank gene essentiality well for a held-out organism was tested empirically. Under organism-holdout, the additive baseline `fit ~ a + α[gene] + β[condition]` — which should capture gene effects and condition effects independently — collapses to the global training mean because all val organisms, their genes, and their conditions are cold-start. The measured Spearman of this "baseline" is ~0.006–0.01, which is noise-level. This forced a re-interpretation of the evaluation framework: the project's bar is not "beat an additive model" (that bar is trivially satisfied by a constant predictor) but "beat the embedding nearest-neighbor baseline" on the primary protocol.

---

### 1.2 Glossary

#### Tn-seq (Transposon Insertion Sequencing)
**Analogy:** Imagine knocking out every employee at a company one at a time and measuring whether the company can still function. Tn-seq does this at the gene level: a mutant library is created where transposons (mobile DNA elements) randomly disrupt individual genes across millions of bacterial cells. After growing this library in a specific condition, DNA sequencing counts how many cells with each disrupted gene survived relative to the start.

**Technical definition:** A saturating mutagenesis technique using a Tn10 or similar transposon to disrupt genes. The abundance of each gene-disruption mutant is measured by sequencing flanking genomic DNA before and after competitive growth. The ratio of post-growth to pre-growth abundance, log-transformed, gives the **fitness score** for each insertion site. Sites within the same gene are aggregated to produce a per-gene fitness estimate.

**Why this over alternatives:** Tn-seq is scalable to genome-wide screens at low cost. Alternatives like CRISPRi interference require careful guide RNA design and are less saturating. The `feba.db` dataset aggregates Tn-seq experiments from ~48 organisms under ~7,552 unique experimental conditions.

---

#### Fitness Score (`fit`)
**Analogy:** A report card grade for a gene's performance in a specific class. A grade of 0 means the gene did nothing helpful or harmful. A grade of −3 means the gene was critical and its loss was catastrophic. A grade of +0.5 means the gene's disruption actually helped (the gene was a burden under this condition).

**Technical definition:** The per-gene, per-experiment `fit` value is a log₂ ratio of post-selection to pre-selection mutant abundance, normalized by library composition and typically expressed relative to the mean of neutral insertion sites. Formally: `fit = log₂(post/pre) - global_normalization`. Strongly negative values indicate essential genes (disruption eliminates the cells); values near zero indicate dispensable genes. The distribution is heavy-tailed with a pronounced left tail (global p1 ≈ −3.0, p5 ≈ −0.86, median ≈ −0.01).

**Why regression over classification:** Thresholding `fit` to binary essential/non-essential discards the gradient of essentiality (a gene at fit=−0.1 behaves very differently from one at fit=−2.5, even though both may be classified as "not essential"). Continuous regression preserves this signal (REFACTORPLAN L1).

---

#### `t`-statistic and `cor12`
**Analogy:** The `t`-statistic asks "how confident are we in this fitness measurement?" `cor12` asks "did the experiment replicate itself?" 

**Technical definition:** `t` is the per-gene fitness score divided by its standard error across insertion sites, measuring statistical confidence. `cor12` is the Pearson or Spearman correlation between the first and second half of the mutant library — a measure of experimental reproducibility within an experiment. Global `cor12` median ≈ 0.23 (many experiments are only modestly reproducible). The training recipe (Stage S5) uses both to weight rows: `weight = clamp(cor12/median_cor12, 0, 1) × clamp(|t|/median_|t|, 0, 1)`.

---

#### ProteomeLM Embeddings
**Analogy:** A gene's amino acid sequence is like a long book. ProteomeLM has read millions of such books and learned to compress each gene into a fixed-length "summary vector" that captures what kind of protein it is, what family it belongs to, and what functional properties it might have — all without needing experimental data.

**Technical definition:** Frozen 768-dimensional vectors from a protein language model (ProteomeLM), specifically extracted from layer 8. Each gene is represented by its amino acid sequence, which is tokenized and passed through the transformer. The layer-8 hidden state is pooled to produce one vector per gene. These vectors are frozen throughout training — the model learns to use them but does not update them. Coverage: 221,030 gene_keys have embeddings; 2,484 fitness genes have no amino acid sequence in the database and are dropped via inner join. The frozen-embedding choice is REFACTORPLAN L3.

**Why frozen:** Fine-tuning is expensive and risky early in the project (confounds representation experiments with optimization dynamics). It is deferred to T3-C as a conditional experiment triggered only if T3-A/B plateau.

---

#### Canonical_ID
**Analogy:** Different labs call the same chemical by different names — one calls it "sodium chloride," another "NaCl," a third "table salt." `Canonical_ID` is the project's controlled vocabulary that maps all these names to one standard identifier, so the model sees the same token regardless of what a specific experiment's media recipe called the compound.

**Technical definition:** A controlled string identifier from the `data/media_composition_v4.xlsx` workbook (sheet `Media_Components_ML`), column `Canonical_ID`. The locked vocab contains 423 chemistry slots from training experiments plus `<UNK>` (index 0) and `<UNK_STRESSOR>` (index 1), for 425 total. Each Canonical_ID maps to a chemical entity (e.g., "Glucose", "Ammonium chloride", "L-Alanine") with known `Decomposition_type` and `Include_in_ml` flags.

**Why not SMILES/InChI:** Structural fingerprints are deferred (REFACTORPLAN §12). Using the curated vocabulary is the minimal viable representation that supports the locked scope-of-claim without requiring novel chemistry generalization.

---

#### Multihot Encoding
**Analogy:** A shopping list as a barcode. If a grocery store has 425 possible items, a shopping list is represented as a 425-position binary vector where position `i` is 1 if item `i` is on the list and 0 otherwise. Two recipes from different organisms that both use glucose and magnesium sulfate will have the same 1s in those positions, even if they use different medium names.

**Technical definition:** A dense binary vector of length 425 where position `i` is 1 if the experiment includes Canonical_ID `i` (either from the base medium or from applied stressor chemistry) and 0 otherwise. The locked S4 artifact (`de21504134c84a6c`) provides `experiment_chemistry.parquet`, a long table of `(experiment_id, canonical_id, role)` rows. Encoding: group by experiment_id, set all listed canonical_id positions to 1. This is the T1 condition encoder (T1-DEC-004: binary, not concentration-weighted).

**Why multihot over media-name embedding:** When a val experiment uses a medium never seen in training, a media-name embedding gets a single UNK token — completely uninformative. The multihot still encodes all the chemistry components, which are largely in-vocabulary even for novel media names. T1-A.2 confirmed this with a 650× gap in RMSE between the two encoders when val media were all unseen.

---

#### RMSE and MAE (Co-Primary Metrics)
**Analogy:** RMSE is like a headmaster who gives extra punishment for any student who fails by a lot — it squares the errors, so large errors hurt more. MAE is like a fair assessor who just asks "on average, how far off are you?"

**Technical definition:**
- **RMSE** = √(mean((ŷ - y)²)) — Root Mean Squared Error. Sensitive to outliers due to squaring. Appropriate when large prediction errors are disproportionately costly.
- **MAE** = mean(|ŷ - y|) — Mean Absolute Error. Robust to heavy tails; treats all errors proportionally.

**Why co-primary:** The `fit` distribution is heavily left-tailed (global p1 ≈ −3.0; severe deviation from Gaussian). RMSE and MAE routinely disagree in this regime because they weight tails differently. T1-B.3 and T1-DEC-001 both demonstrated H-METRIC-01 (RMSE and MAE disagree) in practice. Reporting only one metric hides this regime-specific behavior. Both must be reported and both must show improvement for a promotion.

---

#### Within-Gene Spearman
**Analogy:** For a given gene, look at all the conditions you've tested it in. Does the model correctly rank which conditions make the gene more essential vs less essential, relative to the true fitness ranks?

**Technical definition:** For each gene `g` with at least `m=5` conditions in the val set and IQR of true `fit` values > `v_min`, compute the Spearman rank correlation between the true `fit` values and the model's predictions across those conditions. Average over all eligible genes. Eligibility criteria are frozen in `eval_policy.yaml`. This metric directly tests the model's ability to predict conditional essentiality rank order — the core biological claim.

**Role:** Co-primary in principle, but S2 established that Spearman of null baselines is ~0.006–0.01 under organism-holdout (noise level). The Spearman bar is therefore non-binding for tier promotions; RMSE+MAE are the hard gates. Spearman is reported as a secondary diagnostic.

---

#### Additive Baseline
**Analogy:** A teacher who predicts every student's exam grade simply by knowing their general ability and the exam difficulty, without accounting for the specific interaction between "how well does this student perform on this type of exam?"

**Technical definition:** The OLS fit of `fit ~ a + α[gene] + β[condition]` on training data. This model can capture main effects (some genes are generally essential; some conditions are generally harsh) but not interactions (gene A is essential in medium X but dispensable in medium Y). Under organism-holdout, this baseline collapses to the global training mean because both val genes (α[gene]) and val conditions (β[condition]) are cold-start (never seen in training). H-BASE-01 originally required models to beat this baseline; S2-DEC-001 re-interpreted the gate to require beating the best non-global baseline (embedding nearest-neighbor on `multi_org_balanced`).

---

#### Embedding Nearest-Neighbor Baseline
**Analogy:** "I don't know this patient, but the most similar patient I've seen — judged by their genetic profile — had outcome X. So I'll predict X for this patient too."

**Technical definition:** For each val row, find the gene in the training set with the highest cosine similarity to the val gene's ProteomeLM embedding. Copy that training gene's `fit` value in the same condition. This baseline is competitive only when: (a) the val gene has a highly similar training gene, and (b) the val condition's media is also in the training set. On `multi_org_balanced` (val_seen_rate=1.0, median train cosine=0.92), this baseline achieves RMSE=0.5884 vs global mean 0.6307 — a real improvement, providing the non-trivial bar for T1.

---

#### Organism-Holdout Split
**Analogy:** The classic machine learning train/val/test split, but the split is done along organism lines. All data from specific bacterial species (e.g., Btheta, Pseudomonas) is held out entirely for evaluation. No gene, no condition, no row from those organisms leaks into training.

**Technical definition:** Each `(gene, condition)` row has an associated `orgId`. The split protocol assigns each organism exclusively to train, val, or test. This tests whether the model can generalize to new organisms — it has never seen these genes or these experimental outcomes for those organisms. The locked primary protocol is `multi_org_balanced` (val={pseudo3_N2E3, BFirm, ANA3, SyringaeB728a_mexBdelta}, test={PV4}).

---

#### FiLM (Feature-wise Linear Modulation)
**Analogy:** A light dimmer switch that is controlled by a different signal. Gene embeddings are the lights; the condition features are the dimmer controller. FiLM lets the condition "adjust the brightness" of every dimension of the gene embedding before the two are combined.

**Technical definition:** FiLM applies an affine transformation to one modality (typically the gene embedding) conditioned on another modality (the condition vector): `h_gene_modulated = γ(condition) ⊙ h_gene + β(condition)`, where `γ` and `β` are small learned networks. This allows condition-dependent gating of gene feature dimensions — the model can learn "in this condition, the gene's metabolic pathway annotations are critical; in that condition, only the structural features matter."

**Status:** Planned for T2-C (hypothesis H-FUSE-03). Not yet run.

---

#### Hydra
**Analogy:** A configuration system where you can assemble a complex experiment from Lego blocks — one block for the data, one for the model, one for training, one for evaluation — and swap any block without rebuilding the others.

**Technical definition:** `hydra-core>=1.3` — a Python framework for hierarchically composing configurations from YAML files. The root config `configs/config.yaml` defines a `defaults` list pulling from data/, model/, train/, eval/, and experiment/ groups. Experiments are run as `python -m src.cli.run_experiment +experiment=T1-A_granularity`. Sweep over seeds: `train.seed=0,1,2 -m`. Every run emits artifacts to `artifacts/runs/${datetime}_${experiment_id}/` with a Hydra-captured config snapshot.

---

#### Decomposition Type / Representation Mode
**Analogy:** There are different reasons why "L-Alanine" might appear in a medium's chemical composition: (1) you explicitly added 1 g/L of pure L-Alanine to a defined medium (`direct` = `physical` mode); (2) you added a salt like `L-Alanine·HCl` (`salt` = `physical` mode); (3) it is part of a known vitamin mixture (`mix` = `mix` mode); (4) it was inferred from yeast extract's known composition (`extract` = `extract` mode). These are physically and quantitatively different.

**Technical definition:** The v4 workbook's `Decomposition_type` field has 5 values: `direct`, `salt`, `mix`, `extract`, `unrecoverable`. These are mapped to 4 representation modes: `physical` (direct+salt), `mix`, `extract`, `in_silico` (reserved, currently unused). T1-E (H-ENC-05) tests whether adding explicit mode indicators to the condition vector improves predictions, motivated by the risk that a model conflating `physical` and `extract` sources of the same molecule will learn spurious shortcuts.

---

## 2. Project Architecture — The Tier System

### 2.1 The Logic of the Pipeline

The pipeline separates concerns into two phases: **Stages (S0–S5)** and **Tiers (T1–T4)**. 

The stages are infrastructure and data decisions that must be locked before any modeling begins. Their purpose is to prevent a classic ML failure mode: discovering that your "result" actually reflects a data preprocessing choice that you tuned alongside your model, making the comparison meaningless.

- **S0** proves the infrastructure works (manifests, reproducibility, smoke test).
- **S1** characterizes the data so that split design is evidence-based rather than arbitrary.
- **S2** locks the evaluation framework — baselines, metrics, eligibility thresholds — before a single modeling run. This is analogous to pre-registering a clinical trial.
- **S3** locks the train/val/test split (one primary + two diagnostic protocols). Fixed before model development; no leakage possible.
- **S4** locks the feature contract: what columns the model receives, how they are computed, which chemistries are in-vocabulary. This prevents the model from implicitly tuning its input representation.
- **S5** locks the training recipe: row-quality weighting, organism pool. Prevents confounding representation experiments (T1) with data-selection effects.

The tiers are modeling decisions that build on the locked foundation:

- **T1 (Representation):** Which condition feature schema gives the best regression? Shallow architecture, fixed fusion.
- **T2 (Fusion):** Given the winning representation, how should gene and condition vectors combine? Fixed representation, fixed capacity.
- **T3 (Capacity):** Given locked fusion, how much depth/residuals help? Fixed representation and fusion.
- **T4 (Optimization):** Loss family, target normalization, LR schedule. Final policy locks.

**The key architectural principle:** each tier has a single concern, controls everything else, and emits a handoff artifact that the next tier must consume without re-opening earlier decisions.

### 2.2 The Promotion Rubric

No tier advances without a pre-registered decision ledger entry. The hard gate: (1) RMSE and MAE both improve by the per-protocol locked threshold; (2) Spearman does not degrade; (3) beats the best non-global baseline; (4) leakage and split-integrity tests pass; (5) fixed-seed rerun reproduces within tolerance. A `no_winner` outcome is allowed — underpowered comparisons do not produce promotions.

### 2.3 Model Architecture (Current)

The T1 control architecture is a **concat-linear shallow MLP**: concatenate the gene embedding (768-dim frozen ProteomeLM) with the multihot condition vector (425-dim), pass through a single hidden layer (256-dim, LayerNorm, ReLU, Dropout 0.1), then a linear output layer. The output is a scalar predicted `fit` score. This baseline architecture is intentionally minimal — complexity is added only after representation and fusion choices are locked.

```
gene_embedding (768)    condition_multihot (425)
        |                         |
        +─────── concat ──────────+
                     |
              [Linear → 768+425 × 256]
              [LayerNorm(256)]
              [ReLU]
              [Dropout(0.1)]
                     |
              [Linear → 256 × 1]
                     |
                predicted fit
```

---

## 3. Tier-by-Tier Breakdown

### Stage S0 — Reproducibility & Governance

**Date:** 2026-04-27 | **Status:** Approved (S0-DEC-001)

#### Hypotheses
No biological hypothesis tested here. The assumption was: "the infrastructure produces a manifest that validates against the schema and bit-reproduces across fixed-seed reruns."

#### Methods & Implementation
The S0 smoke pipeline (`src/experiments/stage0/run.py`) demonstrates end-to-end wiring: load data → verify v4 checksums → perform an 80/20 random split → fit no-op model (predicts global training mean) → compute RMSE/MAE → emit `run_manifest.json`. The manifest is validated against `data_contract/schemas/run_manifest_v1.schema.json`, which specifies required fields: `git_sha`, `feba_db_sha256`, `workbook_v4_sha256`, `embedding_manifest_id`, `canonical_manifest_id`, `split_protocol_id`, `preprocessing_artifact_id`, `seed`, `metrics`, `null_baseline_deltas`, `unknown_category_rate`, `scored_rowset_hash`. Language: Python only.

#### Results
Two independent reruns at seed=0 on an 80k-row sample (15,936 val rows) produced **bit-identical** `smoke_digest` = `4796b8b1...`. Global mean RMSE=0.6486, MAE=0.3201. `v4_schema_verification.json` emitted with the 5 distinct `Decomposition_type` values. 26 tests passed, 2 intentionally skipped.

#### Key finding
The additive baseline RMSE=0.7097, **worse** than global mean at 80k-row scale. This is expected: with thousands of gene and condition effect parameters and only 64k train rows, the OLS fit is noisy and overfits. At full scale (>10M rows in S2), additive baseline improves substantially.

---

### Stage S1 — Data Characterization

**Date:** 2026-04-27 | **Status:** Approved (S1-DEC-001, S1-DEC-002, META-DEC-002)

#### Hypotheses
H-DATA-01 (how to handle unmapped chemistry), H-DATA-02 (minimum organism support), H-HOMO-01 (does homology confound evaluation?), and the representation-mode mapping ratification.

#### Methods
Full descriptive analysis of the 27.4M fitness rows × 48 organisms × 7,552 experiments. 25 figures were generated and saved to `research_log/figures/stage1/` with sibling CSVs. Key analyses in `src/experiments/stage1/analyses.py` and `candidates.py`.

**Cross-organism chemistry overlap** (figures 01–03): computed pairwise shared media names and shared Canonical_IDs between all 48×47/2 = 1,128 organism pairs. Pairwise Jaccard at the Canonical_ID level: median 0.854, p90 0.974. At the media-name level: 67.7% of pairs share zero media names.

**Support characterization** (figures 05–09): rows per organism span 22k–2.1M (92× ratio). Gene-level span 1.2k–6.4k (5× ratio). The row imbalance is driven by experiment density, not gene catalog size.

**Quality and noise** (figures 10–13): `fit` has a heavy left tail (p1 ≈ −3.0, p5 ≈ −0.86). The QQ plot shows severe departure from normality. `cor12` median ≈ 0.23 (modest replicate reproducibility). The RMSE-only-is-insufficient conclusion was empirically grounded here.

**Homology analysis** (figures 18–19): maximum per-organism median deviation in nearest-train-gene cosine = 1.606σ, **far above** the 0.5σ trigger threshold. H-HOMO-01 was triggered, requiring S3 to add a homology-aware diagnostic protocol.

#### Key findings
1. Chemistry overlap at Canonical_ID level is dense (~95–100% seen-rate in every candidate protocol), but media-name overlap is sparse. This is the concrete motivation for the multihot encoder over media-name IDs.
2. Extreme row-count imbalance (92×) means global-average metrics are Btheta-dominated.
3. The `fit` distribution is non-Gaussian and heavy-tailed, empirically motivating MAE, the RMSE+MAE co-primary policy, and Huber loss experimentation in T4.
4. H-HOMO-01 triggered: 4 candidate protocols differ substantially in their mean cosine to nearest train gene (0.51 for `largest_by_rows`, 0.92 for `multi_org_balanced`).

#### Critical discovery: the stressor misclassification (OPEN-001)
During S1 review, it was discovered that `condition_1..4` in `experiments.parquet` — originally treated as categorical metadata (growth phase, genotype) — are **chemical stressors** applied to experiments. Inspection of the S4-DEC-001 metadata vocabulary confirmed this: `growth_phase.json` was full of chemical names like "Nickel (II) chloride hexahydrate." 

Quantified: 87.8% of experiments have at least one non-null `condition_*`; 376 unique stressor strings appear; only 60 match the v4 workbook. This discovery required reformulating the feature contract in S4-DEC-002 (see S4 section).

---

### Stage S2 — Evaluation Trustworthiness

**Date:** 2026-04-28 | **Status:** Approved (S2-DEC-001)

#### Hypotheses
H-BASE-01 (additive baseline gate), H-EVAL-01 (Spearman power), H-EVAL-02 (RMSE relative to protocol-specific nulls), H-METRIC-01 (RMSE/MAE may disagree).

#### Methods
Five null baselines were computed on all four candidate protocols at full scale (2.2M val rows for `multi_org_balanced`):
1. **Global train mean** — predicts the single training mean for every row.
2. **Per-condition mean** — predicts the mean `fit` of each condition seen in training; falls back to global mean for unseen conditions.
3. **Per-organism mean** — predicts the mean `fit` of each organism seen in training; falls back to global for unseen organisms.
4. **Additive baseline** — OLS fit of `fit ~ a + α[gene] + β[condition]` on training; predicts by summing estimated gene and condition effects.
5. **Embedding nearest-neighbor** — for each val row, finds the most cosine-similar training gene within the same condition; copies its `fit`.

**Bootstrap power analysis:** 1,000 row resamples per protocol; 200 permutation-null Spearman runs. `within_gene_spearman` computed with eligibility gate `m=5`, `v_min=cross-gene IQR p25`.

#### Key findings (the cold-start collapse)
Under organism-holdout, baselines 2, 3, and 4 all collapsed to the global training mean (RMSE differences < 0.001). The mechanism: val organisms have no training data, so per-condition and per-organism means fall back to the global mean; the additive baseline's gene and condition parameters (`α[gene]`, `β[condition]`) were never fitted for val organisms, so those terms are zero and the prediction reduces to the intercept `a`, which is approximately the global mean.

| Protocol | global | best non-global | gain threshold |
|---|---:|---:|---:|
| `largest_by_rows` | 0.5668 | 0.5668 (no non-global beats global) | 0.005 (floor) |
| `high_overlap_easy` | 0.5866 | 0.5866 | 0.005 (floor) |
| `low_overlap_stress` | 0.9370 | 0.9370 | 0.005 (floor) |
| `multi_org_balanced` | 0.6307 | **0.5884** (NN) | **0.0212** |

**Critical revision:** The original gain threshold rule `0.5 × (global − additive)` produced **negative thresholds** because `additive ≈ global`. The rule was replaced: `threshold = max(0.005, 0.5 × |global − best_non_global|)`. `multi_org_balanced` is the only protocol where embedding_NN beats global mean (val_seen_rate = 1.0 on media names, so the NN baseline has non-trivial signal). This makes `multi_org_balanced` the structurally most informative protocol.

**H-BASE-01 reinterpreted:** "Beat additive" reduces to "beat a constant predictor" under cold-start. The gate was revised to "beat the best non-global baseline," which is embedding_NN on `multi_org_balanced` and global mean on all others.

---

### Stage S3 — Split Protocol Lock

**Date:** 2026-04-28 | **Status:** Approved (S3-DEC-001)

#### Decision
The original S3 rule ("30–80% chemistry overlap") was degenerate: all four candidates have val_canonical_id_seen_rate ≥ 94.7%, so the band excluded every candidate. The selection rule was revised to **power-driven**: select the protocol where the embedding_NN baseline is non-trivial, and where n_eligible genes are highest.

**Locked protocols:**
- **Primary:** `multi_org_balanced` (val={pseudo3_N2E3, BFirm, ANA3, SyringaeB728a_mexBdelta}, test={PV4}). Val_seen_rate=100% media-names, n_eligible=13,804, NN baseline beats global by 0.043 RMSE.
- **Diagnostic (not gating):** `low_overlap_stress` (val={SynE}, test={Magneto}).
- **Mandatory stress diagnostic:** `largest_by_rows` (val=Btheta, test=DvH) — added post-T1-A.2 as a required diagnostic for every tier promotion.

**Homology diagnostic locked (H-HOMO-01):** primary = similarity-bin stratification (cosine bins: [0.0, 0.5), [0.5, 0.7), [0.7, 0.85), [0.85, 1.0)); secondary = masked homology-clean subset with cosine cutoff 0.85.

---

### Stage S4 — Feature Contract (Option D)

**Date:** 2026-04-29 | **Status:** Approved (S4-DEC-002, supersedes S4-DEC-001)

#### The stressor correction
S4-DEC-001 incorrectly encoded `condition_1..4` as categorical metadata. S4-DEC-002 reformulated the schema as **Option D**: a per-experiment long-form chemistry table plus a separate wide metadata table.

#### Artifacts (artifact_id: `de21504134c84a6c`)

**`experiment_chemistry.parquet`** (long format):
- One row per `(experiment_id, canonical_id, role)` where `role ∈ {medium, stressor}`.
- `experiment_id` = SHA256-64 of `(orgId, setName, seqindex, media_key)`.
- `amount` = raw concentration if available (NaN otherwise); `log1p_amount` = train-only log1p z-score.
- Stressor strings are resolved to Canonical_IDs via `stressor_to_canonical_id.yaml`. Unmatched stressors below prevalence threshold → `<UNK_STRESSOR>`.

**`experiment_metadata.parquet`** (wide format):
- One row per experiment: `oxygen_idx`, `experiment_group_idx`, `liquid_state_idx` from frozen train-only vocabs; `temperature_c_z`, `pH_z`, `shaking_rpm_z` from train-only scalers with `*_is_finite` flags.
- `mutantLibrary`/genotype **excluded** — under organism-holdout, this column is non-informative on eval (val organisms have genotypes never seen in training).

**Unified chemistry vocab:** 423 train-seen Canonical_IDs + `<UNK>` (0) + `<UNK_STRESSOR>` (1) = **425 slots**. `val fraction_canonical_id_not_in_train_vocab = 0.0` (every val chemistry string is in the train-seen union).

**The design decision:** keeping chemistry and metadata in separate tables lets T1 iterate on each encoder independently and T2 choose how to fuse them, without premature coupling.

---

### Stage S5 — Training Recipe Lock

**Date:** 2026-04-29 | **Status:** Approved (S5-DEC-001)

#### Hypothesis (H-POLICY-01)
Does `weighted_full` (weight every row by quality, keep all rows) outperform `strict_slice` (hard-filter to high-quality rows) on the locked split?

#### Methods
Two arms × 3 seeds. Row weight: `clamp(cor12/median_cor12, 0, 1) × clamp(|t|/median_|t|, 0, 1)`. `strict_slice` applies the floor (rows below threshold dropped); `weighted_full` keeps all rows but with low weight. Fixed architecture: shallow concat-linear MLP, 8 epochs.

#### Results
| Arm | RMSE (mean ± std) | MAE | Effective train rows |
|---|---:|---:|---:|
| `weighted_full` | 0.5151 ± 0.0017 | 0.2957 | 15,800,521 |
| `strict_slice` | 0.5193 ± 0.0012 | 0.2997 | 13,878,149 |

Gap = 0.004 RMSE, 0.004 MAE — both below the locked 0.005 threshold. **Decision by tiebreaker:** `weighted_full` retains 1.9M more effective training rows at equal-or-better performance. H-BASE-01 passes for both arms (beat additive RMSE 0.632 by 0.117).

**Note (the "train RMSE > val RMSE" artifact):** `strict_slice` showed train RMSE > val RMSE, which looks like "overfitting in reverse." This is a magnitude artifact: strict filtering keeps high-|t| rows which have larger-magnitude `fit` values; their RMSE-of-fit is ~0.78 vs ~0.64 for the full val set. RMSE scales with target magnitude; the model's skill is equivalent, just applied to a different target distribution.

**H-POLICY-02 skipped:** curated organism pool testing was omitted because S1's support curation (`min_org_support_rows=50,000`) already implicitly curated the pool.

---

### Tier 1 (T1) — Representation

**Overall status:** T1-A and T1-A.2 complete (H-ENC-01 confirmed); T1-B and T1-B.3 complete (H-ENC-02 confirmed, binary encoding locked); T1-C/D/E pending.

**Fixed controls across all T1 experiments:** `multi_org_balanced` primary + `largest_by_rows` mandatory diagnostic; feature contract `de21504134c84a6c`; `weighted_full` policy; shallow concat-linear MLP; hidden=256, dropout=0.1, lr=1e-3, weight_decay=1e-4, batch_size=8192; seeds {0,1,2}; 8 epochs.

---

#### T1-A: Granularity Test (H-ENC-01)

**Date:** 2026-05-12 | **Status:** T1-DEC-001 (approved, `no_winner`; tiebreaker → multihot)

**Hypothesis:** Decomposed multihot Canonical_ID encoding beats coarse media-name-only embedding on RMSE+MAE by the S2-locked thresholds.

**Arms:**
- **A1 (media_id):** `nn.Embedding(n_train_media + 1, 425)` indexed by integer media-name index. Acts as a 425-dim lookup table for each media name.
- **A2 (multihot_canonical_id):** 425-dim binary vector aggregated from `experiment_chemistry.parquet`, including both medium and stressor chemistry.

**Results on `multi_org_balanced`:**

| Arm | RMSE (mean ± std) | MAE |
|---|---:|---:|
| media_id | 0.5177 ± 0.0010 | **0.2895** ± 0.0013 |
| multihot_canonical_id | **0.5149** ± 0.0021 | 0.2954 ± 0.0017 |

- RMSE gap: +0.0028 in multihot's favor (13% of 0.0212 threshold → **below threshold**).
- MAE gap: −0.0059 in media_id's favor (above 0.005 threshold, opposite direction).
- **H-METRIC-01 activated:** RMSE and MAE disagree. No automatic winner.

**Structural caveat:** `multi_org_balanced` has val_seen_rate=1.0 at media-name level — every val medium is in training, so the media_id encoder **never hits UNK**. The test doesn't exercise the key advantage of multihot.

**Tiebreaker decision → multihot:**
1. Multihot wins RMSE on all 4 val organisms consistently.
2. Multihot clears H-BASE-01 (beats NN baseline by 0.07 RMSE).
3. Multihot structurally generalizes to unseen media; media_id cannot.
4. Multihot is already the locked S5 substrate.

**Homology-bin crossover (important finding):**

| Cosine bin | multihot RMSE | media_id RMSE | Winner |
|---|---:|---:|---|
| [0.00, 0.50) | 0.406 | 0.398 | media_id |
| [0.50, 0.70) | 0.446 | 0.436 | media_id |
| [0.70, 0.85) | 0.498 | 0.510 | multihot |
| [0.85, 1.01) | 0.549 | 0.559 | multihot |

Crossover at cosine ≈ 0.70. Filed as a T2 motivation: a homology-gated fusion might combine the advantages of both encoders.

---

#### T1-A.2: H-ENC-01 Stress Test on `largest_by_rows`

**Date:** 2026-05-12 | **Status:** T1-DEC-002 (approved, H-ENC-01 confirmed)

**Motivation:** T1-A's `multi_org_balanced` has val_seen_rate=1.0 at media-name level. The test that matters for H-ENC-01 is a protocol where val media are **all unseen** in training. `largest_by_rows` (val=Btheta) has val_seen_rate=0% — every Btheta medium is new.

**Results:**

| Arm | RMSE (mean ± std) | MAE |
|---|---:|---:|
| multihot_canonical_id | **0.5897** ± 0.011 | **0.3675** ± 0.004 |
| media_id | 3.8451 ± 1.296 | 2.6627 ± 1.048 |
| Gap | **+3.255** | **+2.295** |

Gap = **650× the RMSE threshold (0.005)**. Statistically unambiguous.

**Mechanism confirmed:** media_id's cross-seed std = 1.30 vs multihot's 0.011. Every val row hits the single UNK token, which received no gradient during training (no train row has UNK media). The predictions are "random initialization × MLP" — completely uninformative, differing by seed only because of different random UNK initializations.

**Homology-bin crossover absent:** multihot is stable (0.52–0.62 RMSE) across all 4 cosine bins. The T1-A crossover was an artifact of full media coverage in that protocol.

**H-ENC-01 status: CONFIRMED.** "Decomposed chemistry encoding beats coarse medium-name-only encoding when val media are not in training."

**New policy locked:** `largest_by_rows` becomes a **mandatory diagnostic** for every future T1+ tier promotion report.

**Caveat filed:** Neither encoder beats the global mean on `largest_by_rows` (multihot 0.59 > global 0.567 by 0.023). The shallow concat-linear MLP cannot extract real signal from Btheta even with good chemistry features. This is a difficulty signal for T2 (fusion) and T3 (capacity), not an encoder finding.

---

#### T1-B: Numeric Transform Test (H-ENC-02)

**Date:** 2026-05-12 | **Status:** T1-DEC-003 (approved, H-ENC-02 supported; log1p tiebreaker)

**Background:** ~2% of chemistry cells in `experiment_chemistry.parquet` have a recorded `amount` value (mostly stressor concentrations). H-ENC-02 asks: does the choice of numeric transform for these values matter?

**Arms (3 × 3 seeds):**
- **raw:** amount as-is (max ≈ 2000 in the training set).
- **log1p:** log(1 + amount), the locked S5 default column `log1p_amount`.
- **bounded:** clip(amount, p1=0.0002, p99=250) then min-max to [0,1] on train-only statistics.

**Results:**

| Arm | RMSE (mean ± std) | MAE |
|---|---:|---:|
| raw | 0.5306 ± 0.0030 | 0.3000 ± 0.0026 |
| log1p | **0.5159** ± 0.0021 | **0.2940** ± 0.0014 |
| bounded | **0.5145** ± 0.0013 | 0.2940 ± 0.0012 |

- Raw is statistically worse than both compressed arms (CIs non-overlapping across all seeds): **H-ENC-02 supported**.
- log1p vs bounded: RMSE gap = 0.0014 (7% of threshold), MAE gap ≈ 0. **Statistically tied**.

**Tiebreaker → log1p:** log1p is the locked S5 default; no artifact bump needed. bounded requires re-fitting `bounded_reference_stats.json` (S4 left it with n_train=0) and bumping artifact_id for negligible gain. Effect size (0.015 RMSE vs raw) is also small, because only ~2% of cells have amounts — the transform matters only on this sparse subset.

**Note:** This decision was later superseded by T1-B.3.

---

#### T1-B.3: Controlled Binary vs log1p (Concentration-Inclusion Test)

**Date:** 2026-05-12 | **Status:** T1-DEC-004 (approved; supersedes T1-DEC-003 on encoding choice)

**Motivation:** T1-B compared *transforms of* concentration but never tested **"include concentrations at all vs presence-only."** The S4 `concentration_policy` notes that `units_1..units_4` are not applied in S4 — the `amount` values are pooled from experiments with different units (mM, mg/L, g/L, etc.) without conversion. Using them without unit normalization means the "numeric information" the model receives is physically meaningless as a concentration.

**Arms (2 × 3 seeds):**
- **binary:** every present chemistry cell = 1.0 (presence/absence only — identical to T1-A's multihot).
- **log1p:** presence = 1.0 for NaN amount; `log1p(amount)` for cells with recorded concentration.

**Results:**

| Metric | binary | log1p | Gap (binary − log1p) | Threshold |
|---|---:|---:|---:|---:|
| RMSE | 0.5150 ± 0.0020 | 0.5159 ± 0.0021 | −0.0009 (tied) | 0.0212 |
| MAE | 0.2957 ± 0.0014 | 0.2940 ± 0.0014 | +0.0017 (log1p statistically better) | 0.005 |

Bootstrap CIs: RMSE overlaps (tied); MAE disjoint by 0.0015 (log1p statistically better on central mass but 60% of threshold, below the promotion bar). H-METRIC-01 activated for the third time in T1.

**Decision: lock binary (T1-DEC-004, supersedes T1-DEC-003).**
1. No winner per S2-locked rule — RMSE/MAE disagree, neither gap exceeds threshold in the same direction.
2. **Epistemic conservatism:** including unit-mixed concentrations is not honestly describable as "concentration information." The `concentration_policy` in `feature_contract.yaml` explicitly warns against treating pooled `amount` values as cross-experiment molarities.
3. MAE advantage of log1p (0.0017) is real but sub-threshold and doesn't justify the representation complexity.
4. Binary is bit-equivalent to T1-A's multihot arm — no new diagnostic run needed.

**Residual finding:** T1-DEC-003's secondary finding — raw amounts destabilize training relative to log1p/bounded — remains valid and is the correct recommendation if concentrations are ever re-introduced after unit normalization.

---

#### T1-C: Unknown Handling (H-ENC-03) — Pending

**Hypothesis:** Explicit UNK + mask indicators for out-of-vocabulary chemistry improve robustness on novel conditions vs zero-fill. Arms: zero-fill (missing chemistry → all zeros, no flag) vs explicit UNK (missing chemistry → `<UNK>` token in vocab + binary "chemistry unknown" mask flag).

**Motivation:** ~2.7% of experiments have unmapped media (S1 fig 14); even well-covered organisms have some unmapped conditions. The explicit UNK arm signals to the model "I have no chemistry information here" vs zero-fill which is indistinguishable from "this experiment uses none of the 425 canonical chemicals."

---

#### T1-D: Experiment Metadata Bundle (H-ENC-04) — Pending

**Hypothesis:** Adding experiment metadata (oxygen level, temperature, pH, shaking RPM, experiment group, liquid state) to the condition vector improves prediction over chemistry-only features. Arms: chemistry-only multihot (binary) vs chemistry + metadata from `experiment_metadata.parquet`.

**Note:** Genotype/mutantLibrary is **excluded** from the locked metadata contract — under organism-holdout, every val organism uses a genotype never seen in training, making it structurally non-informative.

---

#### T1-E: Decomposition Mode Indicators (H-ENC-05) — Pending

**Hypothesis:** Explicitly flagging whether each Canonical_ID in an experiment came from a `physical` source (directly added, known quantity) vs an `extract` source (inferred from yeast extract composition, unknown exact quantity) mitigates over-coupling. Arms: chemistry-only vs chemistry + per-canonical representation_mode indicator.

**Motivation from S1:** Btheta has 76% physical / 24% extract mode composition in its val partition vs 40%/37%/23% physical/extract/mix in training. A model without mode flags can learn a spurious shortcut: "this experiment is probably Btheta-like because it has many extract-derived amino acids" — an organism shortcut disguised as chemistry.

---

### Tier 2 (T2) — Fusion

**Status:** Not started. Planned experiments:

**T2-A (H-FUSE-01):** Does a shallow nonlinear MLP head (1 hidden layer) beat a simple linear head given the same encoder? Expected: yes, but the gap quantifies how much nonlinearity is useful vs an affine combination of gene+condition features.

**T2-B (H-FUSE-02):** Does a two-tower architecture (separate gene encoder → 128-dim; separate condition encoder → 128-dim; late merge by concat + MLP) beat early concatenation on the held-out organisms? Two towers prevent the condition encoder from "leaking" organism-specific patterns into the gene encoder path.

**T2-C (H-FUSE-03):** Does FiLM-like condition-gating of gene features improve ranking on high-condition-variance genes? Instead of concatenating, compute `γ(condition) ⊙ h_gene + β(condition)` and pass the modulated gene features to the output layer. Tests whether the model benefits from learning "in this condition, these gene features are relevant."

**Open issue from T1-A:** The homology-bin crossover (media_id beats multihot at low cosine; multihot beats at high cosine) motivated a potential T2 addition: a **homology-gated fusion** where the gene encoder's weighting of chemistry vs gene-identity features scales with the val gene's similarity to training genes. This is not in the current T2 plan but could be added if the homology-bin analysis in T2 reports shows a systematic pattern.

---

### Tier 3 (T3) — Capacity

**Status:** Not started. Planned experiments:

**T3-A (H-CAP-01):** Depth and residuals — compare 1-layer (T1/T2 MVP), 2-layer, and 4-layer residual MLP heads. Residual connections: `h_{k+1} = LayerNorm(h_k + FF(h_k))`. Tests whether more depth extracts signal that the shallow architecture misses.

**T3-B (H-CAP-02):** Efficiency frontier — parameter/runtime vs RMSE+MAE trade-off. Pick smallest model within ε of the best.

**T3-C (conditional, H-EMB-01):** Fine-tune top-N ProteomeLM layers. Triggered only if T3-A/B plateau against null baseline delta — i.e., if the bottleneck appears to be the frozen gene representation rather than fusion/capacity.

---

### Tier 4 (T4) — Optimization & Final Locks

**Status:** Not started. Planned:

**H-LOSS-01:** MSE vs Huber loss. The `fit` distribution is heavy-tailed (S1 fig 13, QQ plot shows severe departure from Gaussian). MSE squares residuals and is disproportionately penalized by extreme rows. Huber loss = MSE for |ŷ-y| < δ, linear outside — robust to outliers beyond δ. The δ sweep is critical: too large and Huber ≈ MSE; too small and Huber ≈ MAE (unstable for central mass).

**H-TARGET-01:** Per-experiment z-score normalization. Normalize `fit` within each experiment to zero mean/unit variance before training. This removes experiment-level offset differences (some experiments run at a different absolute scale). Promotion gated on **raw-scale** metric improvement — if z-score helps optimization but the raw RMSE doesn't improve, it doesn't promote.

**Multi-seed confidence intervals:** Final architecture evaluated at 5+ seeds to compute CIs on all co-primary metrics.

---

## 4. Deferred Experiments and Open Issues

### Chemistry-Axis Generalization
All four candidate protocols have ≥94.7% Canonical_ID seen-rate (META-DEC-002). The project cannot claim "generalizes to novel chemistry" — only "generalizes to organisms and condition contexts within the v4 vocabulary." Three paths to go beyond (all deferred): (1) **chemical fingerprint encoders** (Morgan/RDKit/MACCS) that represent unseen chemicals via structural similarity to seen ones; (2) **Canonical_ID-level holdouts** in addition to organism holdouts; (3) **external validation sets** with genuinely novel media.

### Missing Gene Sequences
2,484 fitness genes have no amino acid sequence in the database — not just missing embeddings but missing aaseqs entirely (confirmed 2026-04-27). These are likely Tn-seq insertions in non-protein-coding regions, genome-version drift, or upstream pipeline gaps. Recovery requires a new aaseqs dump, which is out of scope.

### T1-A.1: Strict Stressor Ablation (Deferred)
T1-A conflated two effects: (a) decomposed chemistry generalizes better than media-name IDs, and (b) the multihot arm sees stressor chemistry that the media-id arm does not. To isolate effect (a), a third arm would concatenate `(media, condition_1, condition_2, condition_3, condition_4)` as a single string token — same stressor scope as multihot, but still no decomposition. Trigger: T1-A gap is large AND a causal "decomposed chemistry helps" claim is needed for publication.

### Bounded_reference_stats.json
S4 left `bounded_reference_stats.json` with n_train=0 (the bounded transform was never fitted). T1-B refitted this locally at `artifacts/cache/t1b/bounded_reference_stats.json`. If bounded encoding is ever revisited (e.g., after unit normalization), the S4 artifact must be regenerated with a non-zero n_train.

### The Unit-Normalization Path
`units_1..units_4` in `experiments.parquet` were not consumed in S4. This means the `amount` values in `experiment_chemistry.parquet` are in mixed units — millimolar, mg/L, g/L, µM — pooled without conversion. T1-B.3 locked binary encoding precisely because of this concern. If units are eventually parsed and concentrations normalized, T1-B and T1-B.3 should be re-run. The prior finding (raw < log1p/bounded) remains valid for any future run that includes concentrations.

---

## 5. Quality Gate Checklist

This section confirms the textbook satisfies the TEACHING_BLUEPRINT.md requirements:

**Multi-Language Reality:** The codebase is Python-only (confirmed via full file listing). Shell scripts are not part of the core pipeline — the only non-Python file is a single legacy `.sh` script in `archive/modeling/run_quality_ablation.sh` from the pre-refactor archive. All active experiment runners, data loaders, and evaluation code are Python with PyTorch, NumPy, Pandas, SciPy, Hydra, and PyArrow. There are no R, C++, or Julia files in the active codebase.

**Information Density:** Metric numbers are reproduced from decision ledger files. All gap sizes, threshold comparisons, and n_eligible counts are taken directly from decision logs.

**Mathematical Grounding:**
- RMSE and MAE: defined in §1.2.
- Within-gene Spearman: definition and eligibility criteria in §1.2; Python implementation in `src/evaluation/metrics.py::within_gene_spearman`.
- Additive baseline: `fit ~ a + α[gene] + β[condition]` OLS, with explicit cold-start collapse documented in S2.
- FiLM: `γ(condition) ⊙ h_gene + β(condition)` with learned γ, β networks.
- Huber loss: piecewise defined in §T4 section.
- Shallow MLP: layer structure documented in §2.3 with exact implementation at `src/models/architectures/shallow_mlp.py`.

**Strategic pivots documented:** 
1. Plan v1 → v2 (circular ordering, ownership overlaps — META-DEC-001).
2. S4 stressor misclassification → Option D reformulation (OPEN-001 → S4-DEC-002).
3. H-BASE-01 reinterpretation: additive ≈ global mean under cold-start, forcing a revised threshold rule (S2-DEC-001).
4. T1-DEC-003 (log1p) superseded by T1-DEC-004 (binary): unit-mixing concern + sub-threshold MAE advantage insufficient to justify concentration inclusion.

**Cross-org Spearman at noise level:** Documented in S2: additive baseline Spearman = 0.0056 on `multi_org_balanced`, 0.0098 on `largest_by_rows` — both consistent with chance under permutation null (p95 < 0.01). The 4-of-5 baseline collapse to global mean is the mechanistic explanation.

**Pending tiers (T2–T4):** Documented as planned but not run. The blueprint references failures of FiLM, z-scoring, ESM-C bypass, and T6-A fingerprints — these are experiments planned for T2-C, T4, T3-C, and a deferred fingerprint experiment respectively. None have been executed yet; the repository contains stubs and decision-ledger templates for these tiers.
