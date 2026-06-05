# RPLAN — Ranking Regime (R-Tiers)

**Status:** draft v2, 2026-05-22
**Supersedes:** RPLAN v1 (this file)
**Sibling to:** `docs/REFACTORPLAN.md` (T-tiers, pointwise MSE/MAE regime).

## 0. Why a new tier family

The T-pipeline (S0→T6) optimized RMSE/MAE on **continuous `fit`** under a
**cross-organism** holdout. T7-prep diagnostics revealed:

- Within-gene Spearman of the locked T5-A model on cross-org val: **≈ 0.045**
- Cross-replicate noise floor (per-org median Spearman): **≈ 0.43**
- Within-org cross-experiment Spearman of the same architecture: **0.071–0.077**

In the metric a biologist actually cares about — "for this gene, which
conditions stress it most?" — the T-regime model has essentially zero signal
under cross-org transfer. This is a different problem from "predict the
value of `fit` at this row," and it needs its own pre-registered pipeline.

T-tier locked decisions (T1–T6) carry over to R-tiers as **priors**, not as
locks. Each is eligible for re-examination if (and only if) theory predicts
the ranking objective will change the answer. See §10 for the carry-over table.

## 1. Pipeline shape

```
R0  →  R-LOCK-{1,2}  →  R-LOCK-{3,4}  →  R1  →  R2  →  R3+
       (parallel)        (depend on 1,2)
```

| Phase | Concern | Output | Depends on |
|---|---|---|---|
| **R0** | Data characterization for the ranking regime | 11 figures, candidate frontier, split previews | — |
| **R-LOCK-1** | Eligibility filter (`m_min`, `IQR_min`, weighting scheme) | `data_contract/ranking/eligibility_policy.yaml` | R0 |
| **R-LOCK-2** | Split protocol (within-org cross-experiment) | `data_contract/ranking/split_protocol.yaml` | R0 |
| **R-LOCK-3** | `RankingBatch` contract + default sampler mode + run-manifest v2 | `src/data/datasets/ranking_batch.py`, `data_contract/schemas/run_manifest_v2.schema.json` | R-LOCK-1, R-LOCK-2 |
| **R-LOCK-4** | Primary metric + ranking baseline + noise-floor protocol | `data_contract/ranking/metric_contract.yaml` | R-LOCK-1, R-LOCK-2 |
| **R1** | Representation retest — chemistry encoder (fingerprints vs multihot) under ranking | locked chemistry encoder for R-regime | R-LOCK-{1..4} |
| **R2** | Architecture retest — fusion topology under ranking | locked fusion for R-regime | R1 |
| **R3+** | R-LOSS (loss family), R-CAP (capacity), R-EMB (embedding), R-COND (condition-discriminability weighting), R-SAMPLE (org-balanced sampling), R-META (metadata features: temp/pH/expGroup), R-SNR (SNR-based eligibility), R-CURRIC (iterative pseudo-labeling) — one axis per tier, run as needed | tier-specific | R2 |

**Parallelism:** R-LOCK-1 and R-LOCK-2 are independent — both depend only on R0
outputs and can be done in parallel (or even folded into a single decision-day).
R-LOCK-3 and R-LOCK-4 then run in parallel themselves once both 1 and 2 are
done.

**Exit criterion from R-LOCK phase:** all four locks have approved decision-ledger
entries AND `RankingBatch` is unit-tested AND the metric helper is unit-tested.
Only then does R1 start.

## 2. The Ranking Contract (locked once, applied everywhere)

Every R-tier holds these constant unless that tier is *explicitly* testing the
field. Every field has a default proposal here; R-LOCK-X finalizes the value.

### 2.1 Eligibility & weighting (LOCKED in R-LOCK-1-DEC-001, approved 2026-05-24)

| Field | Locked value | Notes |
|---|---|---|
| `spread_metric` | **`tail_g = p95 − p5`** (NOT IQR) | IQR misses sparsely-conditional genes (a gene essential in 3–5 of 100 conditions has `IQR ≈ 0` but is highly rankable). `tail_g` catches both broadly- and sparsely-conditional genes. |
| `spread_metric_diagnostic` | `iqr_g` | Retained for analysis only. Per-gene `(tail_g, IQR_g)` jointly classifies genes: high tail + high IQR → broadly conditional; high tail + low IQR → sparsely conditional; low tail → genuinely non-conditional. |
| `m_min` | `10` | Defensive floor; functionally a no-op for promotion-target orgs (median `m_g` >> 10). Effectively excludes tiny-org genes from training, consistent with R-LOCK-2's diagnostic-only status for those orgs. |
| `tail_min_scheme` | **per_org** | Required by 5.4× per-org noise spread (R0 fig 04). Global threshold over-includes noisy orgs and over-excludes clean ones. |
| `tail_min_floor` | `0.20` | Cleanest orgs hit this floor. |
| `tail_min_noise_coef` | `0.80` | `tail_min_org = max(0.20, 0.80 × (1 − r_replicate_org_median))`. Calibrated as IQR equivalent rescaled by `tail/IQR ≈ 2.7×`. |
| `tail_min_computed_on` | `train_rows_only` | No leakage from val. |
| `weighting_scheme` | `tail_x_m` | `w_g = clip((tail_g_train − tail_min_org) / (tail_ref_org − tail_min_org), 0, 1) × min(m_g_train/m_min, 1)`, with `tail_ref_org` = per-org p75 of `tail_g_train`. |
| `train_eligibility` | `weighted_all` | Low-`tail_g` genes contribute at low weight (model-stability prior). |
| `val_eligibility` | `hard` filter | Per-gene Spearman is undefined for constant fit; val must hard-filter. `m_min_val = 5`; tail threshold same per-org formula on val rows. |

### 2.2 Split protocol (LOCKED in R-LOCK-2-DEC-001, approved 2026-05-24)

| Field | Locked value | Notes |
|---|---|---|
| `holdout_unit` | **`condition_key`** (NOT `expName`) | Holding out whole `expName`s would leak: replicate siblings share a condition but have different `expName`s, so they'd land in different partitions. Holding out at the `condition_key` level pulls all replicates of a held-out condition into val together. |
| `condition_key` | **`(expDesc, media, temperature)`** | Temperature added after R0 audit (2.4% of `(expDesc, media)` groups hid distinct temperatures). |
| `holdout_scope` | `within_organism` | Each org has its own train/val split. No cross-org transfer claim. |
| `holdout_fraction` | **0.20** (NOT a fixed `k`) | Fraction of `condition_key`s held out per org; clipped to `[min_holdout=3, max_holdout=30]`. Fixed-k was rejected (17× org-size spread makes any fixed k wrong). 38/44 orgs retain ≥100 val-eligible genes. |
| `stratify_by` | `expGroup` (random fallback) | Avoid val being all one condition class. Feasible for all promotion-target orgs (R0 fig 11). |
| `diagnostic_splits` | `cell_holdout` (easy ceiling) + `stressor_class_holdout` (hard floor) | (1) `cell_holdout`: random `(gene, condition)` cells held out from known conditions — bounds "how much of the gap is from condition-novelty vs raw signal limits." (2) `stressor_class_holdout`: hold out one entire chemical family per org — analog of S3's `low_overlap_stress`. Neither is a promotion gate. |
| `cross_org_diagnostic` | required | Run the T-regime `multi_org_balanced` split as a *drift monitor*. Not gated. |
| `condition_key_definition` | `(expDesc, media, temperature)` | NOT `(expDesc, media)`. R0 audit (2026-05-24) found 53 of 2246 (2.4%) `(expDesc, media)` groups contain assays at distinct temperatures — different conditions, not replicates. pH excluded from key (53% null overall, doesn't vary within `(expDesc, media)`). Aerobic excluded (doesn't vary within group). |

### 2.2.1 ⚠ COLD-COLUMN PROPERTY (critical — verified empirically 2026-05-25)

**Under the locked condition-holdout split, val conditions are 100% disjoint
from train conditions** (verified: 0 of 78 val conditions appear in train, on
DvH/Btheta/Caulo; this is true *by construction* — we hold out whole condition
columns). Every held-out condition is **cold**: no gene has observed it in
training. This single fact has large consequences any future audit MUST respect:

1. **`per_condition_train_mean` (the original H-RANK-01) is UNDEFINED** for the
   primary split — there is no train mean for an unseen condition. It returned
   `NaN` for all 28,005 val genes when actually run. The cold-condition null is
   redefined to `chemistry_nearest_condition_profile` (see §2.4).
2. **Pure matrix factorization / collaborative filtering is INVALID** for the
   primary split — it cannot place a column with zero observed entries. MF is at
   most an *optional* baseline on the warm-column `cell_holdout` diagnostic.
3. **The task is cold-start (inductive) matrix completion**, NOT standard
   (warm-column) matrix completion. The "it's just matrix completion" critique
   applies only to `cell_holdout`. The primary task genuinely requires condition
   side-features (chemistry) to place novel conditions — a more defensible setup.
4. **Valid baselines are SPLIT-SPECIFIC** (declared in `metric_contract.yaml`):
   primary → chemistry-kNN; `cell_holdout` → MF + per-condition-mean; `cold_gene`
   → per-condition-mean (conditions warm there). Do not apply a baseline to a
   split where its required information is absent.

This was discovered by *computing baselines on the real materialized split*,
not by reading docs — a reminder that doc-level review cannot catch
data-dependent invariants. See R-LOCK-4-DEC-002 and
`research_log/audits/2026-05-25_ranking_pivot_audit.md`.

### 2.3 Data feeding (locked in R-LOCK-3)

| Field | Default | Notes |
|---|---|---|
| `replicate_handling_train` | per-replicate rows | Each replicate is a noisy observation of the same `(gene, condition)`. Don't collapse. |
| `replicate_handling_val` | mean-pool within `(orgId, gene_key, condition_key)` | Per-replicate noise contributes to the metric. Pool replicate `expName`s of the same `condition_key` first. (NOTE: pooling within `expName` would be a no-op — `expName` *is* the replicate id; the pooling key must be the condition, matching `ranking_metrics.py` and the metric contract.) |
| `sampler_default` | `pointwise` | For R1, R2: preserves backward compat with T5-A training loop. Pairwise/listwise are tested in R-LOSS. |
| `sign_convention` | model predicts `fit` (low → essential) | Pairwise margin sign = `sign(fit_i − fit_j)` so the model is rewarded for *predicting* the same ordering. |
| `ranking_batch_shape` | `{pointwise: (B,), pairwise: (B, 2), listwise: (B, L) + mask}` | Single model `forward(gene_emb, cond_feat) → scalar`; sampler reshapes. |

### 2.4 Metric + baseline + noise floor (locked in R-LOCK-4)

| Field | Default | Notes |
|---|---|---|
| `primary_metric` | `within_gene_spearman` (mean over eligible val genes) | Tie handling: average ranks (scipy default). |
| `secondary_primary` | `within_gene_kendall` (mean over eligible val genes) | Co-primary. Both must be reported. |
| `aggregation` | `mean of per-gene values` (global, NOT balanced per-org) | Default. R-LOCK-4 may switch to per-org-balanced if R0 shows extreme org imbalance. |
| `bootstrap_n` | 1000 | Over eligible val genes (with replacement). |
| `ci_level` | 0.95 | Bootstrap percentile CI. |
| `ranking_baseline` (H-RANK-01) | `per_condition_train_mean` — for each val condition, predict the mean fit of train genes in that condition; assign every val gene that value | Model must beat this on `primary_metric` to be promotable. |
| `noise_floor` (PRIMARY, task-relevant) | **per-gene cross-condition Spearman across replicate pairs**: for each eligible val gene with ≥ 2 replicate measurements at ≥ 5 distinct conditions, Spearman across conditions of `rep_A fit` vs `rep_B fit`; median across genes | This is the actual upper bound on `within_gene_spearman` from technical replicates. Reported as horizontal reference line on all R-tier figures. PROJECT-LEAD DECISION 2026-05-24. |
| `noise_floor_proxy` (SECONDARY, retained) | per-val-row cross-gene Spearman within a single condition between replicate assays (`r_replicate` from R0 fig 04) | Cross-gene within-condition variant. Retained as diagnostic because it's computable for more orgs than the task-relevant version. |
| `cross_org_drift` | T-regime `multi_org_balanced` within-gene Spearman | Secondary metric, drift monitor, not gated. |
| `promotion_delta` | Δ-Spearman ≥ **0.01** AND Δ-Kendall ≥ **0.008** (both with disjoint 95% bootstrap CIs) | R-LOCK-4 may revise after R0 reveals metric variance. |

### 2.5 Training defaults (carry-over from T5-A unless tested)

| Field | Default | Set in |
|---|---|---|
| `optimizer` | Adam | T4-DEC-001 |
| `lr` | 1e-3 constant | T4-DEC-002 |
| `epochs` | 8 | T4-DEC-002 |
| `batch_size` | inherits T5-A | (verify in R-LOCK-3) |
| `weight_decay` | 0 | T4 |
| `gradient_clip` | none | T4 |
| `seeds` | `[0, 1, 2]` | conventional |

Any R-tier may test these as its primary axis (then they're not "constant" for
that tier). Otherwise they're held.

### 2.6 Run manifest v2 fields (locked in R-LOCK-3)

Extends `run_manifest_v1.schema.json`. New fields, all required for R-regime runs:

- `regime: "R"`
- `ranking_loss: <pointwise_mse | pairwise_margin | listmle | softrank>`
- `sampler_mode: <pointwise | pairwise | listwise>`
- `eligibility_filter_hash: <sha256 of locked eligibility policy>`
- `split_protocol_id: <ranking-split id, separate from T-regime split ids>`
- `primary_metric_name: <metric id>`
- `primary_metric_value: float`
- `primary_metric_ci_low: float`
- `primary_metric_ci_high: float`
- `ranking_baseline_value: float`
- `ranking_baseline_beats: bool`
- `noise_floor_value: float`
- `cross_org_drift_value: float | null`
- `n_eligible_val_genes: int`
- `n_val_rows_after_pooling: int`

Old T-regime runs continue to validate against v1; R-regime runs validate
against v2. Dispatcher inspects `cfg.regime` to pick the schema.

## 3. Hard rules (carry-over and additions)

Applies to every R-tier:

- **No promotion** without a pre-declared decision-ledger entry that names
  the exact promotion deltas and the baseline that must be beaten.
- **Train-only preprocessing.** Vocab, weights, eligibility thresholds all
  fit on train rows.
- **Co-primary metrics:** within-gene Spearman + within-gene Kendall's τ.
  Neither may be omitted.
- **Ranking baseline gate (H-RANK-01):** any model that does not beat
  `per_condition_train_mean` on the primary metric is ineligible for promotion.
- **Noise floor reporting:** every R-tier figure that shows a Spearman value
  must also show the cross-replicate noise floor from §2.4.
- **Denominator parity:** model and baseline scored on the exact same
  eligible val gene set.
- **No cross-regime metric transfer.** T-regime RMSE/MAE does not count
  toward R-regime promotion (and vice versa).
- **Every R-run logs the v2 manifest fields (§2.6).** Manifests with
  `regime: R` and missing required fields are rejected.

## 4. R-Tier ownership (no overlap)

- **Eligibility filter + weighting** → R-LOCK-1 only. NOT R1+.
- **Split protocol** → R-LOCK-2 only. NOT R1+.
- **Sampler default + RankingBatch contract + manifest v2** → R-LOCK-3 only.
- **Primary metric + ranking baseline + noise-floor reporting** → R-LOCK-4 only.
- **Chemistry encoder under ranking** → R1 only.
- **Fusion topology under ranking** → R2 only.
- **Loss family under ranking** → R-LOSS (R3+) only.
- **Capacity / depth / width under ranking** → R-CAP (R3+) only.
- **Embedding (ProteomeLM layer, fine-tune, adapter variants)** → R-EMB (R3+) only.
- **Condition-level loss weighting** → R-COND (R3+) only. NOT R-LOSS.
- **Cross-org / per-org training-sample balancing** → R-SAMPLE (R3+) only.
- **Iterative self-training / pseudo-labeling** → R-CURRIC (R3+) only. NOT R-LOSS.
- **Condition-metadata features (temperature, pH, expGroup, aerobic)** → R-META (R3+) only. NOT R1.
- **SNR-based eligibility / weighting** → R-SNR (R3+) only. NOT R-LOCK-1.

## 5. R0 — Data Characterization

> **Post-execution note (2026-05-25).** R0 ran and the locks diverged from the
> originally-planned framing below in three ways that this section preserves
> for the historical record but which the **locked artifacts supersede**:
> (1) the primary eligibility metric is **`tail_g = p95 − p5`**, NOT IQR — IQR
> discards sparsely-conditional genes (R-LOCK-1); IQR is retained only as a
> diagnostic. (2) **Figure 07** (low-IQR Jaccard) was **dropped** — `gene_key`
> is org-prefixed so the cross-org Jaccard is zero by construction (H-R0-03
> deferred, needs homology). (3) **Figure 11** (`expgroup_coverage`) was
> **added** for R-LOCK-2 stratification feasibility. The condition key gained
> **temperature**. So the IQR-worded hypotheses and the 10-figure list below
> describe the *plan*; the *outcome* is tail_g-based with 11 figures (01–06,
> 08–11).

**Hypotheses (informational; not promotion gates — IQR wording is historical, see note):**

| ID | Statement | Tested by |
|---|---|---|
| H-R0-01 | ≥ 30% of genes per organism carry rankable signal (m_g ≥ 10, IQR_g ≥ noise floor) | analysis F (eligibility frontier) |
| H-R0-02 | Within-org cross-experiment splits at k ∈ {1..5} preserve ≥100 val-eligible genes in ≥60% of orgs | analysis J (split feasibility) |
| H-R0-03 | Low-IQR genes are partially shared across organisms (Jaccard ≥ 0.2) → weighting > filtering | analysis H |
| H-R0-04 | Per-org cross-replicate Spearman median spread > 0.1 → single global IQR_min won't fit all orgs | analysis D |

**Contingencies:**

| If | Then |
|---|---|
| H-R0-01 fails (<30% genes eligible at reasonable threshold in most orgs) | R-LOCK-1 drops hard filtering entirely, uses weighting-only with a higher floor `τ`. Decision documents the trade-off. |
| H-R0-02 fails (most orgs lose too much val signal at any k) | R-LOCK-2 considers experiment-level subsampling per org (oversample low-data orgs) or drops smallest orgs from primary; documents which orgs are diagnostic-only. |
| H-R0-03 fails (low-IQR genes are org-specific) | Eligibility filter applied per-org rather than globally. R-LOCK-1 documents. |
| H-R0-04 fails (noise floor consistent across orgs) | Single global IQR_min OK; simplifies R-LOCK-1. |

**Required analyses (per organism, then aggregated):**

| ID | Analysis | Replicate grouping |
|---|---|---|
| A | Conditions per gene `m_g` | n/a |
| B | IQR per gene `IQR_g` (median-pooled over replicates) | per `(org, exp, gene)` median |
| C | Joint `(m_g, IQR_g)` density per org | n/a |
| D | Replicate noise floor per org | `(orgId, expDesc, media)` groups with ≥2 distinct `expName`s — pair the first two `expName`s, compute cross-gene Spearman per pair |
| E | Signal-to-noise per gene per org | `IQR_g / (1 − r_replicate_org_median)` |
| F | Eligibility frontier across `(m_min, IQR_min)` grid | n/a |
| G | Experiment structure per org | n/a |
| H | Low-IQR gene Jaccard across orgs (bottom-decile sets) | n/a |
| I | Condition discriminability (per-experiment cross-gene IQR) | n/a |
| J | Split feasibility preview at `k ∈ {1, 2, 3, 5}` held-out exps per org | n/a |

**Figures produced** (under `research_log/figures/r0_data/`) — 11 total
(07 dropped, 11 added; see post-execution note above):

`01_m_per_gene_per_org.png` · `02_spread_per_gene_per_org.png` (tail_g + IQR) ·
`03_joint_m_iqr.png` · `04_replicate_noise_floor.png` ·
`05_signal_to_noise.png` · `06_eligibility_frontier.png` (on tail_g) ·
~~`07_low_iqr_overlap.png`~~ (dropped) · `08_experiment_structure.png` ·
`09_condition_discriminability.png` · `10_split_feasibility.png` ·
`11_expgroup_coverage.png`

Each PNG accompanied by a sibling CSV.

**Emits:**

- `data_contract/ranking/r0_candidates.yaml` — surviving eligibility and split
  candidates for R-LOCK-{1,2}.
- `research_log/tier_reports/r0_data_characterization.md` — narrative.

**Promotion rule:** R0 emits candidates; promotion is the R-LOCK decisions, not
R0 itself.

## 6. Dev subset (cross-cutting)

For fast iteration during code development (not for promotion runs), every
R-tier may use a **dev subset**:

- One organism: the *median-sized* organism by `(n_genes × n_experiments)` —
  locked once in R-LOCK-2 to a specific `orgId`.
- One seed: `0`.
- Same split protocol (just one org).

Dev-subset runs are NOT eligible for decision-ledger entries. They are for
debugging, not for promotion. Promotion runs always use the full multi-org
dataset.

## 7. R1 — Representation retest (chemistry)

**Hypothesis H-R-CHEM-01:** Under within-org cross-experiment ranking,
structural fingerprints (Morgan / RDKit / MACCS) outperform the 425-dim
multihot baseline, reversing the T6-A signal (which was measured under MSE +
cross-org).

**Theoretical justification:** ranking across chemically similar perturbations
rewards encoders that place similar molecules near each other in feature space.
Multihot has zero structural prior; fingerprints are structurally smooth. Under
MSE + cross-org, the gene-mean component of fit dominated loss and absorbed
chemistry signal; under per-gene ranking, the gene-mean is removed by
construction and chemistry has to do the work.

**Arms (carry-over from T6-A):**

- `multihot_425` (control, T-regime winner)
- `morgan_2048`
- `rdkit_2048`
- `maccs_167`
- `morgan_plus_multihot`
- `maccs_plus_multihot`

**Seeds:** `[0, 1, 2]`. **Total runs:** 18.

**Held constant** (from the Ranking Contract):

- Eligibility: R-LOCK-1
- Split: R-LOCK-2
- Sampler: R-LOCK-3 default (`pointwise`)
- Metric: R-LOCK-4
- Architecture: T5-A locked `adapter_residual_mlp` (Linear(1152,1024) → ReLU →
  Dropout → Linear(1024, 512), then T3 head 2-layer ResidualMLP hidden=512)
- Loss: MSE (T-regime carry-over; not the test axis)
- Optimizer/LR/epochs: T4 defaults

**Promotion:** Δ-Spearman ≥ 0.01 AND Δ-Kendall ≥ 0.008 vs `multihot_425` AND
beats H-RANK-01 baseline.

## 8. R2 — Architecture retest (fusion)

**Hypothesis H-R-FUSE-01:** A condition-modulation fusion (FiLM or
gene-reduce+concat) beats early concat under the ranking objective.

**Mechanism (corrected 2026-05-25 audit).** The earlier framing — "early-concat
absorbed the gene-mean, ranking removes it" — was a non-sequitur: *any*
architecture that takes the gene embedding as input (including FiLM, whose β
term is itself a per-(gene,condition) additive shift) can fit a per-gene
offset. The correct mechanism is about **multiplicative gene×condition
interaction**: within-gene ranking is invariant to per-gene additive/monotone
offsets, so the metric rewards *how a condition reshapes a gene's profile
relative to its own baseline* — a multiplicative interaction that FiLM/gating
parameterize directly and parameter-efficiently, whereas early-concat must
learn it implicitly through the MLP. This is falsifiable: if interaction
structure is weak, early-concat ties or wins. (T2-C context: FiLM lost narrowly
under T-regime MSE — RMSE Δ 0.003 — where fitting the gene-mean *helped* the
loss; under ranking the gene-mean is irrelevant to the metric, changing the
trade-off.)

**Arms:**

- `early_concat` (control, T2-DEC-001 winner): `cat([gene_emb, chem]) → T3 head`
- `gene_reduce_concat`: `Linear(gene_emb, k=64) → cat with chem → T3 head`
  (reduces gene-emb dominance; chem feature weight per parameter increases)
- `film`: `gene_emb * (1 + gamma(chem)) + beta(chem)` then through T3 head
  (gamma/beta are 2-layer MLPs)
- `gated`: `gene_emb * sigmoid(W·chem)` then concat with chem, then T3 head

**Seeds:** `[0, 1, 2]`. **Total runs:** 12.

**Held constant:**

- Eligibility, split, sampler, metric: R-LOCK locks
- Chemistry encoder: R1 winner
- Capacity (T3 head dimensions): inherited
- Loss: MSE
- Optimizer/LR/epochs: T4 defaults

**Promotion:** same as R1.

## 9. R3+ — Deferred axes

Run only if R1 / R2 signal warrants further investigation. Each is one tier:

- **R-LOSS** (H-R-LOSS-01): MSE vs MSE + λ·pairwise_margin vs ListMLE vs SoftRank.
  Requires sampler-mode change; uses RankingBatch (R-LOCK-3) variants.
- **R-CAP** (H-R-CAP-01): depth/width revisit at the R-regime architecture.
- **R-EMB** (H-R-EMB-01): ProteomeLM layer revisit; adapter variants; fine-tune
  top N layers.
- **R-SAMPLE** (H-R-SAMPLE-01): per-org-balanced sampling vs row-proportional;
  effect on Spearman aggregation.
- **R-COND** (H-R-COND-01): Condition-discriminability weighting in the loss.
  Hypothesis: weighting each `(gene, condition)` training pair by the train-set
  cross-gene IQR of that condition (low-discriminability conditions contribute
  less gradient) improves within-gene Spearman over uniform per-condition
  weighting. R0 fig 09 (`condition_discriminability`) shows the per-org
  distribution of cross-gene IQR per condition — the right-tail conditions
  (high cross-gene spread) are the ones any per-gene ranking task actually
  benefits from. Tests two flavors: (a) soft weighting
  `w_c = clip(cond_iqr / cond_iqr_ref, 0, 1)` where `cond_iqr_ref` is the
  per-org p75 of cross-gene condition IQR, (b) hard drop of conditions whose
  `cond_iqr` is below `floor` (the cross-org-replicate noise scale). Final
  per-row loss weight is `w_g × w_c` (R-LOCK-1's gene weight composed with
  R-COND's condition weight). Held constant: R1 chemistry winner, R2 fusion
  winner, R-LOCK eligibility/split/sampler.
- **R-META** (H-R-META-01): Add condition metadata features (temperature, pH,
  aerobic, expGroup) alongside chemistry. T-regime T1-D tested this and found
  "metadata sub-threshold but stabilizes seed variance" (T1-DEC-006). Under
  ranking objective the signal may differ. Arms: (a) chemistry only (R1
  winner; control), (b) chemistry + temperature, (c) chemistry + temperature
  + expGroup one-hot, (d) chemistry + temperature + expGroup + pH (where
  pH is non-null; UNK token elsewhere). Caveat: temperature is already part
  of the condition key (R-LOCK-2), so its inclusion as a feature tests
  whether the model needs to *see* it, not whether it distinguishes
  conditions.
- **R-SNR** (H-R-SNR-01): SNR-based eligibility vs IQR-based eligibility
  (R-LOCK-1 alternative). Arms: (a) R-LOCK-1 default (IQR or tail-based
  weight × m-floor), (b) hard SNR filter at threshold τ (e.g., train only
  on genes with `SNR ≥ 1.0`), (c) continuous SNR weighting `w_g = clip(snr_g, 0, k)`.
  Sample counts available per org (R0 fig 05): ranges from 195 genes
  (Cup4G11) to 3,143 (ANA3) at `SNR ≥ 1.0`. Constraint: only valid for
  23 of 48 orgs with replicate data; rest fall back to default.
- **R-CURRIC** (H-R-CURRIC-01): Iterative pseudo-labeling — train on high-IQR
  genes only, apply to mid-IQR genes to generate "refined" targets, retrain on
  combined set. Hypothesis: this improves within-gene Spearman on a held-out
  high-IQR val subset beyond what the R-LOCK-1 continuous IQR weighting
  achieves in a single training run. Risk: confirmation bias if the
  high-IQR-trained model has systematic feature blind spots. Only run if R1/R2
  plateau and the IQR-weighting in R-LOCK-1 is shown not to capture this gain.

## 10. T-tier carry-over disposition

| T-tier lock | Disposition in R-regime | Reason |
|---|---|---|
| T1 (425-dim multihot chemistry) | **Re-tested in R1** | Theory: fingerprints may win under ranking. |
| T2 (early concat) | **Re-tested in R2** | Theory: FiLM may win under ranking (gene-mean removed). |
| T2 loser (two-tower) | **Not retested** | Decisively lost on both empirical and theoretical grounds. |
| T3 (2-layer, 512 hidden) | Carry over; revisit in R-CAP only if R1/R2 plateau | Likely still good; not the bottleneck. |
| T4 (MSE, raw targets) | MSE **carries over** for R1/R2; loss family re-tested in R-LOSS. Target normalization is moot under ranking (invariant to monotone transforms). | T4-B (z-score) loser is dead under ranking. |
| T5 (adapter_1024_proj) | Carry over; revisit in R-EMB only if signal warrants | Strong prior. |
| T6 (multihot under MSE+cross-org) | **Re-tested in R1** | This is R1's whole point. |

## 11. Decision log

Decisions go in:

- `research_log/decisions/r0/R0-DEC-NNN.md`
- `research_log/decisions/r_lock/R-LOCK-{1,2,3,4}-DEC-NNN.md` (flat directory)
- `research_log/decisions/r1/R1-DEC-NNN.md`
- `research_log/decisions/r2/R2-DEC-NNN.md`
- `research_log/decisions/r3/...`

Using `research_log/decisions/decision_template.md` with `regime: R` and the
R-regime field substitutions called out in the template's inline comments.

## 12. Out of scope for R-tiers

- **Cross-organism generalization claim.** Stays a T-tier diagnostic (reported
  as drift monitor in §2.4). The R-tier generalization claim is
  **within-organism, cross-experiment** only.
- **Absolute `fit` prediction quality** (RMSE/MAE). Reported as secondary
  drift monitor only.
- **Hyperparameter tuning beyond named arms.** Each R-tier tests exactly its
  one axis; LR/batch/etc. carry over.

## 13. Visualization policy

Every R-tier emits figures under `research_log/figures/<r_tier_id>/`
(e.g. `r1_chemistry/`, `r2_fusion/`) as `NN_descriptive_name.png` + sibling
CSV. Every plot of a Spearman or Kendall value must show:

- The bootstrap CI (95%) as error bars or shaded band.
- The H-RANK-01 baseline as a horizontal line.
- The noise floor (cross-replicate Spearman on val) as a horizontal line.
- The T-regime cross-org diagnostic value as a dashed line (drift monitor).

## 14. Glossary

- **Eligible gene** — a gene that passes the locked eligibility filter on the
  data being scored (train- or val-eligible may differ; see §2.1).
- **Within-gene Spearman** — for each eligible val gene, Spearman correlation
  between predicted `fit` and observed `fit` across that gene's val conditions.
- **H-RANK-01 baseline** — predicts each val condition's *train-genes mean*
  for every val gene. All val genes get the same *prediction vector* (the
  condition-mean profile), but each gene's *within-gene Spearman* is that
  shared profile correlated against that gene's own observed fit — so the
  per-gene Spearman **varies across genes** (it is NOT constant): genes whose
  response tracks the bulk stress response score high, idiosyncratic genes
  score low. The baseline value is the mean of those per-gene Spearmans.
  Beating it requires gene-specific deviation from the population profile —
  exactly the gene×condition interaction the task targets.
- **Noise floor** (task-relevant, primary) — for each val gene with ≥2
  replicate `expName`s sharing a `condition_key = (expDesc, media,
  temperature)` at ≥5 distinct conditions, the per-gene Spearman across
  conditions of replicate-A fit vs replicate-B fit. Median across eligible
  val genes. (A secondary cross-gene within-condition *proxy* — the R0 fig 04
  number — is also reported; see §2.4.)
- **Cross-org drift** — within-gene Spearman on the T-regime
  `multi_org_balanced` split using the *same* trained R-regime model. Tracks
  whether R-regime gains harm cross-org transfer.
