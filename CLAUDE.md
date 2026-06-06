# Project: ConditionalGeneEssentiality

Predicting conditional gene essentiality from Tn-seq fitness data using frozen
ProteomeLM gene embeddings + condition (media chemistry) features.

## Research goal & publication framing (locked 2026-05-25)

**Primary scientific question:** *Within a known organism, can frozen
protein-language-model embeddings + media-chemistry features rank a gene's
conditional essentiality across **entirely novel conditions** — better than a
chemistry-similarity baseline that uses condition features but learns no
gene-specific interaction?* Framed as "find the top stressors for a gene."

**The task is COLD-START over conditions (critical — see RPLAN §2.2.1).** The
locked condition-holdout split makes val conditions 100% disjoint from train
(verified: 0/78 overlap). So this is NOT standard matrix completion: a held-out
condition has zero observed entries, so **pure matrix factorization / collaborative
filtering cannot solve it** — you genuinely need condition side-features
(chemistry) to place a never-seen condition. This is **inductive (cold-start)
matrix completion with side information**, a harder and more defensible setup
than warm-column completion.

**Three claims the paper must support (define success):**
1. **Beat the chemistry-similarity baseline (chemistry-kNN)** on within-gene
   ranking / top-stressor retrieval — otherwise the learned gene×condition
   interaction adds nothing over "this new condition behaves like its nearest
   known chemistry." (Matrix factorization is NOT the primary competitor — it
   cannot predict cold columns; it is at most an *optional* baseline on the
   warm-column `cell_holdout` diagnostic.)
2. **Report the cold-gene split** (held-out whole genes) so reviewers see the
   *inductive-over-genes* generalization, not just per-gene-offset memorization.
3. **Frame the cross-organism failure as a finding** (T-regime: Spearman ≈
   noise on held-out orgs), not hidden — the cross-org drift monitor is the
   evidence of honesty.

**Honest venue scope:** a solid *applied / computational-biology* contribution
(Bioinformatics, Cell Systems, ISMB, NeurIPS-bio workshops). **Not** a top-tier
ML-methods paper as scoped — the method is an MLP on frozen features; the
novelty is biological, not algorithmic. Direct precedent: drug-response
prediction on DepMap/GDSC/CCLE matrices — specifically their **leave-drugs-out /
cold-start** evaluation setting (the analog of our cold conditions), which needs
drug features and is the right comparison, NOT the random-cell-holdout setting.

**Two active regimes:**
- **T-regime (REFACTORPLAN):** pointwise MSE/MAE regression on continuous `fit`
  under cross-organism holdout. Completed through T6-A.
- **R-regime (RPLAN):** per-gene ranking of conditions under within-organism
  cross-experiment holdout. Forked from T7-prep diagnostics; this is where new
  modeling work happens.

## Active branch

`refactor` — all new work goes here. `main` has the pre-refactor code.

## Governing plans

- **`docs/REFACTORPLAN.md` (v2)** — T-tier pipeline (S0..S5, T1..T6). T-regime is
  considered complete at T6-A; the within-gene Spearman of its best model
  (≈ 0.045 vs noise-floor ≈ 0.43) motivated the R-regime fork.
- **`docs/RPLAN.md` (v1)** — R-tier pipeline (R0, R-LOCK-1..4, R1, R2, R3+).
  Read this for any ranking-regime work.
- v1 of REFACTORPLAN archived at `archive/docs/REFACTORPLAN_v1.md`.

## Stage / Tier pipeline

**T-regime (REFACTORPLAN, locked at T6-A):**
```
S0 → S1 → S2 → S3 → S4 → S5 → T1 → T2 → T3 → T4 → T5 → T6 → [T7-prep → forked to R]
```

**R-regime (RPLAN, active):**
```
R0 → R-LOCK-1..4 → R1 → R2 → R3+
```

| Phase | Concern | Status |
|---|---|---|
| S0 | Reproducibility & Governance (smoke pipeline, run manifest, v4 verification) | **approved** (S0-DEC-001) |
| S1 | Data Characterization → emits candidate protocols | **approved** (S1-DEC-001); H-HOMO-01 triggered at 1.6σ |
| S2 | Evaluation Trustworthiness (null baselines + power report per candidate) | **approved** (S2-DEC-001); 4/4 protocols → primary; H-BASE-01 reinterpreted (beat best-non-global, not additive) |
| S3 | Split Protocol Lock | **approved** (S3-DEC-001); primary=`multi_org_balanced`, diagnostics=`low_overlap_stress` + homology (0.85 cutoff) |
| S4 | Feature Contract (Option D: chem long + metadata wide) | **approved** (S4-DEC-002 supersedes S4-DEC-001); artifact_id=`de21504134c84a6c` |
| S5 | Training-Recipe Lock (row-quality + organism pool) | **approved** (S5-DEC-001); policy=`weighted_full`, organism_pool=`full` |
| T1 | Representation winner | **COMPLETE.** T1-A (T1-DEC-001), T1-A.2 (T1-DEC-002), T1-B (T1-DEC-003), T1-B.3 (T1-DEC-004) complete. T1-C **skipped** (T1-DEC-005: zero chemistry unknown rate). T1-D no_winner (T1-DEC-006: metadata sub-threshold but stabilizes seed variance). T1-E no_winner (T1-DEC-007: mode flags negligible). **Locked representation: 425-dim binary multihot over Canonical_ID vocab, chemistry only.** No metadata, no concentrations, no mode flags. `largest_by_rows` mandatory diagnostic for all future T1+ promotions. |
| T2 | Fusion winner (all topology decisions live here, not T3) | **COMPLETE** (T2-DEC-001). T2-A: MLP >> linear (gap 0.095). T2-B: early concat >> two-tower (gap 0.049). T2-C: FiLM sub-threshold (RMSE 0.003, MAE 0.004 — disjoint CIs but below bar). **Locked fusion: early-concat shallow MLP** (cat→256→ReLU→dropout→1). |
| T3 | Capacity (depth, width, FiLM retest; conditional embedding fine-tune) | **COMPLETE** (T3-DEC-001). T3-A: 2-layer >> 1-layer (RMSE gap 0.009); 4-layer sub-threshold over 2-layer. T3-B: width 512 selected (monotonic improvement 128→1024, 512 chosen for seed stability over 1024). T3-D: FiLM worse at depth (reversed T2-C signal). T3-C not triggered (no plateau). **Locked architecture: 2-layer ResidualMLP, hidden_dim=512** (cat→512→ReLU→Dropout→ResBlock(512)→1). |
| T4 | Optimization + loss family + target normalization locks | **COMPLETE** (T4-DEC-001, T4-DEC-002). T4-A: MSE retained (Huber 0.5 better on MAE but regresses RMSE). T4-B: raw targets retained (z-score substantially worse, RMSE +0.018). T4-C: 8-epoch constant-LR baseline retained (cosine+32ep is sub-threshold gain; val plateau diagnosed as representation-limited, not training-time-limited). T4-D (multi-seed CIs) deferred until post-T5. |
| T5 | Embedding (questioning frozen ProteomeLM-L assumption) | **COMPLETE Phase 1 + Phase 2** (T5-DEC-001 Phase 1). T5-A: **adapter_1024_proj wins** (RMSE -0.0043, MAE -0.0025, disjoint CIs — first real gain since T3). T5-B: layer 8 confirmed optimal (layers 12/18 much worse). T5-C: ProteomeLM beats raw ESM-C by RMSE 0.006. T5-D (adapter variants: output-dim, width, depth, LayerNorm) ran — **baseline 1024_512 retained** (no variant beat). T5-DEC-002 pending write-up. **Locked architecture: adapter_residual_mlp** (Linear(1152,1024)→ReLU→Dropout→Linear(1024,512), then T3 head). |
| T6 | Chemistry encoding (fingerprints vs multihot) | **COMPLETE under T-regime** (T6-DEC-001 pending write-up). T6-A: 6 chemistry arms (multihot_425, morgan_2048, rdkit_2048, maccs_167, morgan_plus_multihot, maccs_plus_multihot). Under MSE + cross-org split, **multihot_425 retained**; fingerprints did not improve. **To be retested under ranking regime in R1** — theory *permits* (does not predict) a fingerprint gain: removing the gene-mean forces chemistry to carry signal, but if held-out conditions are recombinations of *seen* chemistries (≥95% medium overlap per S1), multihot's exact-match bits may already be sufficient. Prior ≈ 50/50; add a chemistry-space nearest-train-condition diagnostic before R1. |
| **T7-prep** | Diagnostic suite (NOT a promotion gate) | **COMPLETE.** Within-gene Spearman of T5-A: 0.045 (model worse than per-cond-mean baseline 0.056). Within-org cross-condition Spearman: 0.071–0.077 (~60% better than cross-org). Cross-replicate noise floor: median 0.43, mean 0.49. Conclusion: **cross-org transfer is the dominant bottleneck**; ranking regime warranted. Forked to RPLAN.md. |
| **R0** | Data characterization for ranking regime | **COMPLETE** (11 figures + candidates yaml + report; fig 07 dropped, fig 11 added). Surfaced major audit findings: tail_g preferred over IQR for sparse-conditional genes; temperature must be in condition key; H-R0-03 deferred (homology needed). |
| **R-LOCK-1** | Eligibility filter + weighting | **approved** (R-LOCK-1-DEC-001 2026-05-24). `tail_g = p95−p5` (NOT IQR); per-org threshold `max(0.20, 0.80 × (1−r_replicate))`; `weighted_all` train, `hard` filter val. |
| **R-LOCK-2** | Split protocol | **approved** (R-LOCK-2-DEC-001 2026-05-24). Within-org condition holdout, fraction=0.20, `condition_key = (expDesc, media, temperature)`, replicate-grouped, stratified by expGroup; diagnostic splits: `cell_holdout` (easy ceiling) + `stressor_class_holdout` (hard) + cross-org drift. Dev subset: Keio. |
| **R-LOCK-3** | RankingBatch contract + sampler default + run-manifest v2 | **approved** (R-LOCK-3-DEC-001 2026-05-25). `RankingBatch` dataclass + pointwise/pairwise/listwise samplers + collate functions in `src/data/datasets/ranking_batch.py`; run-manifest v2 schema in `data_contract/schemas/run_manifest_v2.schema.json`; 15 unit tests pass. Pointwise is the default for R1/R2. |
| **R-LOCK-4** | Primary metric + ranking baseline + task-relevant noise floor | **approved** (R-LOCK-4-DEC-001 2026-05-25). within-gene Spearman + Kendall co-primary, bootstrap n=1000 over genes, H-RANK-01 = per-condition train mean, primary noise floor = per-gene cross-condition Spearman across replicate pairs; helpers in `src/evaluation/ranking_metrics.py`; 17 unit tests pass; contract at `data_contract/ranking/metric_contract.yaml`. Promotion: Δ-Spearman ≥ 0.01 AND Δ-Kendall ≥ 0.008 AND CIs disjoint AND beats H-RANK-01. |
| **R1** | Representation retest under ranking (chemistry: fingerprints vs multihot) | **COMPLETE — provisional (R1-DEC-001, proposed).** Full sweep 6 arms × 3 seeds × 23 orgs. **no_winner:** multihot_425 best (NDCG@5 0.422), fingerprints uniformly worse → H-R-CHEM-01 rejected. **No arm beats chem-kNN** (NDCG@5 0.485) — the deep model loses to a non-parametric gene-specific lookup. Verdict PROVISIONAL (confounded by MSE+T5-A); multihot carried forward; fingerprints **deferred to R1-revisit** after R2+R-LOSS. chem-kNN proposed as the promotion-gate baseline going forward. |
| **R-LOSS** | Loss family under ranking (6 losses), multihot held constant | **COMPLETE — approved (R-LOSS-DEC-001).** no_winner. Huber > MSE (carried forward); ranking losses (RankNet/LambdaRank/ListMLE/ApproxNDCG) rejected — lambdarank tanks Spearman, approxNDCG worst. **No loss beats chem-kNN** (best 0.435 vs 0.485). Combined with R1: encoder+objective+capacity all fail → bottleneck is LOCAL-vs-GLOBAL. |
| **R-HYBRID** | **NEXT.** Global parametric model + local chem-kNN (residual / retrieval-augmented / learned ensemble), + cold-gene diagnostic. The lever the data points to. | next. |
| **R2 (DEMOTED)** | Fusion topology | demoted — capacity lever, low value (R1 showed linear-MF ≈ deep model). NeuralNDCG also skipped (lambdarank already lost). |
| **R1-revisit** | Re-run chemistry arms under the final architecture | deferred. |
| **R2** | Architecture retest (early-concat / gene-reduce+concat / FiLM / gated) | pending R1. |

## Authoritative data inputs

| Artifact | Path |
|---|---|
| Raw fitness DB | `data/raw/feba.db` |
| Condition workbook | `data/media_composition_v4.xlsx`, sheet `Media_Components_ML` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` |
| Canonical fitness | `data/derived/canonical/v0/fitness_experiment_long.parquet` |
| Canonical experiments | `data/derived/canonical/v0/experiments.parquet` |
| Media master | `data/derived/canonical/v0/media_master.parquet` |
| Media components | `data/derived/canonical/v0/media_components_long.parquet` |

Older workbook versions (v1–v3) are explicitly out of scope.

## Project layout

```
src/                 implementation code (single source of truth)
configs/             Hydra config tree (config.yaml + group dirs)
data_contract/       schemas + frozen handoff artifacts (one per stage/tier)
tests/               unit + integration tests (must stay green)
research_log/        decision ledger entries + tier reports
artifacts/           run outputs (gitignored)
archive/             legacy pre-refactor code (reference only)
```

## Running experiments

Hydra entrypoint:

```bash
python -m src.cli.run_experiment +stage=s0_reproducibility
python -m src.cli.run_experiment +experiment=T1-A_granularity
python -m src.cli.run_experiment +experiment=T1-A_granularity train.seed=0,1,2 -m
```

## Scope of generalization claim (locked, REFACTORPLAN L7)

> "Given a gene and a growth medium **and applied stressor chemistry** drawn from a
> known chemistry vocabulary (including explicit `<UNK>` / `<UNK_STRESSOR>`
> fallbacks), our model predicts conditional gene essentiality — including for
> organisms not seen during training, and conditions structured differently from
> those the gene appeared in during training."

S1 confirmed v4 **medium** chemistry overlap is ≥95% in every candidate protocol at
the Canonical_ID level; S4-DEC-002 adds stressor-derived canonical slots and logs
stressor unknown rates. We do NOT claim "generalizes to any chemistry." Going beyond requires fingerprint
encoders or Canonical_ID-level holdouts (REFACTORPLAN §12, Deferred Experiments).

## Hard rules (clean-room charter)

**Apply to both T-regime and R-regime:**

- **No tier or stage promotion** without pre-declared success criteria + decision-ledger entry.
- **Train-only preprocessing.** Vocab/scalers/eligibility-thresholds fit on train rows only;
  val/test unseen categories → explicit `<UNK>`; log unknown-category rate every run.
- **Denominator parity:** model and any baseline scored on the exact same row set
  (and, in R-regime, the same eligibility-filtered gene set).
- **Legacy results untrusted.** Prior metrics from `archive/` (and across regimes —
  T-regime RMSE/MAE does not transfer to R-regime promotion) may only appear as
  hypotheses to re-test. Nothing auto-promotes.

**T-regime specific:**
- **Co-primary metrics:** RMSE + MAE. Neither may be omitted from a promotion decision.
- **Additive baseline gate (H-BASE-01):** any model that does not beat
  `fit ~ a + α[gene] + β[condition]` on RMSE+MAE is ineligible for promotion.
- **Every run must log:** git SHA, data-contract checksums (feba_db, workbook_v4,
  embeddings, canonical), split protocol id, preprocessing artifact id, config snapshot,
  seed, scored_rowset_hash, metrics, null-baseline deltas, unknown-category rate.
  Validated against `data_contract/schemas/run_manifest_v1.schema.json`.

**R-regime specific** (see RPLAN.md §2 for the Ranking Contract):
- **Co-primary metrics:** within-gene Spearman + within-gene Kendall's τ.
- **Ranking baseline gate (H-RANK-01):** any model that does not beat the
  per-condition-mean baseline (predicts each val condition's train mean for
  every gene) on within-gene Spearman is ineligible for promotion.
- **Noise-floor reporting:** cross-replicate Spearman on the *exact* val rows
  must be plotted alongside model results.
- **Eligibility filter hash** is part of the run manifest (schema v2, locked
  in R-LOCK-3).

## Stage/Tier ownership of decisions (no overlap)

**T-regime:**
- **Feature schema instance** (which encoder wins) → T1 only. NOT S5.
- **All fusion topology decisions** (concat / two-tower / FiLM / gating) → T2 only. NOT T3.
- **Capacity decisions** (depth / width / residuals) → T3 only. NOT T2.
- **Loss family + target normalization** → T4 only. NOT S5.

**R-regime:**
- **Eligibility filter + weighting** → R-LOCK-1 only. NOT R1.
- **Split protocol** → R-LOCK-2 only. NOT R1.
- **Sampler default + RankingBatch contract** → R-LOCK-3 only.
- **Primary metric + ranking baseline + noise-floor reporting** → R-LOCK-4 only.
- **Chemistry encoder under ranking** → R1 only. NOT R-LOCK.
- **Fusion topology under ranking** → R2 only. NOT R1.
- **Loss family under ranking** → R-LOSS (R3+) only.

## Test policy

`pytest tests/` must stay green before any commit.
Stub tests for unimplemented modules use `@pytest.mark.skip` with a
`reason=` pointing at the stage/tier that will activate them.

## Visualization policy (REFACTORPLAN §11)

Every stage/tier with a tier report must include figures that inform its
hard-gate decisions. Figures live at `research_log/figures/<stage_or_tier>/`
as `NN_descriptive_name.png` + sibling `.csv`. Plotting helpers live in
`src/evaluation/reporting.py` — stages call them rather than reinventing
matplotlib code. Required figure lists are inline in each stage's spec
(see e.g. §7 S1 → 24 required + 1 optional).

## Decision log

Decisions go in `research_log/decisions/<stage_or_tier>/` using the template at
`research_log/decisions/decision_template.md`. R-regime decisions use the same
template with the R-regime field substitutions called out in `docs/RPLAN.md §9`
(metrics_primary, baseline gate name, eligibility_filter_hash).

R-LOCK decisions live in `research_log/decisions/r_lock/` (flat directory,
files named `R-LOCK-1-DEC-001.md` etc.).
