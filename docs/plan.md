# plan.md — the board

**Last updated:** 2026-08-25

Read at session start. This file exists so a new session never has to grep
`docs/memory.md` to know where things stand.

---

## Current state

- **Branch:** `ranking` is the trunk. Develop here.
- **Phase:** `P` — the paper / honest-benchmark push. The warm "beat chem-kNN"
  modelling programme is closed; see the 2026-07-06 reframe in `docs/memory.md`.
- **Scientific position:** the within-organism condition-ranking benchmark is
  memorization-dominated. A lookup handed each gene's own history beats every global
  model tested across encoder, objective, capacity, hybridization and training-organism
  volume. Removing that history (`cold_gene`) makes the learned model the best
  available predictor — so the warm negative is a memorization gap, not an
  embedding-value gap. The live science is whether embedding + chemistry contain
  generalizable `I(g,c)` signal on honest splits.
- **Data footing RESTORED (2026-08-25).** ProteomeLM-L8 was regenerated for all 62
  organisms from ESM-C 600M plus the frozen `Bitbol-Lab/ProteomeLM-L` checkpoint, both of
  which survived the 2026-07-06 incident. The pipeline runs again. **Caveat:** ProteomeLM
  is proteome-contextual and this is the 62-organism release, so regenerated embeddings are
  not guaranteed identical to the lost 48-organism ones. The numbers in the table below are
  the OLD pinned values; they are being re-measured and any movement must be attributed,
  not absorbed.
- **Paper items A-H implemented** (see the 2026-08-25 entries in `docs/memory.md`): paired
  bootstrap, parity ceiling, leave-compound-out split, similarity-stratified diagnostic,
  GBDT + ResMem baselines, de-meaned/denoised targets, a 39-organism external panel, and a
  verified related-work section. All are code-complete and unit-tested; most still need to
  be RUN.

## Headline result

Metrics are rows; methods are columns. Floor and ceiling are their own columns.

| metric | chem_null (floor) | model | linear_mf | chem_knn (reference) | replicate ceiling |
|---|---|---|---|---|---|
| `ndcg_at_5` | 0.3434 | 0.4319 | 0.4290 | **0.4852** | 0.658 † |
| `within_gene_spearman_mean` | 0.0235 | 0.1522 | 0.1426 | **0.2402** | 0.393 † |
| `within_gene_kendall_mean` | 0.0175 | 0.1103 | 0.1031 | **0.1783** | n/a — not computed for the ceiling |
| `precision_at_5` | 0.3448 | 0.4014 | 0.3940 | **0.4342** | 0.604 † |
| `coverage` | 1.000 | 1.000 | 1.000 | 1.000 | n/a — the ceiling is not a predictor |
| `n_genes` | 39,356 | 39,356 | 39,356 | 39,356 | 45,943 † |

**Provenance.** Methods: `data_contract/ranking/reval_baseline.json`, tag `full`,
`condition_holdout` split, split_seed 0, model_seeds [0,1,2], 8 epochs, locked loss
`pointwise_huber`, 23 replicate organisms, MEAN over the parity-eligible val gene set
(n=39,356). † Ceiling: the 2026-05-25 measurement in
`data_contract/ranking/metric_contract.yaml`, n=45,943 — **a different gene set**; see
the open question below.

**COLUMNS**
- `chem_null` — the population condition profile: predict from the train-gene mean at the
  chemically nearest train condition. Gene-identity-free, so it captures the condition
  main effect and nothing gene-specific. The floor.
- `model` — AdapterResidualMLP over the frozen embedding concatenated with chemistry.
  The intended system; a global parametric model.
- `linear_mf` — bilinear inductive MF with a free per-gene latent. Differs from `model`
  by being ~13k params and linear, and by learning its gene representation from fitness
  rather than sequence. The capacity-and-representation control.
- `chem_knn` — reads the target gene's OWN measured fit at chemically near train
  conditions. Differs from every other column by having access to the gene's answer key.
  The memorization reference.
- `replicate ceiling` — replicate A's fit used to predict replicate B's. Not a method;
  the best any predictor could score on this noisy target.

**ROWS**
- `ndcg_at_5` — top-5 agreement with the true stressors; higher is better. PRIMARY. Read
  each method against `chem_knn` (must be beaten by the promotion delta) and against the
  ceiling (how much achievable signal is captured).
- `within_gene_spearman_mean` — full-list rank agreement; higher is better. SECONDARY,
  completeness. Note it separates the methods far more sharply than `ndcg_at_5` does.
- `within_gene_kendall_mean` — as above, Kendall tau; higher is better. Carried as the
  no-regression guard, not gated on.
- `precision_at_5` — fraction of the predicted top-5 that are true stressors; higher is
  better. DIAGNOSTIC. Note how compressed the range is: even the floor scores 0.34.
- `coverage` — fraction of eligible val genes the method can score; higher is better.
  All 1.000 here by construction (denominator parity). It is the row that matters on
  `cold_gene`, where `chem_knn` drops to 0.0000.
- `n_genes` — the denominator. Identical across methods is the parity guarantee; the
  ceiling's differing value is the defect flagged below.

**SYNTHESIS.** Every learned global model lands between the floor and the lookup, and
the ordering is stable across all four metrics: `chem_null` < `linear_mf` ~ `model` <
`chem_knn` < ceiling. That `linear_mf` matches `model` at 1/200th the parameters, while
learning its gene representation from fitness rather than sequence, rules out both
capacity and the embedding's fitness-blindness as the warm bottleneck. The gap the model
must close to reach the lookup (0.053 `ndcg_at_5`) is small next to the gap from the
lookup to the ceiling (0.173), and the latter is an unquantified mix of irreducible
measurement noise and structure the fixed lookup misses. On `cold_gene` the picture
inverts: `chem_knn` coverage goes to 0.0000 and `model` beats `chem_null` — see the
2026-06-23 entry in `docs/memory.md`.

## Work in progress

- **Re-pinning the regression gate** on the regenerated embeddings. Until that lands, the
  headline table below is the previous release's values.
- The new diagnostics (leave-compound-out, similarity-stratified, external organisms,
  paired CI) are built and configured but not yet run end-to-end.

## Next up

| id | what | why now | blocks |
|---|---|---|---|
| `P-FIX-data-repin` | Pin one archived data release; regenerate or restore embeddings; re-pin `reval_baseline.json`; recompute every quoted number. | Every table in the project is currently unreproducible. Nothing else should start first. | everything |
| `P-EVL-ceiling-parity` | Compute the replicate ceiling on the parity-eligible gene set and add it to `reval_baseline.json`. | The ceiling has never been measured on the same gene set as the methods it is quoted against. Cheap — the harness computes it already, it just needs the `eligibility` filter applied. | the paper's Figure 2 and every "fraction of ceiling" claim |
| `P-ANL-cold-gene-paired-ci` | Paired/pooled bootstrap on the per-gene `ndcg_at_5` delta (model - chem_null) on `cold_gene`. | The one positive result is not CI-significant on the primary metric. Determines whether the paper ends on a positive or a diagnostic note. | the paper's payoff |
| `P-EVL-leave-compound-out` | Build the leave-compound-out materializer; re-run model and baselines under it. | The honest-benchmark arm. No materializer exists. | the paper's Result 5 |
| `P-ANL-similarity-stratified` | Bucket held-out cells by the gene's chemical distance to its nearest in-train condition; show lookup dominance decaying with distance. | Cheap, no training. Turns the "the split is rigged" argument into a measurement. | the paper's most persuasive figure |
| `P-TRN-learned-local` | Add learned non-parametric baselines (EASE / learned-metric kNN / TabR / ResMem) and a gradient-boosted-trees arm. | Every model beaten so far was global/compressing, and no tree baseline was ever run. Closes the obvious review. | the paper's Result 1 |
| `P-ANL-table-generation` | Generate every manuscript table from `reval_baseline.json` by script rather than by hand. | The paper's draft Table 1 had been assembled from three sources and inherited three different denominators — the exact parity violation the paper argues against. | manuscript tables |
| `P-EVL-random-baseline` | Add `random` and `constant` predictors as permanent rows in the standard report, computed on the exact eligible val gene set. | Chance on `ndcg_at_5` is ~0.16, not 0. Every reported value should be read against it, and a paper about honest baselines cannot omit the most basic one. | every table |
| `P-ANL-interaction-concentration` | Measure whether interaction variance is concentrated in a minority of cells (Gini / top-k share of `I(g,c)`), per gene and overall. | The decomposition says interaction is 47.9% of variance while the ranking ceiling is modest. Concentration is the likely reconciliation, and if true it reframes the task as sparse-hit detection rather than dense prediction. | the paper's framing |
| `P-TRN-main-effect-vs-interaction` | Train the SAME model on two targets: `a_g` (non-conditional essentiality) and `I(g,c)` (the interaction), same features, same protocol. | The centrepiece contrast: shows quantitatively that the non-conditional problem is easy and the conditional one is not, with everything else held fixed. | the paper's Result 1 |
| `P-DAT-homology-controlled-genes` | Cluster genes by sequence identity (MMseqs2/CD-HIT or `feba.db`'s `Ortholog` table) and hold out whole clusters in `cold_gene`; report stratified by max identity to any train gene. | `cold_gene` currently holds out RANDOM genes, so a held-out gene's paralog can sit in train and the embedding transfers trivially. This is exactly the sequence-similarity leakage Bernett et al. removed, and our splits do not control for it. | the benchmark's credibility |
| `P-DAT-scaffold-controlled-compounds` | Cluster compounds by Morgan-fingerprint Tanimoto or Murcko scaffold; hold out clusters. The materializer already accepts a compound-to-group mapping for exactly this. | Exact-compound holdout still leaves near-identical analogues in train (two tetracyclines). | `P-EVL-leave-compound-out-full` |
| `P-ANL-profile-similarity-strat` | For each held-out condition compute its max genome-wide fitness-profile correlation to any TRAIN condition; stratify performance by it. | Chemically unrelated compounds can be phenotypic twins (same pathway), which no structural filter catches. Stratifying rather than splitting keeps it leakage-free. | the strongest similarity control |
| `P-EVL-cold-organism` | Add a cold-organism split under the R harness. | Cross-organism transfer was only ever measured in the T regime under RMSE, never under the locked ranking harness. The benchmark should ship all four generalization axes. | the benchmark release |
| `P-EVL-leave-compound-out-full` | Run leave-compound-out on the full 23-organism panel, 3 seeds. | The 3-organism pilot shows the lookup's margin collapsing 59%. This is the paper's central experiment and currently rests on one seed and three organisms. | the paper's Result 5 |
| `P-ANL-lco-null-rise` | Diagnose why `chem_null` rises from 0.3426 to 0.4284 under leave-compound-out. | If the held-out compounds' conditions are simply easier, the margin collapse is partly an artifact of split difficulty rather than near-neighbour removal. **Blocks publication of the LCO result.** | `P-EVL-random-baseline` | Add `random` and `constant` predictors as permanent rows in the standard report, computed on the exact eligible val gene set. | Chance on `ndcg_at_5` is ~0.16, not 0. Every reported value should be read against it, and a paper about honest baselines cannot omit the most basic one. | every table |
| `P-ANL-interaction-concentration` | Measure whether interaction variance is concentrated in a minority of cells (Gini / top-k share of `I(g,c)`), per gene and overall. | The decomposition says interaction is 47.9% of variance while the ranking ceiling is modest. Concentration is the likely reconciliation, and if true it reframes the task as sparse-hit detection rather than dense prediction. | the paper's framing |
| `P-TRN-main-effect-vs-interaction` | Train the SAME model on two targets: `a_g` (non-conditional essentiality) and `I(g,c)` (the interaction), same features, same protocol. | The centrepiece contrast: shows quantitatively that the non-conditional problem is easy and the conditional one is not, with everything else held fixed. | the paper's Result 1 |
| `P-DAT-homology-controlled-genes` | Cluster genes by sequence identity (MMseqs2/CD-HIT or `feba.db`'s `Ortholog` table) and hold out whole clusters in `cold_gene`; report stratified by max identity to any train gene. | `cold_gene` currently holds out RANDOM genes, so a held-out gene's paralog can sit in train and the embedding transfers trivially. This is exactly the sequence-similarity leakage Bernett et al. removed, and our splits do not control for it. | the benchmark's credibility |
| `P-DAT-scaffold-controlled-compounds` | Cluster compounds by Morgan-fingerprint Tanimoto or Murcko scaffold; hold out clusters. The materializer already accepts a compound-to-group mapping for exactly this. | Exact-compound holdout still leaves near-identical analogues in train (two tetracyclines). | `P-EVL-leave-compound-out-full` |
| `P-ANL-profile-similarity-strat` | For each held-out condition compute its max genome-wide fitness-profile correlation to any TRAIN condition; stratify performance by it. | Chemically unrelated compounds can be phenotypic twins (same pathway), which no structural filter catches. Stratifying rather than splitting keeps it leakage-free. | the strongest similarity control |
| `P-EVL-cold-organism` | Add a cold-organism split under the R harness. | Cross-organism transfer was only ever measured in the T regime under RMSE, never under the locked ranking harness. The benchmark should ship all four generalization axes. | the benchmark release |
| `P-EVL-leave-compound-out-full` |
| `P-ANL-eligibility-sensitivity` | Re-run the headline at several `tail_g` thresholds. | The eligibility rule decides which genes are scored, what the metric measures, and how high the ceiling sits — and has never been varied. If the ordering changes, the headline is fragile; if not, it is a strong robustness figure. We do not currently know which. | the headline's robustness |
| `P-EVL-split-seed-variance` | Re-run at split seeds 1 and 2. | `model_seeds` varies only network init; `split_seed` has been 0 throughout. Every CI and every "disjoint across 3 seeds" claim describes init noise on ONE partition. | every CI in the paper |
| `P-EVL-knn-sensitivity` | Sweep chem-kNN's `k` and report the envelope. | The gate was fixed at k=5 and never tuned. Citing Dacrema et al. on under-tuned baselines while shipping an untuned baseline is an open invitation. | the gate's credibility |
| `P-ANL-relevance-sensitivity` | Vary the NDCG gain (`max(0,-fit)` vs t-thresholded binary vs rank-based). | The relevance function defines what "top stressor" means and was never varied. | the task definition |
| `P-FIX-pooled-ci` | Replace the mean-of-per-seed-CIs band with a pooled bootstrap. | A summary band is currently presented where a confidence interval is claimed. | every reported CI |
| `P-ANL-premetric-finding-audit` | Re-derive every durable finding stated before 2026-06-24 under `ndcg_at_5`. | Two findings have now failed this check (the cold-gene CI read, and the population-signal claim at 79.5% of the model on NDCG). Both were Spearman-era conclusions never re-checked when the primary metric changed. | the abstract |
| `P-FIX-dashboard-view` | Make `dashboard/` a generated view of `docs/`, or retire it. | Its content is currently accurate, but it is hand-maintained duplication that no verifier covers, so it can drift silently. Prevention, not repair. | nothing |

## Open questions

- **The ceiling is not parity-matched.** `reval_baseline.json` contains no ceiling entry;
  every published "fraction of ceiling" figure mixes n=45,943 (ceiling) with n=39,356
  (methods). Tracked as `P-EVL-ceiling-parity`.
- **The ceiling is a function of the target, not a constant.** It is the ceiling of the
  single-noisy-measurement task; replicate-averaging or `t`-shrinking the target raises
  it. Any "fraction of achievable" claim must say which target it means.
- **Is the eligibility filter defensible externally?** It selects high-spread genes, which
  replicate better, so it raises the ceiling it is measured against. Correct for internal
  parity; a reviewer will press on whether the resulting numbers describe the assay.
- **`R-AUG`'s negative transfer was measured warm-only.** More diverse training organisms
  may help the inductive cold-gene regime even though they hurt the warm headline.

## Algorithm notes — to consider, not yet queued

Recorded so they are not lost. All are premised on `P-ANL-interaction-concentration`
confirming that interaction signal is sparse and concentrated; if it is dense, most of this
list is the wrong response.

- **Two-stage / hurdle models.** Classify "is this cell a hit?", then regress magnitude among
  hits. Directly matches a sparse-strong-interaction structure. Nothing tried so far does
  this — every model assumed a dense signal.
- **Imbalance-aware ranking objectives.** If hits are a few percent of cells, pointwise Huber
  over all cells is the wrong loss. This may be the real reason the ranking losses
  underperformed, rather than "ranking losses do not help" — worth revisiting that verdict.
- **Explicit interaction models on the de-meaned target.** Factorization machines, or a
  bilinear head trained on `I(g,c)` directly, so no capacity is spent re-learning `a_g`. The
  de-meaning machinery now exists in `src/ranking/targets.py`.
- **EASE and TabR.** The learned-local family cited in the literature review but never run.
  Cheap, and would close the "you only tried global models" objection completely. Note the
  interpretive trap: a learned-local winner CONFIRMS the memorization finding.
- **Chemistry-aware condition kernels.** Tanimoto-kernel ridge regression or a Gaussian
  process — an interpretable middle ground between a fixed kNN and an opaque MLP, and a
  natural fit for a benchmark paper.
- **Deprioritized: fine-tuning the encoder.** Not until `P-TRN-main-effect-vs-interaction`
  says whether sequence carries interaction information at all. High cost, and the prior
  from linear-MF is discouraging — though weaker than previously stated, since that latent
  is transductive.

## Ledger

Newest first. One row per completed unit of work. Full record in `docs/memory.md`
under the same date.

| date | id | verdict | one-line result |
|---|---|---|---|
| 2026-08-25 | `P-ANL-framing-locked` | done | Foundational benchmark framing adopted; `paper/PAPER.md` created as the central document. Similarity control (homology / scaffold / fitness-profile) added to the gold-standard design after the lead identified it as a gap. |
| 2026-08-25 | `P-ANL-variance-decomposition` | **KEY** | `a_g` 40.6% / `b_c` 1.7% / `I(g,c)` **47.9%** / noise 9.8%. The interaction is the largest component and 83% of the post-main-effect residual is real signal — inverting the "signal is weak and noisy" narrative. |
| 2026-08-25 | `P-EVL-random-baseline` | done (pilot) | Chance on `ndcg_at_5` is ~0.161; `chem_null` sits well above it, so existing comparisons are sound, but raw values overstate progress. Needs computing on the exact reported gene set. |
| 2026-08-25 | `P-EVL-leave-compound-out` | **POSITIVE (pilot)** | Holding out whole stressor compounds collapses the lookup's margin over the model by 59% (0.0483 -> 0.0199) — the first empirical evidence for the rigged-benchmark argument. Pilot only: 3 orgs, 1 seed, and an unexplained `chem_null` rise that must be diagnosed first. |
| 2026-08-25 | `P-FIX-data-repin` | **RESOLVED** | ProteomeLM-L8 regenerated from ESM-C 600M + the frozen checkpoint; pipeline runs again. Gate failed as designed and the movement was fully attributed: only `model` moved (+0.0140 NDCG@5), every embedding-independent quantity bit-exact. The regenerated embeddings are BETTER. Fast gate re-pinned; full 23-org re-pin queued. |
| 2026-08-25 | `P-ANL-defensibility-audit` | done | 13 claims audited. One durable finding failed: "population structure carries ~0 within-gene signal" is Spearman-only and false on NDCG@5 (chem_null is 79.5% of the model). Six new vulnerabilities added to the board. |
| 2026-08-25 | `P-ANL-items-A-to-H` | done | Paired bootstrap, parity ceiling, leave-compound-out split, similarity-stratified diagnostic, GBDT + ResMem, de-meaned/denoised targets, 39-org external panel, verified related work. 24 new unit tests. |
| 2026-08-25 | `P-FIX-metric-contract` | done | `metric_contract.yaml` amended to v3: ceiling keys disambiguated (median/unfiltered vs parity-eligible mean vs high-rep subset), promotion delta restated on `ndcg_at_5`, refuted objective claim retracted in place. |
| 2026-08-25 | `P-FIX-docs-system` | done | Six-file docs system adopted; 5 live contradictions found and corrected; self-referential `artifacts/artifacts` symlink removed; 2 stale worktrees removed after rescuing a unique 266-line audit. |
| 2026-08-25 | `P-ANL-paper-outline` | done | Bernett et al. fits as genre, not structure; the replicate ceiling is the distinguishing asset; leave-compound-out is the missing experiment. |
| 2026-07-06 | `P-DAT-canonical-rebuild` | done | Canonical parquet rebuilt on the 62-org release: 33,433,466 rows, 9,438 experiments. |
| 2026-07-06 | `P-DAT-chemistry-esmc` | done | Media chemistry filled for 69 new media; ESM-C 300M generated for 279,140 proteins; Phase-0 ortholog-clustering gate PASSED (AUROC 0.97-0.998). |
| 2026-07-06 | `P-ANL-reframe` | done | The warm benchmark is rigged three ways; H1 retired, H2 adopted; `chem_knn` demoted from gate to memorization reference. |
| 2026-07-06 | `P-FIX-data-root-loss` | recovered | Data root vanished; raw DB re-downloaded as a NEWER 62-org release; ProteomeLM not recoverable. |
| 2026-06-29 | `R-ANL-direction-slate` | done | Five-direction slate; ortholog conditional-response divergence is the one clean escape from the lookup wall. |
| 2026-06-25 | `R-ANL-dark-residual` | negative | Chemically-unexpected essentiality is real (~10x null) but orphan genes are LESS surprising than annotated ones — the dark-genome premise is falsified. |
| 2026-06-25 | `R-ANL-literature` | done | The wall is a named cross-field regime (kNN-LM, Feldman, tabular-DL-vs-trees, recsys); `chem_knn` winning is expected, not a leak. |
| 2026-06-24 | `R-LCK-ndcg-primary` | done | `ndcg_at_5` made primary project-wide; NDCG CI added; corrected a Spearman-only significance read on the cold-gene result. |
| 2026-06-23 | `R-EVL-cold-gene` | **POSITIVE** | `chem_knn` coverage 0.0000; model beats `chem_null` 0.2748 vs 0.2447 on unseen genes. The embedding generalizes; the warm negative is a memorization gap. |
| 2026-06-18 | `R-TRN-org-augmentation` | negative | Training on 48 organisms HURTS the 23-organism headline (-0.0152); negative transfer, every seed disjoint. |
| 2026-06-17 | `R-TRN-topk-loss` | negative | Top-5-truncated NDCG losses fall below their untruncated forms; the objective axis is closed. |
| 2026-06-17 | `R-FIX-ranking-batch` | done | `RankingBatch` samplers wired into training; gate re-pinned; movement confirmed as RNG variance. |
| 2026-06-15 | `R-FIX-cleanup` | done | 130 files / -13.4k lines pruned; modular `src/ranking/` core and shared runner; every step verified bit-exact. |
| 2026-06-06 | `R-TRN-hybrid` | negative | Three learned fusions converge on ~0; the static ensemble's +0.008 is below the promotion delta. The lookup ceiling is real, not a fusion-design artifact. |
| 2026-06-06 | `R-TRN-loss-family` | negative | No ranking loss beats the gate; `pointwise_huber` stays best. |
| 2026-06-06 | `R-ANL-confidence-strat` | negative | `chem_knn` wins in every confidence quartile; label noise depresses everything but never closes the gap. |
| 2026-06-06 | `R-TRN-chemistry-encoder` | negative | No fingerprint beats 425-d multihot; `linear_mf` ~ deep model, so neither capacity nor fitness-blindness is the bottleneck. |
| 2026-05-25 | `R-LCK-instrument` | done | Split, eligibility, parity, metrics, bootstrap, baselines and ceiling locked; 48 orgs / 182,447 gene records / ~4,200 conditions. |
| 2026-05 | `R-ANL-reframe-within-org` | done | Narrowed from cross-organism to within-organism; metric changed from RMSE to per-gene ranking. |
| 2026-04/05 | `T-TRN-cross-org` | negative | Cross-organism within-gene ranking ~0.045 Spearman, indistinguishable from random. RMSE is gene-mean-dominated. |
| 2026-04 | `S-DAT-foundation` | done | S0-S5: characterization, baselines, split lock, frozen 425-d feature contract, quality policy. |
