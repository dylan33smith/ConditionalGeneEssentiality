# Design decisions log (restart)

Short record of Phase 0 choices. Expand as modeling stages land.

## 2026-04-23 — Media workbook v3

1. **`data/media_composition_v3.xlsx`** (120 media, 1,627 component rows). The `"LB"` entry has been completely replaced with a richer in-silico representation: the old 3-ingredient physical recipe (Tryptone, Yeast Extract, NaCl) was removed and replaced with **74 in-silico metabolite components** (all 20 amino acids, nucleosides, B-vitamins, minerals, cofactors — `Units = "In Silico"`, indicating metabolic availability rather than measured concentration). LB is the largest single medium in the fitness data (5,287,640 rows = 19.5% of all rows).

2. **Component names aligned to workbook conventions.** The original in-silico data used BIGG metabolite ID suffixes (e.g. `'L-Alanine (ala-L)'`). All 74 component names were remapped to match the naming conventions used by other media entries: amino acids → plain free-acid names (`'L-Alanine'`), vitamins → closest existing name (`'Biotin'`, `'Riboflavin'`, `'Folic acid'`, `'Thiamine HCl'`, `'Pyridoxine HCl'`, `'Cyanocobalamin'`, `'Calcium pantothenate'`, `'Lipoic acid'`, `'4-Aminobenzoic acid'`, `'Nicotinic acid'`), simple organics → stripped names (`'Glycerol'`, `'D-Glucose'`, etc.). Result: **32 of 74 LB components now exactly match component names used by at least one other medium**, enabling recognized chemistry overlap in the audit and condition feature vector.

3. **In-silico vs physical representation.** The 42 remaining LB components (ions like `'Sodium'`, `'Magnesium'`; nucleosides like `'Adenosine'`; cofactors like `'Heme'`) have no matching names in other media, which use compound-level reagent names (e.g. `'Sodium Chloride'`, `'MgSO4·7H2O'`). These represent genuine new vocabulary for LB. The mixed representation (in-silico metabolites for LB, physical reagents for all other media) is a known inconsistency to resolve if the in-silico approach is extended to additional media.

4. **Default updated.** `repo_paths.py`, `modeling/train.py`, and `data_analysis/run_phase0.py` now default to v3. v1 (`MEDIA_XLSX`) and v2 (`MEDIA_XLSX_V2`) remain as named constants in `repo_paths.py` for reference; all active code uses `MEDIA_XLSX_V3`.

## 2026-04-05 — Phase 0 (M1) scaffolding

1. **Media workbook “composition” sheet.** The v1 note pointed at sheet index 2; in the current `data/media_composition.xlsx`, **component-level rows** live on **`Media_Components` (index 1)**. Sheet **`Experiments` (index 2)** holds experiment metadata and a **join key** to the DB: `(orgId, name)` ↔ `(orgId, expName)` with `Media` matching `Experiment.media` where both exist. Documented in [`outputs/media_composition_audit.md`](outputs/media_composition_audit.md).

2. **Fitness EDA sample.** Full `GeneFitness` is ~27.4M rows. Plots and sequential variance decomposition use a **systematic subsample** `rowid % 313 = 0` (~88k rows joined to `Experiment`). This is cheap and reproducible but **not** a uniform random sample; tail behaviour should be checked against SQL aggregates if needed.

3. **Exploratory “variance decomposition”.** Reported fractions are **sum-of-squares decompositions on that sample** (marginal `orgId` vs grand mean; then top-400 `expName` buckets on org-centered residuals). This is **not** a fitted mixed model or ANOVA with interaction; it is an order-of-magnitude guide until a proper model is specified (see plan §5.1).

4. **Connected-media / benchmark plot.** [`figures/phase0/06_bipartite_org_media_benchmark.png`](../figures/phase0/06_bipartite_org_media_benchmark.png) uses the **§12.1 v1 heuristic** (LB/RCH2/M9 prefix, `cor12 >= 0.2`, exclude `plant`, exclude Potato Dextrose Broth). **K-organism connectivity** from §12.2 should replace or refine this after review.

5. **Excel coverage.** ~2755 DB experiments have no row on the Excel `Experiments` sheet (see `phase0_summary.json`). Unmapped media handling remains open (plan §1.3.1).

## 2026-04-05 — Canonical tables (M2, `canonical_v0`)

1. **Layout.** Versioned outputs under `data/derived/canonical/v0/` (gitignored with `data/`): `fitness_experiment_long.parquet` (inner join `GeneFitness` ⋈ `Experiment`, 27,410,721 rows), `experiments.parquet`, `media_master.parquet`, `media_components_long.parquet` from the Excel workbook.

2. **Dtypes.** Column kinds follow SQLite `PRAGMA table_info` so every row-group chunk shares one Parquet schema: numeric INT/REAL → `float64`, TEXT → pandas `string[pyarrow]`. Derived columns: `gene_key`, `abs_t`, `has_media_composition` (whether `media` appears in `Media_Components.Media`).

3. **Build.** [`data_processing/build_canonical_v0.py`](../data_processing/build_canonical_v0.py) scans `GeneFitness` by `rowid` ranges (400k rows/step). Committed checksum manifest: [`docs/canonical_build_manifest_v0.json`](../docs/canonical_build_manifest_v0.json).

4. **Weights.** Row-level `cor12` and `abs_t` are present for §2.2-style weighting later; no merged `w_row` column yet (define in modeling or next canonical revision).

## 2026-04-05 — Organism splits (M3)

1. **Source of truth.** Split definitions are derived from **`fitness_experiment_long.parquet`** (canonical_v0) by scanning `orgId` in row batches; expected SHA-256 is copied from [`docs/canonical_build_manifest_v0.json`](../docs/canonical_build_manifest_v0.json).

2. **`organism_single_holdout_largest_v0`.** Val = organism with **most** rows (`Btheta`); test = **second-most** (`DvH`); remaining 46 organisms = train. Purely **data-driven** default for a large val/test slice; swap for a named organism if the science requires it.

3. **`organism_looo_v0`.** One fold per `orgId` (48 folds); each fold JSON lists `val_org_id` and full `train_org_ids`. No test split within a fold.

4. **Regenerate** with `python splits/build_organism_splits.py` after rebuilding canonical Parquet.

## 2026-04-05 — Null baselines (M3.5)

1. **Script:** [`evaluation/compute_null_baselines.py`](../evaluation/compute_null_baselines.py) streams canonical Parquet twice for single-holdout (train stats, then val/test SSE) and twice for LOOO org totals + per-org SSE.

2. **Organism cold start.** For `organism_single_holdout_largest_v0`, val (`Btheta`) and test (`DvH`) never appear in train, so **per-(org,exp)** and **per-org** train means **do not exist** for those rows — implementation **falls back** to **global train mean**; all three RMSEs match (see JSON `n_val_rows_fallback_*`).

3. **LOOO.** Each fold’s baseline is **mean(fit) over all rows except val organism** — the standard null for that fold.

4. **Nearest-neighbour baseline** (plan §7.4) deferred until an evaluation step has embedding lookup.

## 2026-04-05 — Embedding manifest (M4)

1. **Script:** [`embeddings/build_embedding_manifest_m4.py`](../embeddings/build_embedding_manifest_m4.py) — SHA-256 per `data/processed/ProtLM_embeddings_layer8/*_proteomelm.pt`, `torch.load` shape / dim, set overlap vs unique `gene_key` from streaming `fitness_experiment_long.parquet`.

2. **Committed output:** [`docs/embedding_manifest_m4.json`](../docs/embedding_manifest_m4.json) ties bundle to expected canonical SHA-256 from [`docs/canonical_build_manifest_v0.json`](../docs/canonical_build_manifest_v0.json).

3. **Coverage:** 48/48 organisms; canonical genes not in `group_labels` are a **small minority** per org; embeddings include **extra** keys vs the inner-joined long table — **inner-join at train time** (or explicit policy) per plan §17.

## 2026-04-05 — Training harness (M4.5)

1. **Entrypoint:** [`modeling/train.py`](../modeling/train.py) — loads split protocol JSON, scans **train-only** `media` vocabulary (optional row cap for smoke), loads frozen `*_proteomelm.pt` for train∪val orgs, streams `fitness_experiment_long.parquet` with **inner-join** on `gene_key`, **weighted_full** vs **strict_slice** (`--arm`), Huber loss with row weights (arm A) or unit weight (arm B).

2. **Outputs:** `runs/<run_id>/` — `config.json`, `metrics.json` (val RMSE, mean within-gene Spearman + `n_genes_used`, `reference_null_rmse_val_global_train_mean` from M3.5 when present), `model.pt`. **`runs/`** is gitignored.

3. **Null comparison:** M3.5 val RMSE is on **all** val rows with `fit`; the model’s val RMSE is only on rows that pass the embedding join (and strict filters if arm B). `metrics.json` includes a short **note**; subset null or intersection null is future work if needed.

4. **Smoke flags:** `--max-train-rows`, `--max-val-rows`, `--vocab-scan-max-rows`, `--shuffle-buffer` (see train.py docstring).
