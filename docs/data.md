# data.md — registry of datasets, schemas, splits, runs, artifacts

Read before touching data, runs or paths. Every path that exists, and its state.

**State values:** `OK` (present, current) · `MISSING` (referenced but absent) ·
`DEPRECATED` (renamed, retained) · `PLANNED` (not built) · `STALE` (present but
superseded)

> **STANDING BLOCKER (2026-08-25).** Every historical number in `docs/memory.md`
> was produced on the 48-organism `feba.db` (sha `627f2097...`) with
> ProteomeLM-L8 embeddings. **Both are gone.** The current data root is a newer
> 62-organism release with ESM-C embeddings. Nothing in the ledger is
> reproducible from the current tree until the pipeline is re-pinned. See
> `P-FIX-data-repin` in `docs/plan.md`.

---

## Raw and canonical data

| path | what | state |
|---|---|---|
| `data` | machine-local symlink -> `/home/ds85/projects/GeneEssentiality/data`. **Untracked and gitignored** — a checkout can drop it; recreate per worktree. | `OK` |
| `data/raw/feba.db` | Fitness Browser SQLite. Current: 62 orgs / 9,532 experiments / 33.8M rows, sha `b6627137...`. Re-downloaded 2026-07-06. | `OK` |
| `data/raw/aaseqs` | Protein sequences, 279,140 proteins / 62 orgs. Source for ESM-C. | `OK` |
| `data/derived/canonical/v0/fitness_experiment_long.parquet` | Canonical fitness table. Rebuilt 2026-07-06 on the 62-org release: 33,433,466 rows. | `OK` |
| `data/derived/canonical/v0/experiments.parquet` | Experiment metadata. | `OK` |
| `data/derived/canonical/v0/media_master.parquet` | 191 media. | `OK` |
| `data/derived/canonical/v0/media_components_long.parquet` | 4,078 component rows. | `OK` |
| `data/canonical_build_manifest_v0_62org.json` | Build manifest + SHA256 for the rebuild. | `OK` |
| `data/excluded_undefined_media.json` | 94 experiments excluded for wholly-undefined media (Potato tuber/stem, Potato Dextrose Broth, Clovis xylem sap). | `OK` |
| `data/media_composition_v5.xlsx` | Curated media chemistry, 191 media. Supersedes v4. Provenance audited — recipes trace to primary literature, not generated. See `archive_docs/v4_workbook_audit.md`. | `OK` |
| `data/media_composition_v4.xlsx` | Prior curation, 121 media. | `STALE` |
| the pinned 48-organism `feba.db` (sha `627f2097...`) | The release every historical result used. | `MISSING` |

## Embeddings

| path | what | state |
|---|---|---|
| `data/processed/ESMC_embeddings/<org>_esmc.pt` | Mean-pooled ESM-C **300M**, 960-d, fp16, 62 orgs. Input to the cross-organism shared-embedding proposal. **Not usable as ProteomeLM input — wrong width.** Anisotropic; centre/whiten before any similarity use. The directory name records no variant, which is how it was mistaken for the 600M set — see `docs/bugs.md`. | `OK` |
| `data/processed/ESMC_embeddings_600m/<org>_esmc.pt` | Mean-pooled ESM-C **600M**, 1152-d, 62 orgs, 279,140 proteins. Regenerated 2026-08-25. The input ProteomeLM-L requires. | `OK` |
| `data/processed/ProtLM_embeddings_layer8/*.pt` | ProteomeLM layer-8, 1152-d. The original artifact behind every historical number. | `MISSING` |
| `data/processed/PLM_embeddings_layer8/<org>_proteomelm.pt` | ProteomeLM-L layer-8, 1152-d, **regenerated 2026-08-25** from ESM-C 600M + the frozen `Bitbol-Lab/ProteomeLM-L` checkpoint. Functionally the replacement for the lost directory above. NOT guaranteed bit-identical: ProteomeLM is proteome-contextual and this runs on the 62-organism release. | `OK` |
| `/data/ds85/huggingface_cache/models--Bitbol-Lab--ProteomeLM-L` | The frozen ProteomeLM-L checkpoint (snapshot `0f834036...`). Survived the data-root incident; this is what makes the embeddings regenerable. | `OK` |

## Contracts and policies

| path | what | state |
|---|---|---|
| `data_contract/ranking/metric_contract.yaml` | The R-LOCK-4 metric contract: metrics, bootstrap scheme, per-split baselines, promotion rule. Amended to **v3** on 2026-08-25: ceiling keys renamed and disambiguated, promotion delta restated on `ndcg_at_5`, and a claim about the training objective that R-LOSS/R-TOPK refuted retracted in place. No code reads this file; it is the human-facing spec that `src/ranking/eval/` implements. | `OK` |
| `data_contract/ranking/eligibility_policy.yaml` | R-LOCK-1 eligibility thresholds and `w_g` policy. | `OK` |
| `data_contract/ranking/split_protocol.yaml` | R-LOCK-2 split protocol. | `OK` |
| `data_contract/ranking/reval_baseline.json` | The R-EVAL regression gate: pinned model / chem_knn / linear_mf / chem_null values for the `fast` and `full` tags. **Pinned on the lost 48-org release.** | `STALE` |
| `data_contract/ranking/r0_candidates.yaml` | R0 organism candidate list. | `OK` |
| `data_contract/preprocessing/de21504134c84a6c/` | Frozen S4 feature contract: 425-d multihot vocab, scalers, metadata. | `OK` |
| `data_contract/chemistry/canonical_fingerprints.npz` | Morgan / RDKit / MACCS fingerprints for canonical compounds. | `OK` |
| `data_contract/chemistry/canonical_id_smiles.json` | SMILES for ~77% of canonical IDs. | `OK` |
| `data_contract/chemistry/experiment_fingerprints.npz` | Per-experiment fingerprints. | `OK` |
| `data_contract/schemas/*.schema.json` | Canonical table, condition-feature and run-manifest (v1, v2) schemas. | `OK` |
| `data_contract/splits/*.yaml` | Locked, candidate and diagnostic split protocols. | `OK` |
| `data_contract/policy/quality_policy.yaml`, `data_contract/policy/eval_policy.yaml` | S5 quality policy, eval policy. | `OK` |
| `data_contract/preprocessing/vocab_policy.md`, `data_contract/preprocessing/scaler_policy.md` | Train-only preprocessing policies. | `OK` |
| `data_contract/manifests/canonical_build_manifest_v0.json` | Canonical-table build manifest. | `OK` |
| `data_contract/manifests/condition_encoding_manifest_v0.json` | Condition-encoding build manifest. | `OK` |
| `data_contract/manifests/data_inputs_manifest_M0.json` | M0 data-inputs manifest. Note the milestone suffix is upper-case here and lower-case in the two below — an inconsistency retained because these are frozen artifacts. | `OK` |
| `data_contract/manifests/embedding_manifest_m4.json` | m4 embedding manifest. | `OK` |
| `data_contract/manifests/quality_tier_manifest_v0.json` | Quality-tier build manifest. | `OK` |
| `data_contract/manifests/splits_build_manifest_m3.json` | m3 splits build manifest. | `OK` |

## Splits

| split | materializer | state |
|---|---|---|
| `condition_holdout` | `src/data/datasets/build_ranking_split.py:materialize_condition_holdout` | `OK` |
| `cold_gene` | `src/data/datasets/build_ranking_split.py:materialize_cold_gene` | `OK` |
| `leave_compound_out` | `src/data/datasets/build_ranking_split.py:materialize_leave_compound_out` (+ `assert_no_compound_leakage`); pipeline hook `prepare_leave_compound_out_data`; config `configs/experiment/P-EVL_leave_compound_out.yaml` | `OK` |

## Run directories

Layout: `artifacts/runs/<YYYYMMDD>_<HHMMSS>_<EXPERIMENT>_s<split_seed>/`.
Named aggregate dirs (`rcold/`, `reval/`, `t1a/`, ...) hold the tidy per-experiment
CSVs the runner emits.

| path | what | state |
|---|---|---|
| `artifacts/runs/` | Timestamped per-invocation run dirs, 27 of them, spanning `S0`-`S5`, `T1-A`/`T1-B`, `R-EVAL`, `R-COLD`. | `OK` |
| `artifacts/runs/rcold/` | R-COLD aggregate outputs. | `OK` |
| `artifacts/runs/reval/` | R-EVAL aggregate outputs. | `OK` |
| `artifacts/runs/t1a/`, `t1a2/`, `t1b/`, `t1b3/` | T1 tier aggregates. | `OK` |
| `artifacts/runs/DEPRECATED_rhybrid_alpha_curve.csv` | R-HYBRID-A alpha sweep. Was a loose file at the results root with no owning run dir, so unattributable. Renamed 2026-08-25, retained. | `DEPRECATED` |
| `artifacts/baselines/` | Cached baseline predictions. | `OK` |
| `artifacts/cache/` | Pipeline cache (~3.8M). Regenerable. | `OK` |
| `artifacts/indexes/` | Gene/condition indexes. | `OK` |
| `artifacts/deck/` | Presentation exports. | `OK` |
| `artifacts/checksums_cache.json` | Input checksum cache. | `OK` |
| `artifacts/artifacts` | Self-referential symlink -> `artifacts/`. Caused infinite recursion in any recursive walk. **Removed 2026-08-25** — see `docs/bugs.md`. | `MISSING` (deliberately) |

## New evaluation modules (2026-08-25)

| path | what | state |
|---|---|---|
| `src/ranking/eval/extra_baselines.py` | `gbdt_predict` (tree baseline) and `resmem_predict` (learned-local). The two model families the project had never tested. | `OK` |
| `src/ranking/targets.py` | `fit_additive_effects`, `apply_demeaning`, `replicate_average_target` — de-meaned and denoised targets. | `OK` |
| `configs/experiment/P-EVL_external_orgs.yaml` | External replication on the 39 organisms that informed no design decision. | `OK` |
| `configs/experiment/P-EVL_leave_compound_out.yaml` | The honest-generalization split. | `OK` |

## Figures

`research_log/figures/<experiment>/` — 26 directories: `r0_data`, `r_conf`,
`stage1`, `stage2`, `stage5`, `t7_prep`, `tier1_a`, `tier1_a2`, `tier1_b`,
`tier1_b3`, `tier1_d`, `tier1_e`, `tier2_a`, `tier2_b`, `tier2_c`, `tier3_a`,
`tier3_b`, `tier3_d`, `tier4_a`, `tier4_b`, `tier4_c`, `tier5_a`, `tier5_b`,
`tier5_c`, `tier5_d`, `tier6_a`. All `OK`.

## Other registered paths

| path | what | state |
|---|---|---|
| `research_log/notes/unique_stressor_strings_from_source_experiment_conditions.txt` | Raw stressor strings from source conditions. Data, not documentation. | `OK` |
| `archive_docs/` | Superseded documentation, moved unchanged 2026-08-25. Read-only; never edited, never a citation target for current claims. | `OK` |
| `paper/SCIENTIFIC_SYNTHESIS.md` | The scientific narrative. Moved out of `research_log/` 2026-08-25; it is manuscript material, not a working doc. | `OK` |
| `paper/PAPER.md` | **The central paper document**: hypothesis, framing, introduction, the variance decomposition, the three-scenario structure, claim boundaries, the gold-standard design, and the algorithm shortlist. Everything we write draws from here. | `OK` |
| `paper/PAPER_OUTLINE_2026-08-25.md` | Paper outline against the Bernett et al. negative-result genre. | `OK` |
| `dashboard/` | Static dashboard generator + content model (`dashboard/content/experiments.json`, `dashboard/content/learnings.json`). Content is **accurate** — it carries the cold-gene CI caveat correctly. The risk is structural, not factual: hand-maintained duplication of ledger content that nothing verifies, so it can drift silently. See `P-FIX-dashboard-view` in `docs/plan.md`. | `OK` |
| `DEPRECATED_testing.ipynb` | Scratch notebook, was loose at repo root. | `DEPRECATED` |
| `DEPRECATED_s5_additive_gate.parquet` | S5 output, was loose at repo root. | `DEPRECATED` |
| `DEPRECATED_s5_metrics.parquet` | S5 output, was loose at repo root. | `DEPRECATED` |
| `DEPRECATED_s5_summary.parquet` | S5 output, was loose at repo root. | `DEPRECATED` |
| `.claude/worktrees/` | Two stale git worktrees holding duplicate doc trees. Unique content rescued to `archive_docs/v4_workbook_audit.md`; worktrees removed 2026-08-25. | `MISSING` (deliberately) |

## The full-panel headline command

The 23 replicate organisms are the headline eval subset. `orgs=null` means ALL
organisms and gives different, lower numbers — it is not the headline.

```bash
python -m src.cli.run_experiment +experiment=R-EVAL_regression experiment.tag=full \
    experiment.model_seeds=[0,1,2] \
    'experiment.orgs=[ANA3,BFirm,Btheta,Burk376,Caulo,Cola,Cup4G11,Dda3937,Ddia6719,DdiaME23,Dino,DvH,Dyella79,HerbieS,Kang,Keio,Korea,Koxy,MR1,Marino,Methanococcus_JJ,Methanococcus_S2,Miya]'
```
