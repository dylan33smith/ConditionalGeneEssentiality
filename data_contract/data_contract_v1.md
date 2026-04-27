# Data Contract v1

**Status:** pending — fill in after Stage 0 governance lock.

## Authoritative sources

| Artifact | Path | Note |
|---|---|---|
| Raw fitness DB | `data/raw/feba.db` | Never modify |
| Condition workbook | `data/media_composition_v4.xlsx` | Sheet `Media_Components_ML` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` | Frozen |
| Canonical fitness table | `data/derived/canonical/v0/fitness_experiment_long.parquet` | Inner join of GeneFitness ⋈ Experiment |
| Canonical experiments | `data/derived/canonical/v0/experiments.parquet` | |
| Media master | `data/derived/canonical/v0/media_master.parquet` | |
| Media components | `data/derived/canonical/v0/media_components_long.parquet` | |

## Required fields

See `schemas/canonical_tables.schema.json` and `schemas/condition_features.schema.json`.

## Preprocessing policy

- Condition vocab/scalers fit on **train split only**.
- Unseen val/test categories → explicit `UNK` token; log unknown-category rate per run.
- Any feature trimming fit on train only and persisted as a named artifact.

## Checksums

Populate from `docs/canonical_build_manifest_v0.json` and `docs/embedding_manifest_m4.json`
after Stage-0 verification pass.
