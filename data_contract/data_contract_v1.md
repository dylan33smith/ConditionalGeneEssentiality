# Data Contract v1

**Status:** scaffolded; checksums and verification artifact populated by S0.

This contract pins what data is in scope, what its schema is, and what every run
must record in its manifest. Authoritative for all stages and tiers per
`docs/REFACTORPLAN.md` §3.

## Authoritative sources

| Artifact | Path | Notes |
|---|---|---|
| Raw fitness DB | `data/raw/feba.db` | Never modified |
| Condition workbook | `data/media_composition_v4.xlsx` | Sheet `Media_Components_ML` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` | Frozen |
| Canonical fitness table | `data/derived/canonical/v0/fitness_experiment_long.parquet` | Inner join GeneFitness ⋈ Experiment |
| Canonical experiments | `data/derived/canonical/v0/experiments.parquet` | |
| Media master | `data/derived/canonical/v0/media_master.parquet` | |
| Media components | `data/derived/canonical/v0/media_components_long.parquet` | |

## Schemas

| Artifact | Schema |
|---|---|
| Run manifest | `schemas/run_manifest_v1.schema.json` |
| Canonical tables | `schemas/canonical_tables.schema.json` |
| Condition features (v4 sheet) | `schemas/condition_features.schema.json` |

## Stage/tier handoff artifacts

These files are emitted by the indicated stage and consumed by downstream stages.
Any stage may not advance without producing its handoff artifact and updating the
decision ledger.

| File | Emitter | Consumers |
|---|---|---|
| `v4_schema_verification.json` | S0 | all |
| `splits/candidate_protocols.yaml` | S1 | S2, S3 |
| `policy/eval_policy.yaml` | S2 | S3, all tiers |
| `splits/locked_protocol.yaml` | S3 | S4, S5, all tiers |
| `splits/diagnostic_protocols.yaml` | S3 | T1+, reporting only |
| `feature_contract.yaml` | S4 | S5, all tiers |
| `preprocessing/<artifact_id>/` | S4 | all tiers |
| `policy/quality_policy.yaml` | S5 | all tiers |
| `representation_winner.yaml` | T1 | T2 |
| `fusion_winner.yaml` | T2 | T3 |
| `architecture_winner.yaml` | T3 | T4 |

## Preprocessing policy

- Condition vocab/scalers fit on **train rows only** under the locked split.
- Unseen val/test categories → `<UNK>` (index 0); unknown-category rate logged per run.
- Any feature trimming fit on train only and persisted as a named artifact in
  `preprocessing/<artifact_id>/`.
- The `Decomposition_type` column drives the per-medium `representation_mode` tag
  (`physical` | `mix` | `in_silico`); physical and in-silico features must not be
  silently merged into one untagged feature space.

## Per-run hard gate

Every reported run must validate against `schemas/run_manifest_v1.schema.json`
and include:
- `feba_db_sha256`
- `workbook_v4_sha256` + sheet id `Media_Components_ML`
- `embedding_manifest_id`
- `canonical_manifest_id`
- `git_sha`, `code_sha`, `seed`
- `split_protocol.protocol_id` + `split_manifest_sha256`
- `preprocessing_artifact_id`
- `scored_rowset_hash`, `n_rows_scored`, inclusion/exclusion counters
- `metrics.rmse`, `metrics.mae` (co-primary)
- `null_baseline_deltas.global_train_mean` and `null_baseline_deltas.additive_baseline`
- `unknown_category_rate`

Runs missing any required field are `exploratory` and ineligible for promotion.
