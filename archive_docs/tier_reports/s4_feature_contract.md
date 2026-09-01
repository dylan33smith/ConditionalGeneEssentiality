# Stage 4 Report — Feature Contract (Option D)

**Status:** approved (2026-04-29). **Regenerated 2026-04-30** (raw `canonical_id` in
`experiment_chemistry.parquet`, chemistry prevalence threshold **0.0**, metadata
only — **no** `genotype` / `mutantLibrary` in the locked wide table). See
[`decisions/stage4/S4-DEC-002.md`](../decisions/stage4/S4-DEC-002.md) (supersedes
S4-DEC-001).

---

## TL;DR

- S4-DEC-002 locks **Option D**: per-experiment **long** chemistry
  (`experiment_chemistry.parquet`) + **wide** encoded metadata
  (`experiment_metadata.parquet`).
- Preprocessing artifact id: **`de21504134c84a6c`**.
- Contract: `data_contract/feature_contract.yaml` (`schema_version: s4_option_d_v1`).
- Stressor resolution: `data_contract/preprocessing/_review/stressor_to_canonical_id.yaml`
  (Phase 0: `scripts/build_stressor_match_report.py` + CSV report).
- Unified canonical vocab (train listing at chem threshold 0.0): **423** workbook/stressor
  canonicals + `<UNK>` + `<UNK_STRESSOR>` = **425** slots (`n_posttrim` matches `n_pretrim`).
- Eval chemistry rows: **`fraction_canonical_id_not_in_train_vocab` = 0.0** (every eval
  row’s `canonical_id` string appears in the train union; OOV/rare **encoding** is T1’s job).
- **`representation_mode_per_canonical.parquet`** — one row per chemistry-union
  `canonical_id`; stressor-only canonicals use `dominant_mode=stressor`.

## Inputs consumed

- `data_contract/splits/locked_protocol.yaml`
- `data/derived/canonical/v0/experiments.parquet`
- `data/media_composition_v4.xlsx` (`Media_Components_ML`)
- `data_contract/representation_mode_mapping.yaml`
- `data_contract/preprocessing/_review/stressor_to_canonical_id.yaml`

## Locked outputs

- `data_contract/feature_contract.yaml`
- `data_contract/preprocessing/de21504134c84a6c/`
  - `canonical_id_vocab.json`
  - `experiment_chemistry.parquet`
  - `experiment_metadata.parquet`
  - `representation_mode_per_media.parquet`
  - `representation_mode_per_canonical.parquet`
  - `amount_scaler.json`
  - `bounded_reference_stats.json`
  - `numeric_metadata_scalers.json`
  - `stressor_to_canonical_id.yaml` (copy of ratified map)
  - `metadata_vocabs/{oxygen,experiment_group,liquid_state}.json`
  - `artifact_manifest.json`

## Contract summary

### Unified chemistry

- **Parquet policy:** raw `canonical_id` strings; **no** S4 prevalence replacement in the
  long table. `<UNK>` / `<UNK_STRESSOR>` are reserved vocab indices for **T1** mapping.
- Prevalence unit (vocab listing): **distinct train experiments** per `canonical_id`
  (medium ∪ stressor); `chemistry_prevalence_threshold: 0.0` ⇒ all train-seen ids listed.
- `log1p_amount` uses one train-fitted scaler over finite non-negative `amount` values
  (workbook + stressor concentrations; **`units_1..units_4` not read** — see contract
  `concentration_policy`).

### Metadata (wide table)

| Logical field | Source | Eval unknown rate |
|---|---|---:|
| oxygen | `aerobic` | 0.0 |
| experiment_group | `expGroup` | ~0.0033 |
| liquid_state | `liquid` | 0.0 |

Numeric z-scores: `temperature_c`, `pH`, `shaking_rpm` (+ `_is_finite` flags).

## S3 cross-check (Option D)

`scripts/audit_s3_chemistry_seen_option_d.py`: val experiment all-canon-seen rate remains
**1.0** for `multi_org_balanced` (best among four candidates). Footnote recorded on
`S3-DEC-001.md`.

## Acceptance evidence

- Stage run: `python -m src.cli.run_experiment +stage=s4_feature_contract`
- Manifest SHA256 recorded in `feature_contract.yaml`
- Unit tests: `tests/unit/test_stage4_feature_contract.py`, `tests/unit/test_stressor_matcher.py`

## Next action

Proceed to **S5** using this contract and artifact id **`de21504134c84a6c`**
(or whatever `artifact_id` the root `feature_contract.yaml` declares after the next S4 run).
