# T1 — Option D condition encoding (mandatory reading)

**Audience:** whoever implements the first real T1 dataloader / encoder.  
**Authority:** `data_contract/feature_contract.yaml` (read `artifact_id` there — it
changes when S4 is re-run) + checksum tree under
`data_contract/preprocessing/<artifact_id>/`.

## What S4 emits (no old multihot)

| Artifact | Role |
|---|---|
| `experiment_chemistry.parquet` | Long table: one row per `(experiment_id, canonical_id, role)` with raw `canonical_id` strings, `amount`, train-fitted `log1p_amount`. |
| `experiment_metadata.parquet` | Wide table: one row per `experiment_id` with categorical `*_idx` and numeric `*_z` / `*_is_finite`. **No `genotype` / `mutantLibrary` column** — removed from the locked contract because organism-holdout makes it non-informative on eval. |
| `representation_mode_per_media.parquet` | Per **medium name** (workbook). |
| `representation_mode_per_canonical.parquet` | Per **`canonical_id`** appearing anywhere in the chemistry union. Workbook-derived canonicals get modes from `Decomposition_type`; stressor-only canonicals get `dominant_mode=stressor`, `prop_stressor=1`. |
| `canonical_id_vocab.json` | Train-only registry: `<UNK>` and `<UNK_STRESSOR>` reserved slots + every `canonical_id` seen on **train** experiments at least once (S4 uses `chemistry_prevalence_threshold: 0.0` ⇒ no prevalence drop). **Eval-only** canonical strings are **not** rewritten in the parquet — they remain literal strings until **your encoder** maps them to `<UNK>` / `<UNK_STRESSOR>` / a rare bucket / etc. |
| `stressor_to_canonical_id.yaml` | Human-auditable string → workbook canonical merges (`data_contract/preprocessing/_review/stressor_match_report.csv` + `.draft.yaml` from `scripts/build_stressor_match_report.py`). |

## Join key

`experiment_id` = SHA256-hex **64** chars over UTF-8  
`"{orgId}\x1f{setName}\x1f{seqindex}\x1f{media_key}"`  
where **`media_key` is the empty string** when `media` is missing (61 *Btheta* in vivo mouse time-course rows with no plate medium).

## Concentrations — **do not trust magnitudes yet**

S4 copies `concentration_1..4` into `amount` for `role=stressor` **without reading `units_1..units_4`**. All stressor amounts share one `log1p` scaler with workbook `Amount`. **T1 must not run experiments that interpret raw concentration as comparable molarity across rows** until units are normalized (or concentration features are explicitly disabled behind a flag).

## Matcher QA (human eyes)

- CSV: `data_contract/preprocessing/_review/stressor_match_report.csv` (376 rows at last build).  
- Ratified map: `data_contract/preprocessing/_review/stressor_to_canonical_id.yaml` (auto tier = exact + fuzzy ≥ 0.95).  
- Rebuild after editing the workbook or stressor strings: `python scripts/build_stressor_match_report.py`.

## What *not* to wire

- `media_to_multihot.parquet` — **gone** from the S4-DEC-002 contract.  
- `mutantLibrary` from S4 metadata — **not emitted**.
