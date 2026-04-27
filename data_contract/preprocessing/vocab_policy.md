# Vocab Policy

**Status:** pending — fill in during Stage 1 condition encoding specification.

## Chemistry vocabulary

- Source: `Media_Components_ML` sheet, rows where `Include_in_ml == True`.
- Canonical ID vocab: ordered list of `Canonical_ID` values.
- Duplicate `(Media, Canonical_ID)` rows: idempotent (de-duplicate before indexing).
- Fit on train rows only; persist as artifact with split protocol id.

## Unknown category handling

- Any val/test `Canonical_ID` not in train vocab → index 0 (`UNK`).
- Log `unknown_category_rate` per run.
