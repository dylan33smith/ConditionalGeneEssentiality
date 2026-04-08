# Missing items inventory vs `data/media_composition.xlsx`

This inventory lists items present in canonical data (`data/derived/canonical/v0/experiments.parquet`) but missing from workbook sheets.

## Counts

- Missing `(orgId, name)` rows in `Experiments`: **2755**
- Missing media names in `Media_Components`: **76**
- Missing media names in `Media` sheet: **76**
- Missing `Experiments` rows where media already exists in `Media_Components`: **913**
- Canonical orgIds absent from `Experiments` sheet orgId column: **16**

## Files to give AI (together with workbook)

- `data_analysis/outputs/missing_in_workbook__Experiments_sheet_rows.csv`
- `data_analysis/outputs/missing_in_workbook__Media_Components_media_names.csv`
- `data_analysis/outputs/missing_in_workbook__Media_sheet_media_names.csv`
- `data_analysis/outputs/missing_in_workbook__Experiments_rows_where_media_components_exist.csv`
- `data_analysis/outputs/missing_in_workbook__orgIds_absent_from_Experiments_sheet.csv`

## Important schema note

- `Organisms_Refs` does not contain `orgId` (columns are name/source-oriented), so exact machine join from canonical `orgId` to that sheet is not possible without an alias map.

## Column mapping hint for `Experiments` sheet

- canonical `expName` -> workbook `name`
- canonical `media` -> workbook `Media`
- canonical `expGroup` -> workbook `Group`
- canonical `condition_1..4`, `concentration_1..4`, `units_1..4`, `temperature`, `pH` map by name/case