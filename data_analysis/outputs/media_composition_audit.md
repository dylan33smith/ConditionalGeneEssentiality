# Media composition workbook audit (Phase 0)

Generated: 2026-04-05T19:25:15Z UTC
File: `data/media_composition.xlsx` (repo root-relative)

## Sheet inventory

| Index | Sheet name | Rows | Cols |
| ----- | ---------- | ---- | ---- |
| 0 | Media | 46 | 6 |
| 1 | Media_Components | 423 | 8 |
| 2 | Experiments | 4871 | 21 |
| 3 | Organisms_Refs | 33 | 8 |
| 4 | Coverage_Summary | 6 | 2 |

## DB vs `Experiments` sheet (join keys)

- Matched `(orgId, expName)` rows: **4797** / 7552 DB experiments.
- Unmatched DB experiments (no Excel row): **2755**.
- Media string mismatches where both present: **0**.
- Excel `Experiments` sheet rows: 4870.

## Authoritative mapping (restart)

- **Medium → components:** sheet **`Media_Components`** (index 1): columns *Media*, *Component*, *Concentration*, *Units*, …
- **Experiment ↔ medium / metadata:** sheet **`Experiments`** (index 2): join to `Experiment` table on **`(orgId, name)`** ↔ **`(orgId, expName)`**; medium string in column *Media*.
- v1 loader note referenced sheet index 2 for *composition*; this workbook places **composition** on **`Media_Components`**, not index 2. Index 2 is experiment-level metadata.

## `Media_Components` header row (row 0)

| Media | Component | Concentration | Units | Media_description | Minimal | Source_dataset | Source_url |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALP_defined_noC | Ammonium chloride | 0.25 | g/L | Defined media for Dechlorosoma suillum PS minus carbon, pH should be 7.2 | 1.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_defined_noC | Potassium Chloride | 0.1 | g/L | Defined media for Dechlorosoma suillum PS minus carbon, pH should be 7.2 | 1.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_defined_noC | Potassium phosphate dibasic | 0.97 | g/L | Defined media for Dechlorosoma suillum PS minus carbon, pH should be 7.2 | 1.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_defined_noC | Sodium phosphate monobasic monohydrate | 0.49 | g/L | Defined media for Dechlorosoma suillum PS minus carbon, pH should be 7.2 | 1.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_defined_noC | Wolfe's mineral mix | 1.0 | X | Defined media for Dechlorosoma suillum PS minus carbon, pH should be 7.2 | 1.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_defined_noC | Wolfe's vitamin mix | 1.0 | X | Defined media for Dechlorosoma suillum PS minus carbon, pH should be 7.2 | 1.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Ammonium chloride | 0.25 | g/L | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Potassium Chloride | 0.1 | g/L | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Potassium phosphate dibasic | 0.97 | g/L | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Sodium D,L-Lactate | 40.0 | mM | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Sodium acetate | 0.8203 | g/L | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Sodium phosphate monobasic monohydrate | 0.49 | g/L | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Sodium pyruvate | 1.1 | g/L | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Wolfe's mineral mix | 1.0 | X | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |
| ALP_rich | Wolfe's vitamin mix | 1.0 | X | Rich media for Dechlorosoma suillum PS, pH should be 7.2 | 0.0 | Price et al. bigfit Supplementary_Tables_final.xlsx, TableS18_Medias | https://genomics.lbl.gov/supplemental/bigfit/Supplementary_Tables_final.xlsx |

*(422 component rows total.)*

## `Experiments` sheet columns (header row)

- `orgId`
- `organism`
- `name`
- `Group`
- `short`
- `Media`
- `Condition_1`
- `Concentration_1`
- `Units_1`
- `Temperature`
- `pH`
- `Aerobic_v_Anaerobic`
- `Shaking`
- `Growth.Method`
- `has_exact_replicate`
- `has_similar_replicate`
- `Organism_reference`
- `Experiment_source_dataset`
- `Experiment_source_url`
- `Media_source_dataset`
- `Media_source_url`
