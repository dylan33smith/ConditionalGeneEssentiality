# OPEN-001 — Condition stressors are chemistry, not metadata

**Status:** RESOLVED (see footer — S4-DEC-002)
**Priority:** high
**Filed:** 2026-04-28
**Decision (2026-04-29):** Option D selected (per-experiment chemistry table + per-experiment metadata table). See "Reformulation options" below.
**Owner:** next session
**Affects:** S1 scope-of-claim, S2 additive baseline interpretation, S3 split selection (potentially), S4 feature contract (directly)

---

## TL;DR

`condition_1..4` in `experiments.parquet` are not metadata. They are **chemicals applied to the experiment** — additional perturbants on top of the base medium. **Before S4-DEC-002**, S4 incorrectly encoded them as categorical metadata tokens (`growth_phase`, `genotype`), which was doubly wrong:

1. It treats chemicals as opaque category names.
2. The current S4 vocabularies for `growth_phase` and `genotype` are full of stressor chemical names (e.g. `Nickel (II) chloride hexahydrate`), confirming the misclassification.

The chemistry encoding side of S4 (media-only multihot, pre–S4-DEC-002) was unaware of these applied stressors entirely.

**Resolved:** S4-DEC-002 reformulated the contract (Option D). Proceed to S5 using the `artifact_id` in `data_contract/feature_contract.yaml` (currently `de21504134c84a6c`).

---

## Quantified evidence

Run on the locked dataset (`experiments.parquet`, `media_composition_v4.xlsx` ML sheet), April 2026:

- 376 unique stressor strings appear across `condition_1..4`.
- 87.8% of experiments (6,630 / 7,552) have at least one non-null `condition_*`.
- Stressor mentions matched against the v4 chemistry vocab (case-insensitive, against `Canonical_ID` ∪ `Compound_name` ∪ `Source_row_component`):
  - **2.5%** of stressor mentions are already listed in the experiment's own media (pure redundancy).
  - **32.9%** are chemicals that exist in the v4 vocab but are NOT in the experiment's media row (the chemistry vector under-represents the actual condition).
  - **64.6%** are chemicals that the v4 workbook does not catalog at all (mostly carbon sources, antibiotics, ionic liquids, furans).
- Only 60 of the 376 unique stressor strings (16%) match anywhere in the chemistry vocab.

Examples already-in-media (redundancy):
- `L-Lysine` stressor on `marine_broth_2216`
- `sodium fluoride` stressor on `marine_broth_2216`
- `Sodium D,L-Lactate` stressor on `Dv_base_medium`

Examples chemistry-known but missing from media (under-represented in chemistry vector):
- `Cobalt chloride hexahydrate` on `LB`
- `copper (II) chloride dihydrate` on `LB`
- `D-Glucose` on `RCH2_defined_noCarbon`

Examples not in v4 at all (chemistry-vocab gap):
- `1,2-Propanediol`, `2-Furfuraldehyde`, `2-Mercaptopyridine N-oxide sodium salt`,
  `1-ethyl-3-methylimidazolium acetate`, `Sisomicin sulfate salt`, `2-n-Heptyl-4-hydroxyquinoline N-oxide`.

---

## Impact on prior stages

- **S1**: The 95% Canonical_ID overlap claim is scoped to media chemistry only. The "known chemistry vocabulary" wording in the locked scope-of-claim (REFACTORPLAN L7) is ambiguous about whether stressor chemistry counts. Status quo: the contract only covers media.
- **S2**: Baseline numbers are mechanically unaffected (baselines only used `condition = media`), but the additive baseline `fit ~ a + α[gene] + β[condition]` cannot see applied stressors, so it is artificially weak. The H-BASE-01 bar is therefore lower than it should be. Re-running null baselines is not strictly required but should be reconsidered after the chemistry reformulation.
- **S3**: Split selection used media-name and Canonical-ID-of-media seen-rates. If "condition" is redefined to include applied stressors, val/test seen-rates change. After S4 reformulation, S3 selection may need to be rerun and `S3-DEC-001` re-checked or amended.
- **S4**: Directly broken. Two interlocking issues:
  1. The metadata-field mapping in `configs/stage/s4_feature_contract.yaml` (`growth_phase: condition_1`, `genotype: condition_2`) treats chemicals as categories. The emitted `metadata_vocabs/growth_phase.json` and `metadata_vocabs/genotype.json` are full of chemical names — proof that the design is wrong.
  2. The chemistry-side multihot (`media_to_multihot.parquet`) does not include applied stressors. The model has no way to know that LB+Cobalt-chloride differs from plain LB.

The current `feature_contract.yaml` (`artifact_id: 1f6f32d31d50b1c8`) is therefore incorrect and should not be consumed by T1.

---

## Reformulation options

### Option A — Single collapsed condition vector
Replace `media_multihot` with a per-experiment `condition_multihot = (media chemistry) ∪ (applied stressor chemistry)`.

- Pro: matches the modeling intent (one chemistry context per experiment).
- Con: loses the structural distinction between background medium and applied perturbation; multihot cannot express dose; requires extending v4 to cover the missing 316 stressor compounds (or absorbing them into `<UNK_STRESSOR>`).

### Option B — Parallel multihots (recommended)
Keep `media_multihot` as-is. Add a second `stressor_multihot` indexed against the same Canonical-ID vocab, plus an `<UNK_STRESSOR>` slot for unmatched compounds.

- Pro: preserves S1's analysis of media chemistry; lets the fusion stage learn distinct weights for "what's in the medium" vs "what was added"; the simplest local reformulation.
- Con: vocab needs a stressor-name → Canonical-ID resolution table. `<UNK_STRESSOR>` rate must be logged per experiment. Some workbook expansion is desirable to shrink the unknown rate.

### Option C — Defer chemistry-of-stressor to T1
Treat `condition_1..4` as raw strings in S4. Push the encoding choice into T1-D and the fusion topology into T2.

- Pro: avoids locking in a possibly-wrong design.
- Con: defeats the purpose of S4 (locking the schema before T1 starts); the missing-compound problem still has to be solved later.

### Option D — Per-experiment chemistry + metadata tables (SELECTED)

S4 freezes two normalized parquet artifacts and lets T1 choose how to encode each.

#### `experiment_chemistry.parquet` — long format, one row per (experiment, chemical, role)

| Column         | Type   | Notes |
|----------------|--------|-------|
| experiment_id  | str    | Joins to `experiments.parquet` |
| canonical_id   | str    | Train-only frozen vocab; unmatched stressors → `<UNK_STRESSOR>` |
| role           | enum   | `medium` or `stressor` |
| log1p_amount   | float? | Train-only fitted log1p scaler; `NaN` if unknown/inapplicable |

Built by:
- Joining each `experiments.parquet` row to `media_components_long.parquet` on `media`, emitting `role=medium` rows with the workbook's `Amount` (log1p-scaled).
- Resolving each non-null `condition_1..4` value via a `stressor_to_canonical_id.yaml` lookup table, emitting `role=stressor` rows.

#### Tokenization policy for stressor `canonical_id`s (corrected 2026-04-29)

Each unique stressor string becomes its own `canonical_id` slot — we do **not** collapse the 316 unmatched stressors into a single `<UNK_STRESSOR>` token. The chemistry vocab grows by adding one new `canonical_id` for each (lightly normalized) stressor string, so e.g. `Cisplatin`, `Vancomycin Hydrochloride Hydrate`, `Paraquat dichloride` each get distinct slots alongside the existing media-derived canonical_ids.

Resolution flow per stressor string:
1. If the string matches the v4 workbook (case-insensitive against `Canonical_ID ∪ Compound_name ∪ Source_row_component`), use the workbook's existing canonical_id. Currently 60 of 376 unique stressor strings match this way; a normalization-cleanup pass (see action 4 below) is expected to grow that number by absorbing obvious synonyms like `Ethanol`, `Agar`, `Sodium sulfate` that should already be matching but aren't due to a string-matcher bug.
2. Otherwise the string itself becomes a new canonical_id (after light normalization: case-fold, whitespace collapse).
3. Apply the existing prevalence trim (currently `0.001 × n_train_experiments ≈ 7`) to **both** media-derived and stressor-derived canonical_ids uniformly. Train-prevalence numbers measured 2026-04-29 (see "Prevalence under D" table below): at this threshold, 160 of the 316 unmatched stressors survive as first-class canonical_ids and cover 91.8% of stressor mentions. Surviving slot count for the unified vocab is roughly `~118 (existing media) + ~160 (new stressor-derived) ≈ ~278`.

**Role of `<UNK_STRESSOR>` (corrected).** Reserved for two cases only:
- A stressor string that fell below the train-only prevalence threshold (long tail).
- A stressor string never seen in train but appearing in val/test/inference (unseen-at-eval).

It is **not** used as a "we don't have workbook metadata for this chemical" bucket. Workbook annotation status is orthogonal to vocab membership — see next subsection.

#### Prevalence under D (train-only, locked split, 6,938 train experiments)

| Min train-experiments to keep | Stressor canonical_ids retained | Stressor mentions covered |
|---|---|---|
| ≥1 (no trim) | 316 | 100.0% |
| ≥2 | 285 | 99.3% |
| ≥3 | 217 | 96.3% |
| ≥6 (current 0.001 chem threshold) | 160 | 91.8% |
| ≥10 | 115 | 84.4% |
| ≥20 | 70 | 69.8% |

S4-DEC-002 should ratify a single threshold (consistency with media canonical_ids → 0.001 default; could lower to e.g. 0.0003 if the long tail matters).

#### Tokenization vs annotation are orthogonal

- **Tokenization** (which strings become canonical_ids in the vocab) happens automatically from the rules above. No workbook expansion needed.
- **Annotation** (filling in `Compound_name`, `Decomposition_type`, `Source_row_component`, etc. for the new stressor canonical_ids in `media_composition_v4.xlsx`) is a separable cleanup pass. Until it happens, stressor canonical_ids will have `representation_mode = in_silico` (or whatever the default is) and no per-component `Amount` entry. That is acceptable — the model still has a distinct slot per chemical to learn against.

Encoders downstream (T1) can choose freely:
- Group by experiment and OR-aggregate canonical_id → recovers the old single multihot.
- Group by (experiment, role) → recovers Option B's parallel multihots.
- Embed each canonical_id, weight by `log1p_amount`, pool per experiment → role- and dose-aware (Option D-native).
- Anything T1-D wants to test.

#### `experiment_metadata.parquet` — wide format, one row per experiment

Recommended starting field list (final list locked in S4-DEC-002):

Categorical (each with frozen train-only vocab + `<UNK>`/`<MISSING>` fixed indices):
- `oxygen` ← `aerobic` (3 unique: Aerobic / Anaerobic / Microaerobic)
- `genotype` ← `mutantLibrary` (~60 unique strain libraries — the actual genotype source, replacing the broken `condition_2` mapping)
- `experiment_group` ← `expGroup` (~48 unique like `stress`, `lb`, `carbon source`, `temperature`, `motility`, `pH`; replaces the broken `growth_phase`/`condition_1` mapping)
- `liquid_state` ← `liquid` (Liquid / Solid)

Numeric (each with frozen train-only scaler):
- `temperature_c` ← parsed from `temperature` (string in raw, needs cleanup)
- `pH` ← `pH`
- `shaking_rpm` ← parsed from `shaking` (e.g. `'750 rpm'`, `'200 rpm'`); unparseable values (`'orbital'`, `'0 rpm'`) fall back to a categorical `<UNK_SHAKING>` token and `NaN` rpm

#### Why two files, not one?

Chemistry is one-to-many per experiment (long table). Metadata is one-to-one per experiment (wide table). Putting `temperature` / `pH` / `oxygen` into the chemistry table would duplicate the scalar across every chemistry row of that experiment, denormalize the schema, and force the chemistry encoder and the metadata encoder to be developed in lockstep. Keeping them separate lets T1 iterate on each independently, and the join is trivial (`experiment_id`).

#### What stays the same as the current S4

- Train-only `canonical_id_vocab.json` (with `<UNK_STRESSOR>` added if we add the slot).
- Train-only `amount_scaler.json` (log1p Amount).
- `representation_mode_per_media.parquet`.
- `bounded_reference_stats.json`.
- `<UNK>=0`, `<MISSING>=1` index policy on every metadata field.
- The artifact-id-pinned directory layout.

#### What changes

- `media_to_multihot.parquet` is replaced by `experiment_chemistry.parquet`. (T1 can re-derive the old multihot from the table in two lines if it wants the baseline.)
- The current `metadata_vocabs/growth_phase.json` and `metadata_vocabs/genotype.json` (which are full of stressor chemicals) are deleted and replaced with the corrected metadata field set above.
- New `experiment_metadata.parquet` artifact emitted alongside the JSON vocabs/scalers.
- New `stressor_to_canonical_id.yaml` resolution table emitted as part of the locked artifact bundle.
- New `artifact_id`; re-emitted `feature_contract.yaml`.

---

## Specific actions before moving to S5

1. **Author S4-DEC-002** (supersedes S4-DEC-001).
   - Status: Option D selected.
   - Pre-register the two new artifact schemas (`experiment_chemistry.parquet` long; `experiment_metadata.parquet` wide).
   - Pre-register the final metadata field list (the categorical + numeric column choices in Option D above; narrow or widen as desired).
   - Lock the `<UNK_STRESSOR>` policy: single canonical slot, treated as a normal vocab token, log1p_amount=NaN.
   - Decide whether `<UNK_STRESSOR>` is added to the canonical_id vocab (and therefore the canonical_id vocab grows by one slot) or kept as a sentinel string that the encoder maps separately. (Recommended: add to vocab so all canonical_id strings live in one namespace.)

2. **Build `data_contract/preprocessing/<artifact_id>/stressor_to_canonical_id.yaml`.**
   - Programmatic seed: case-insensitive match of unique `condition_1..4` strings against `Canonical_ID ∪ Compound_name ∪ Source_row_component` (current data: 60 of 376 unique stressor strings match the workbook).
   - Manual review of the matched rows for ambiguity (a stressor name may collide with multiple canonical IDs).
   - Unmatched stressor strings (currently 316) get **their own new `canonical_id` slots** (lightly normalized: case-fold, whitespace collapse), and become first-class entries in the unified chemistry vocab. They are subject to the same train-only prevalence trim as media canonical_ids; entries below threshold fall to `<UNK_STRESSOR>`.
   - Fix the string-matcher bug visible in the top-30 list: entries like `Ethanol`, `Agar`, `Sodium sulfate`, `sodium sulfite` should be matching the workbook but aren't. A pass that normalizes `hydrochloride`/`hydrate`/`sodium salt` suffixes and case before lookup will likely rescue 5–20 of the unmatched entries into existing canonical_ids.

3. **Refactor S4 to emit Option D artifacts.**
   - Replace `media_to_multihot.parquet` with `experiment_chemistry.parquet` (built from the join described above).
   - Replace the current `metadata_vocabs/{growth_phase,genotype}.json` (contents are wrong — full of stressor chemicals) with the corrected metadata field set.
   - Emit `experiment_metadata.parquet` alongside the per-field JSON vocab files.
   - Keep the train-only `canonical_id_vocab.json`, `amount_scaler.json`, `representation_mode_per_media.parquet`, `bounded_reference_stats.json` artifacts (rebuilt under the augmented vocab if `<UNK_STRESSOR>` is added).
   - New `artifact_id`. Re-emit `feature_contract.yaml` with `experiment_chemistry_table` and `experiment_metadata_table` schema entries replacing the old `media_to_multihot` entry.

4. **One-pass workbook annotation** (recommended but not strictly required for vocab membership).
   - Tokenization is already handled in step 2 — the new stressor canonical_ids exist in the vocab regardless of whether the workbook has metadata for them.
   - This step adds *side metadata* for the new stressor canonical_ids in `media_composition_v4.xlsx`: `Compound_name`, `Decomposition_type`, `Source_row_component`, etc. That metadata feeds into `representation_mode_per_media.parquet` and similar artifacts. Until annotated, new stressor canonical_ids default to `representation_mode = in_silico` (or whatever the policy default is).
   - Priority: the surviving ~160 stressor canonical_ids (those passing the prevalence trim). The long tail is a low-priority cleanup.
   - This is workbook *annotation*, not a v5 bump — the workbook gains rows but the canonicalization scheme is unchanged.

5. **Re-examine S3 split selection under the new condition definition.**
   - S3 used media-level seen-rates only. Under D, "condition" includes stressor chemistry. Recompute val/test seen-rates for the candidate protocols using the new chemistry table.
   - Either confirm `multi_org_balanced` still wins (S3-DEC-001 stays valid, footnote added) or amend in S3-DEC-002.

6. **Update REFACTORPLAN scope-of-claim wording (L7).**
   - Clarify that "known chemistry vocabulary" includes both medium-derived and applied (stressor) chemistry, with `<UNK_STRESSOR>` rate disclosed in every run manifest.
   - Update §S4 section to describe the new schema (chemistry table + metadata table) and to remove the stale "media_to_multihot" verbiage.

7. **Note in S2 tier report.**
   - The additive baseline `fit ~ a + α[gene] + β[condition]` in current S2 numbers uses `condition = media`. Under D, the additive baseline can use the (chemistry table, metadata table) joint; the H-BASE-01 bar may rise modestly. Re-running null baselines under the new condition definition is optional and can be deferred to a single re-run before T1 if desired.

8. **Update the test suite.**
   - `tests/unit/test_stage4_feature_contract.py` currently asserts the old multihot artifact and the old (broken) metadata-field set. It needs to be rewritten for the new schemas: chemistry-table integrity, role enum exhaustiveness, `<UNK_STRESSOR>` round-trip, metadata-table column presence, and per-field UNK/MISSING rates on the held-out slice.

---

## How this was discovered

S4 review surfaced that `metadata_vocabs/growth_phase.json` contained entries like `Nickel (II) chloride hexahydrate`. Tracing back showed `growth_phase` was wired to `experiments.parquet:condition_1`, which is a stressor chemical column, not a phase label. From there the question generalized to: how much of the stressor space is actually chemistry the v4 workbook already covers? Answer: a lot of it, and the part it doesn't cover is bigger.

---

## Resolved (2026-04-29)

Implemented as **S4-DEC-002** (supersedes S4-DEC-001): Option D chemistry +
metadata tables, stressor matcher + ratified
`data_contract/preprocessing/_review/stressor_to_canonical_id.yaml`, new
`artifact_id` from `data_contract/feature_contract.yaml` (currently **`de21504134c84a6c`**). S3 primary protocol re-audited under Option D
chemistry; `multi_org_balanced` unchanged (see appendix on `S3-DEC-001.md`).
