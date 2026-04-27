# Project: ConditionalGeneEssentiality

Predicting conditional gene essentiality from Tn-seq fitness data using gene embeddings +
condition (media chemistry) features. Primary objective: regression on continuous `fit` scores
for (gene, condition) pairs.

## Active branch

`refactor` — all new work goes here. `main` has the pre-refactor code.

## Governing plan

`docs/REFACTORPLAN.md` is the single source of truth for what to build and in what order.
Read it before proposing any new work.

## Current stage

**Stage 0 — Reproducibility & Governance** (not yet complete)

The scaffold is in place but nothing runs end-to-end yet. Before any modeling work:
- Data contract v1 needs to be filled in (`data_contract/data_contract_v1.md`)
- `media_composition_v4.xlsx` sheet `Media_Components_ML` needs to be inspected and its schema documented
- A working training pipeline needs to exist (all `src/` modules are stubs)
- Stage 0 acceptance gate: a run must reproduce from config + seed + artifacts

After Stage 0: Stage 0.5 (null baselines), Stage 1 (data-sheet analysis), Stage 2 (split lock),
Stage 2.5 (data-quality policy), then Tier 1–4 experiments.

## Authoritative data inputs

| Artifact | Path |
|---|---|
| Raw fitness DB | `data/raw/feba.db` |
| Condition workbook | `data/media_composition_v4.xlsx`, sheet `Media_Components_ML` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` |
| Canonical fitness table | `data/derived/canonical/v0/fitness_experiment_long.parquet` |
| Canonical experiments | `data/derived/canonical/v0/experiments.parquet` |
| Media master | `data/derived/canonical/v0/media_master.parquet` |
| Media components | `data/derived/canonical/v0/media_components_long.parquet` |

Older workbook versions (v1–v3) are out of scope. Do not use them.

## Project layout

```
src/          all implementation code (single source of truth — no logic outside here)
configs/      YAML configs per stage/tier/experiment
data_contract/ frozen data contract + schemas + preprocessing policies
tests/        unit + integration tests (must stay green)
research_log/ decision ledger entries + tier reports
archive/      legacy pre-refactor code (reference only — do not import from here)
artifacts/    run outputs (gitignored)
```

## Hard rules (from clean-room charter)

- **No tier promotion without pre-declared success criteria and a decision-ledger entry.**
- **Train-only preprocessing:** vocab and scalers fit on train rows only; val/test unseen
  categories → explicit `UNK`; log unknown-category rate every run.
- **Legacy results are untrusted.** Prior metrics/conclusions may only appear as hypotheses
  to re-test. Do not auto-promote anything from `archive/`.
- **Denominator parity:** model and baseline must be evaluated on the exact same scored row set.
- **Every run must log:** git SHA, split protocol id, preprocessing artifact id, config snapshot,
  seed, metrics, null-baseline deltas, unknown-category rates.

## Test policy

`pytest tests/` must stay green before any commit. Two tests are intentionally skipped
(stubs for `fit_condition_scalers` and `build_model_dataset`); all others must pass.

## Decision log

Decisions go in `research_log/decisions/<tier>/` using the template at
`research_log/decisions/decision_template.md`.
