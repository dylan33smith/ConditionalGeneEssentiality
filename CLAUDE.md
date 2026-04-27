# Project: ConditionalGeneEssentiality

Predicting conditional gene essentiality from Tn-seq fitness data using frozen
ProteomeLM gene embeddings + condition (media chemistry) features.
Primary objective: regression on continuous `fit` scores for `(gene, condition)` pairs.

## Active branch

`refactor` — all new work goes here. `main` has the pre-refactor code.

## Governing plan

`docs/REFACTORPLAN.md` (v2) is the single source of truth for what to build and in
what order. Read it before proposing any new work. v1 plan archived at
`archive/docs/REFACTORPLAN_v1.md`.

## Stage / Tier pipeline

```
S0 → S1 → S2 → S3 → S4 → S5 → T1 → T2 → T3 → T4
```

| Phase | Concern | Status |
|---|---|---|
| S0 | Reproducibility & Governance (smoke pipeline, run manifest, v4 verification) | **approved** (S0-DEC-001) |
| S1 | Data Characterization → emits candidate protocols | **approved** (S1-DEC-001); H-HOMO-01 triggered at 1.6σ |
| S2 | Evaluation Trustworthiness (null baselines + power report per candidate) | next |
| S3 | Split Protocol Lock | not started |
| S4 | Feature Contract (train-only vocab/scalers) | not started |
| S5 | Training-Recipe Lock (row-quality + organism pool) | not started |
| T1 | Representation winner | not started |
| T2 | Fusion winner (all topology decisions live here, not T3) | not started |
| T3 | Capacity (depth/residuals/efficiency frontier; conditional embedding fine-tune) | not started |
| T4 | Optimization + loss family + target normalization locks | not started |

## Authoritative data inputs

| Artifact | Path |
|---|---|
| Raw fitness DB | `data/raw/feba.db` |
| Condition workbook | `data/media_composition_v4.xlsx`, sheet `Media_Components_ML` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` |
| Canonical fitness | `data/derived/canonical/v0/fitness_experiment_long.parquet` |
| Canonical experiments | `data/derived/canonical/v0/experiments.parquet` |
| Media master | `data/derived/canonical/v0/media_master.parquet` |
| Media components | `data/derived/canonical/v0/media_components_long.parquet` |

Older workbook versions (v1–v3) are explicitly out of scope.

## Project layout

```
src/                 implementation code (single source of truth)
configs/             Hydra config tree (config.yaml + group dirs)
data_contract/       schemas + frozen handoff artifacts (one per stage/tier)
tests/               unit + integration tests (must stay green)
research_log/        decision ledger entries + tier reports
artifacts/           run outputs (gitignored)
archive/             legacy pre-refactor code (reference only)
```

## Running experiments

Hydra entrypoint:

```bash
python -m src.cli.run_experiment +stage=s0_reproducibility
python -m src.cli.run_experiment +experiment=T1-A_granularity
python -m src.cli.run_experiment +experiment=T1-A_granularity train.seed=0,1,2 -m
```

## Scope of generalization claim (locked, REFACTORPLAN L7)

> "Given a gene and a growth medium drawn from a known chemistry vocabulary, our
> model predicts conditional gene essentiality — including for organisms not seen
> during training, and conditions structured differently from those the gene
> appeared in during training."

S1 confirmed v4 chemistry overlap is ≥95% in every candidate protocol. We do
NOT claim "generalizes to any chemistry." Going beyond requires fingerprint
encoders or Canonical_ID-level holdouts (REFACTORPLAN §12, Deferred Experiments).

## Hard rules (clean-room charter)

- **No tier or stage promotion** without pre-declared success criteria + decision-ledger entry.
- **Train-only preprocessing.** Vocab/scalers fit on train rows only; val/test unseen
  categories → explicit `<UNK>`; log unknown-category rate every run.
- **Co-primary metrics:** RMSE + MAE. Neither may be omitted from a promotion decision.
- **Additive baseline gate (H-BASE-01):** any model that does not beat
  `fit ~ a + α[gene] + β[condition]` on RMSE+MAE is ineligible for promotion.
- **Denominator parity:** model and baseline scored on the exact same row set.
- **Legacy results untrusted.** Prior metrics from `archive/` may only appear as
  hypotheses to re-test. Nothing auto-promotes.
- **Every run must log:** git SHA, data-contract checksums (feba_db, workbook_v4,
  embeddings, canonical), split protocol id, preprocessing artifact id, config snapshot,
  seed, scored_rowset_hash, metrics, null-baseline deltas, unknown-category rate.
  Validated against `data_contract/schemas/run_manifest_v1.schema.json`.

## Stage/Tier ownership of decisions (no overlap)

- **Feature schema instance** (which encoder wins) → T1 only. NOT S5.
- **All fusion topology decisions** (concat / two-tower / FiLM / gating) → T2 only. NOT T3.
- **Capacity decisions** (depth / residuals / efficiency frontier) → T3 only. NOT T2.
- **Loss family + target normalization** → T4 only. NOT S5.

## Test policy

`pytest tests/` must stay green before any commit.
Stub tests for unimplemented modules use `@pytest.mark.skip` with a
`reason=` pointing at the stage/tier that will activate them.

## Visualization policy (REFACTORPLAN §11)

Every stage/tier with a tier report must include figures that inform its
hard-gate decisions. Figures live at `research_log/figures/<stage_or_tier>/`
as `NN_descriptive_name.png` + sibling `.csv`. Plotting helpers live in
`src/evaluation/reporting.py` — stages call them rather than reinventing
matplotlib code. Required figure lists are inline in each stage's spec
(see e.g. §7 S1 → 24 required + 1 optional).

## Decision log

Decisions go in `research_log/decisions/<stage_or_tier>/` using the template at
`research_log/decisions/decision_template.md`.
