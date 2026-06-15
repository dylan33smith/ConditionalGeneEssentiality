# Project: ConditionalGeneEssentiality

Predicting **conditional gene essentiality** from Tn-seq fitness data using frozen
ProteomeLM gene embeddings + media-chemistry features. The active objective is
**within-organism top-k ranking** of a gene's conditions ("find the top stressors
for this gene"). This file holds durable facts only — see `PLAN.md` for current
work and `PROGRESS.md` for history.

## Build / test / run

```bash
# run a ranking experiment (Hydra entrypoint; handler chosen by stage_or_tier)
python -m src.cli.run_experiment +experiment=R-LOSS_loss_family
python -m src.cli.run_experiment +experiment=R1-A_chemistry
python -m src.cli.run_experiment +experiment=R-CONF_confidence_strat

# REGRESSION CHECK — reproduces the locked-best model vs the chem-kNN baseline
# in one pinned command (split seed 0, k=5). Run after any change to ranking
# behavior; it gates against data_contract/ranking/reval_baseline.json.
python -m src.cli.run_experiment +experiment=R-EVAL_regression                # fast gate (Keio+Caulo+MR1)
# full 23-org headline (the replicate-org subset the published ~0.435/0.485 use —
# NOT orgs=null, which is all organisms and gives different, lower numbers):
python -m src.cli.run_experiment +experiment=R-EVAL_regression experiment.tag=full \
    experiment.model_seeds=[0,1,2] \
    'experiment.orgs=[ANA3,BFirm,Btheta,Burk376,Caulo,Cola,Cup4G11,Dda3937,Ddia6719,DdiaME23,Dino,DvH,Dyella79,HerbieS,Kang,Keio,Korea,Koxy,MR1,Marino,Methanococcus_JJ,Methanococcus_S2,Miya]'

# tests (must stay green before any commit)
python -m pytest tests/
```

Registered CLI handlers: `R0` (data characterization), `R1` (encoder),
`R-LOSS` (loss family), `R-CONF` (confidence stratification), `R-EVAL`
(regression check). The legacy T-regime handlers were pruned (see
`docs/PRUNED_INDEX.md`).

## Where things live

```
src/ranking/            the ranking objective (self-contained reusable core)
  data/                 fingerprints (+ data layer); split/eligibility/chemistry live in src/data/datasets
  models.py             AdapterResidualMLP + ResidualBlock (the locked model)
  losses/               pointwise (mse/huber) + ranking (ranknet/lambdarank/listmle/approxndcg) — top-k substrate
  eval/                 harness.py (canonical metrics + baselines + stats) + contract.py (R-LOCK-4 gate helpers)
  pipeline.py           prepare_r1_data (split+eligibility+features), train_r1_arm, eval harness
  train.py              loss-family trainers (pointwise row-batched / ranking gene-batched)
  runner.py             shared train→eval→report runner: ArmSpec + run_experiment + standardized_report
src/experiments/<R*>/run.py   thin CLI handlers that declare arms and call the runner
src/data/               data ingestion / preprocessing / encoders (provenance) + the ranking data modules
src/cli/run_experiment.py     Hydra entrypoint + handler dispatch
configs/experiment/     one yaml per experiment (R0/R1/R-LOSS/R-CONF/R-EVAL)
data_contract/          frozen handoff artifacts + schemas + the ranking metric contract + R-EVAL baseline
research_log/           decisions/ (the decision ledger), figures/, SCIENTIFIC_SYNTHESIS.md (canonical learnings)
ARCHITECTURE.md PLAN.md PROGRESS.md docs/PRUNED_INDEX.md
```

## How to add a new ranking test

A new test is declarative — no copy-pasted pipeline:

```python
from src.ranking.runner import ArmSpec, run_experiment
run_experiment(
    [ArmSpec("huber", loss="pointwise_huber"), ArmSpec("lambda", loss="lambdarank")],
    orgs=None, model_seeds=(0, 1, 2), out_dir="artifacts/runs/my_test", tag="my_test")
```
You get the standardized comparison (model + chem-kNN/NULL/linear-MF on the same
eligible val genes, per-seed + seed-mean, tidy CSV + side-by-side vs the gate).

## Conventions (hard rules)

- **Train-only preprocessing.** Vocab/scalers/eligibility-thresholds fit on train
  rows only; val unseen categories → explicit `<UNK>`; log unknown-category rate.
- **Denominator parity.** Model and every baseline are scored on the identical
  eligible val gene set.
- **The gate.** chem-kNN (NDCG@5 ~0.485) is the baseline a learned model must beat;
  promotion needs ΔNDCG@5 ≳ 0.026 with disjoint hierarchical-bootstrap CIs.
- **Co-primary metrics.** within-gene Spearman + NDCG@5 (k=5).
- **Regression discipline.** After any change to ranking behavior, run `R-EVAL`;
  if a number moves beyond tolerance, stop and investigate (do not "absorb" it).
- **Tests green** before any commit.

## Authoritative data inputs

| Artifact | Path |
|---|---|
| Raw fitness DB | `data/raw/feba.db` |
| Canonical fitness | `data/derived/canonical/v0/fitness_experiment_long.parquet` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` |
| Feature contract (S4, frozen) | `data_contract/preprocessing/de21504134c84a6c/` |

## Documentation-maintenance rule

At the end of any task that changes behavior, append an entry to PROGRESS.md and
update PLAN.md. Edit ARCHITECTURE.md only if component structure or data flow
changed. Never put in-progress status in CLAUDE.md or ARCHITECTURE.md.
