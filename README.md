# ConditionalGeneEssentiality

Predicting **conditional gene essentiality** from Tn-seq fitness data using frozen
ProteomeLM gene embeddings + media-chemistry features. The active objective is
**within-organism top-k ranking** of a gene's conditions ("find the top stressors
for this gene").

## Start here

- **[CLAUDE.md](CLAUDE.md)** — build/test/run commands, the regression-check command, conventions, where things live.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — components, data flow, locked design decisions + rationale.
- **[PLAN.md](PLAN.md)** — current objective and next tasks.
- **[PROGRESS.md](PROGRESS.md)** — append-only log of what was done and learned.
- **[research_log/SCIENTIFIC_SYNTHESIS.md](research_log/SCIENTIFIC_SYNTHESIS.md)** — the canonical "what we've learned" narrative.

## Quick start

```bash
# run a ranking experiment
python -m src.cli.run_experiment +experiment=R-LOSS_loss_family

# regression check: locked-best model vs chem-kNN baseline, one pinned command
python -m src.cli.run_experiment +experiment=R-EVAL_regression

# tests
python -m pytest tests/
```

## Layout

```
src/ranking/       the ranking objective (models, losses, eval, pipeline, train, runner)
src/experiments/   thin CLI handlers (R0/R1/R-LOSS/R-CONF/R-EVAL) over the runner
src/data/          data ingestion / preprocessing / encoders (provenance) + ranking data modules
src/cli/           Hydra entrypoint + handler dispatch
configs/           Hydra config tree (one experiment yaml per test)
data_contract/     frozen handoff artifacts + schemas + ranking metric contract + R-EVAL baseline
research_log/      decision ledger, figures, SCIENTIFIC_SYNTHESIS
docs/              data manifests + PRUNED_INDEX (what was pruned and where its learning lives)
tests/             unit + integration (keep green before any commit)
```

The legacy T-regime (cross-organism fitness regression) was pruned from this
branch; its results are preserved in the decision ledger and synthesis (see
[docs/PRUNED_INDEX.md](docs/PRUNED_INDEX.md)).
