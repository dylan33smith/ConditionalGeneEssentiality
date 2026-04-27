# ConditionalGeneEssentiality

Tiered ML research system for conditional gene essentiality prediction from Tn-seq fitness data.

## Project plan

See [`docs/REFACTORPLAN.md`](docs/REFACTORPLAN.md) for the full tiered refactor plan, clean-room charter, and experiment roadmap.

## Repository layout

```
project_root/
  data_contract/      ← frozen data contract + schemas + split manifests
  src/                ← all implementation code (single source of truth)
    domain/           ← entities, contracts, metrics contract, split contract
    data/             ← ingestion, preprocessing, datasets
    models/           ← condition encoders, fusion modules, architectures
    train/            ← training loop, losses, optimizer, evaluator, checkpointing
    evaluation/       ← null baselines, metrics, reporting
    experiments/      ← per-tier experiment runners
    cli/              ← run_experiment entrypoint
  configs/            ← YAML configs per stage/tier/experiment
  tests/              ← unit + integration tests
  research_log/       ← decision ledger + tier reports
  artifacts/          ← run outputs (gitignored)
  docs/               ← REFACTORPLAN + data manifests
  archive/            ← legacy code from pre-refactor work (reference only)
```

## Environment

```bash
conda env create -f environment.yml
# or (lighter)
conda env create -f environment.min.yml
```

## Data inputs (authoritative)

- Raw fitness: `data/raw/feba.db`
- Condition composition: `data/media_composition_v4.xlsx` (sheet `Media_Components_ML`)
- Gene embeddings: `data/processed/ProtLM_embeddings_layer8/*.pt`
- Canonical tables: `data/derived/canonical/v0/*.parquet`

See `docs/REFACTORPLAN.md` §Authoritative Data Scope for full policy.
