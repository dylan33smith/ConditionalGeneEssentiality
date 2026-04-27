# Archive — legacy pre-refactor code

This directory contains all code and outputs from the project prior to the clean-room refactor
described in `docs/REFACTORPLAN.md`.

**Do not import from or depend on anything here.** These files are preserved for reference only.
Prior results are treated as untrusted inputs; any hypothesis derived from them must be re-tested
under the tiered experiment framework.

## Contents

| Directory | Description |
|---|---|
| `data_analysis/` | Phase-0 exploratory analysis scripts and outputs |
| `data_processing/` | Canonical table builder (`build_canonical_v0.py`) |
| `embeddings/` | Embedding manifest script |
| `evaluation/` | Null-baseline scripts and outputs |
| `figures/` | Phase-0 figures |
| `modeling/` | Legacy training harness (train.py, fast_data.py, model.py, etc.) |
| `splits/` | Legacy split protocol builders and JSON definitions |
| `docs/` | Legacy design docs (PROJECT_RESTART_PLAN.md, PreExperimentsPresentation.md) |
| `conditional_gene_essentiality.egg-info/` | Old setuptools build artifact |
| `repo_paths.py` | Old centralized path module |
