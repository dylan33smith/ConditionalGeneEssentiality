# ConditionalGeneEssentiality

Predicting **conditional gene essentiality** from Tn-seq fitness data using frozen
protein-language-model gene embeddings + media-chemistry features. The active
objective is **within-organism top-k ranking** of a gene's conditions — *"find the
top stressors for this gene."*

This file is a pointer, not a source of truth. Documentation is six files, split
by how often each is read.

| you want | read |
|---|---|
| the contract: mission, environment, hard rules, conventions | [CLAUDE.md](CLAUDE.md) |
| **where things stand right now** — state, work in progress, the ledger | [docs/plan.md](docs/plan.md) |
| what a metric or term means, and what changes its meaning | [docs/terms.md](docs/terms.md) |
| every dataset, split, run directory and artifact, and its state | [docs/data.md](docs/data.md) |
| the permanent record of results and decisions — **grep it, don't read it** | [docs/memory.md](docs/memory.md) |
| a symptom you are debugging | [docs/bugs.md](docs/bugs.md) |
| the manuscript and the scientific narrative | [paper/](paper/) |
| superseded documentation, kept unchanged | [archive_docs/](archive_docs/) |

```bash
python -m pytest tests/                        # all tests
python -m pytest tests/test_docs_contract.py   # the documentation contract
python -m src.cli.run_experiment +experiment=R-EVAL_regression   # the regression gate
```
