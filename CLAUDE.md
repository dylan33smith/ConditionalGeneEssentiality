# Project: ConditionalGeneEssentiality

Predicting **conditional gene essentiality** from Tn-seq fitness data using frozen
protein-language-model gene embeddings + media-chemistry features. The active
objective is **within-organism top-k ranking** of a gene's conditions ("find the
top stressors for this gene").

This file is the **contract**: mission, environment, hard rules, conventions.
It carries **no findings, no results, no numbers** — those live in `docs/`.

---

## The six-file documentation system

| file | when it is read | what it holds |
|---|---|---|
| `CLAUDE.md` | auto-loaded every session | this contract |
| `docs/plan.md` | read at session start | current state, WIP, the ledger |
| `docs/terms.md` | searched before naming anything | glossary, provenance, status |
| `docs/data.md` | read before touching data/runs/paths | registry of every path and its state |
| `docs/memory.md` | **NEVER read whole — grep it** | permanent chronological ledger |
| `docs/bugs.md` | grep by symptom | `[Symptom]` → `[Proven fix]` |

Do not create documentation files outside this set. Anything that is not one of
these is code, config, data, an artifact, `paper/`, or `archive_docs/`.

---

## Build / test / run

```bash
# run a ranking experiment (Hydra entrypoint; handler chosen by stage_or_tier)
python -m src.cli.run_experiment +experiment=R-LOSS_loss_family
python -m src.cli.run_experiment +experiment=R1-A_chemistry
python -m src.cli.run_experiment +experiment=R-CONF_confidence_strat

# REGRESSION CHECK — the locked-best model vs the chem-kNN baseline in one
# pinned command. Run after ANY change to ranking behavior; it gates against
# data_contract/ranking/reval_baseline.json.
python -m src.cli.run_experiment +experiment=R-EVAL_regression     # fast gate
# full replicate-org headline: see the exact command in docs/data.md

# tests (must stay green before any commit)
python -m pytest tests/
python -m pytest tests/test_docs_contract.py    # the docs contract — see below
```

Registered CLI handlers: `R0`, `R1`, `R-LOSS`, `R-CONF`, `R-COLD`, `R-AUG`,
`R-EVAL`. Legacy T-regime handlers were pruned.

---

## Where things live

```
src/ranking/            the ranking objective (self-contained reusable core)
  data/                 fingerprints (+ data layer)
  models.py             AdapterResidualMLP + ResidualBlock (the locked model)
  losses/               pointwise (mse/huber) + ranking (ranknet/lambdarank/listmle/approxndcg)
  eval/                 harness.py (canonical metrics + baselines + stats) + contract.py
  pipeline.py           prepare_r1_data (split+eligibility+features), train_r1_arm
  train.py              loss-family trainers
  runner.py             shared train->eval->report runner: ArmSpec + run_experiment
src/experiments/<R*>/run.py   thin CLI handlers that declare arms and call the runner
src/data/               data ingestion / preprocessing / encoders + ranking data modules
src/cli/run_experiment.py     Hydra entrypoint + handler dispatch
configs/experiment/     one yaml per experiment
data_contract/          frozen handoff artifacts + schemas + the ranking metric contract
docs/                   the five working docs (see table above)
archive_docs/           superseded documentation, unchanged, read-only
paper/                  manuscript drafts and the scientific narrative
tests/                  unit + integration + the docs contract test
```

---

## Conventions (hard rules)

- **Train-only preprocessing.** Vocab/scalers/eligibility thresholds fit on train
  rows only; val unseen categories -> explicit `<UNK>`; log unknown-category rate.
- **Denominator parity.** The model and every baseline are scored on the identical
  eligible val gene set. A ceiling or floor quoted against a method must come from
  the same gene set and the same aggregator.
- **The gate.** A learned model must beat the chem-kNN baseline by the promotion
  delta recorded in `docs/terms.md`, with disjoint hierarchical-bootstrap CIs.
- **Primary metric.** `ndcg_at_5` outranks `within_gene_spearman_mean` everywhere:
  report it first, gate on it, lead CIs and verdicts with it.
- **Regression discipline.** After any change to ranking behavior run `R-EVAL`; if
  a number moves beyond tolerance, stop and investigate — do not absorb it.
- **Tests green** before any commit, including the docs contract test.

## Naming

- **Work items:** `<phase>-<KIND>-<slug>`, `KIND` in `DAT TRN EVL ANL FIX LCK`,
  slug lowercase-hyphenated and meaningful (`azole`, never `A0`). Current phase
  letter: `P`. Example: `P-EVL-leave-compound-out`.
- **Historical IDs are frozen** (`S0`-`S5`, `T1`-`T6`, `R*`, `*-DEC-NNN`, `H-*`).
  They are referenced by `docs/memory.md` and must never be rewritten. The
  old-to-new bridge table is in `docs/terms.md`.
- **Directories/files:** `<phase>_<TARGET>[_<variant>]`. Never two names differing
  only in case. Never a loose file at a results root — everything belongs to an
  owning directory. Put the varying parameter IN the filename. No brace/glob
  shorthand in docs (write `run_a/`, `run_b/`). Deprecated things are renamed
  `DEPRECATED_*` or deleted, and the choice is recorded in `docs/data.md`.

---

## THE IN-PLACE CORRECTION RULE

`docs/memory.md` is permanent. **Never delete or overwrite a historical entry.**
When something in it is proven wrong:

1. Find the exact original line.
2. Prepend `[INCORRECT] - `, preserving the original text verbatim.
3. Insert directly below it: `[CORRECTION - YYYY-MM-DD]: ` with the new finding
   and what changed.

Preserving the wrong version is what makes the reasoning legible later. Grep
`[INCORRECT]` to list everything the project has been wrong about.

**Corrections do not propagate themselves.** After writing one, hunt its stale
copies — grep the distinctive *number or phrase*, not the topic, across `docs/`,
`src/`, `scripts/`, `CLAUDE.md`, `paper/`, and the assistant memory directory.
Anything published outside the repo has no verifier; republishing is part of the
Wrap-Up Protocol, not a courtesy.

---

## THE WRAP-UP PROTOCOL — run when any unit of work completes

Not at the end of the project — at the end of each completed piece of work.

1. **Archive to `docs/memory.md`.** Hypothesis/goal, method, provenance, result,
   under today's date, newest-first at the top.
2. **Compress to the ledger.** One row in `docs/plan.md`.
3. **Document fixes.** `[Symptom]` → `[Proven fix]` into `docs/bugs.md`.
4. **Define new terms.** Full six-field entry in `docs/terms.md`.
5. **Register artifacts.** Every new output/dataset/checkpoint/directory gets a
   `docs/data.md` row.
6. **Reset the board.** Rewrite `docs/plan.md` Current State, add the ledger row,
   bump `Last updated`.
7. **Verify.** `python -m pytest tests/test_docs_contract.py` must pass before the
   session ends.
