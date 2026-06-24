# ConditionalGeneEssentiality

Predicting **conditional gene essentiality** from Tn-seq fitness data using frozen
ProteomeLM gene embeddings + media-chemistry features. The active objective is
**within-organism top-k ranking** of a gene's conditions — *"find the top
stressors for this gene."*

This README is the single source of truth for the **current state** of the
project. Deep history, rationale, and AI working-memory live in modular files
(see [Project memory & where to look](#project-memory--where-to-look)).

---

## Current state (June 2026)

- **Branch:** `ranking` is the trunk (clean, post-cleanup). Develop here.
- **Status:** clean modular `src/ranking/` core with a shared runner and a
  bit-exact regression gate. The warm-split exploration is complete (all global-
  model levers lost to chem-kNN); the **cold-gene diagnostic (R-COLD) is the first
  positive** and points the way. **Next up: the inductive cold-start-over-genes
  objective** (see [docs/project_memory/progress.md](docs/project_memory/progress.md)
  and the Next-tasks section there).
- **Headline result (23 replicate orgs, 3 seeds):** a chemistry-similarity kNN
  is the strong baseline; learned global models do not beat it.

NDCG@5 is the **primary** metric (leftmost); within-gene Spearman is secondary.

| method | NDCG@5 (primary) | within-gene Spearman | what it is |
|---|---|---|---|
| chem-NULL | ~0.31 | ~0.02 | population condition profile (no gene specificity) |
| deep model (frozen emb + chem) | ~0.42 | ~0.13 | the intended model (R1) |
| linear-MF (learned latents) | ~0.43 | ~0.14 | a learned, fitness-aware gene rep |
| **chem-kNN — the gate** | **~0.485** | **~0.24** | the gene's OWN history, chemistry lookup |
| replicate ceiling | ~0.66 | ~0.39 | biological-replicate agreement (the achievable max) |

**Central finding:** the within-org task is **memorization-dominated** — the
signal is local and gene-idiosyncratic, so a global model averages it away while
the kNN lookup preserves it. The negative is *noise-robust* (holds at every
measurement-confidence stratum).

**Refinement (2026-06-23, R-COLD-DEC-001):** that gap is specifically a
*memorization* gap, not an "embeddings carry nothing" gap. On the **cold-gene**
split (whole genes held out), chem-kNN is structurally inapplicable — it has no
own-gene history to retrieve (coverage 0%) — so the only baseline left is chem-NULL
(the population profile). There the global model **beats** chem-NULL on genes it
never trained on: NDCG@5 **0.275 vs 0.245** (Δ+0.030), Spearman **0.074 vs 0.036**,
with point estimates disjoint across all 3 seeds (n=11,761). Caveat on strength:
the **primary-metric (NDCG@5) bootstrap CIs OVERLAP** (0.275 [0.242, 0.318] vs
0.245 [0.212, 0.282]) — only the secondary Spearman CI is disjoint — so this is a
directionally robust *lead*, not a CI-confirmed win. Still, the frozen embedding
*does* carry transferable gene-specific signal; it is just outgunned by per-gene
memorization wherever a gene's own history is available. This is not a promotion vs the locked
chem-kNN gate (chem-NULL is a weaker bar); it makes the **inductive
cold-start-over-genes** objective the live lever. Full narrative:
[research_log/SCIENTIFIC_SYNTHESIS.md](research_log/SCIENTIFIC_SYNTHESIS.md).

---

## The task

Within a **known organism**, rank a gene's conditional essentiality across
**novel conditions**, better than a chemistry-similarity baseline that uses
condition features but learns no gene-specific interaction. Retrieval relevance =
`max(0, −fit)` (a stressor reduces fitness).

**Structural fact that shapes everything:** under condition-holdout the held-out
conditions are 100% disjoint from train (*cold columns*). So this is **inductive
(cold-start) matrix completion with chemistry as condition side-features** — not
warm-column collaborative filtering. Vanilla matrix factorization cannot be
applied to the primary split; the chemistry features are what place a never-seen
condition.

---

## Quick start

```bash
# run a ranking experiment (Hydra entrypoint; handler chosen by stage_or_tier)
python -m src.cli.run_experiment +experiment=R-LOSS_loss_family
python -m src.cli.run_experiment +experiment=R1-A_chemistry
python -m src.cli.run_experiment +experiment=R-CONF_confidence_strat

# REGRESSION CHECK — locked-best model vs chem-kNN in one pinned command
# (split seed 0, k=5). Run after ANY change to ranking behavior; gates against
# data_contract/ranking/reval_baseline.json.
python -m src.cli.run_experiment +experiment=R-EVAL_regression                 # fast gate (Keio+Caulo+MR1)
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
(regression check). The legacy T-regime handlers were pruned
(see [docs/PRUNED_INDEX.md](docs/PRUNED_INDEX.md)).

### Adding a new ranking test (declarative — no copy-pasted pipeline)

```python
from src.ranking.runner import ArmSpec, run_experiment
run_experiment(
    [ArmSpec("huber", loss="pointwise_huber"), ArmSpec("lambda", loss="lambdarank")],
    orgs=None, model_seeds=(0, 1, 2), out_dir="artifacts/runs/my_test", tag="my_test")
```
You get the standardized comparison (model + chem-kNN/NULL/linear-MF on the same
eligible val genes, per-seed + seed-mean, tidy CSV + side-by-side vs the gate).

---

## Repository layout

```
src/ranking/            the ranking objective (self-contained reusable core)
  models.py             AdapterResidualMLP + ResidualBlock (the locked model)
  losses/               pointwise (mse/huber) + ranking (ranknet/lambdarank/listmle/approxndcg)
  eval/                 harness.py (canonical metrics + baselines + stats) + contract.py (gate helpers)
  data/                 fingerprints (+ data layer; split/eligibility/chemistry live in src/data/datasets)
  pipeline.py           prepare_r1_data (split + eligibility + features), train_r1_arm, eval harness
  train.py              loss-family trainers (pointwise row-batched / ranking gene-batched)
  runner.py             shared train→eval→report runner: ArmSpec + run_experiment + standardized_report
src/experiments/<R*>/run.py   thin CLI handlers that declare arms and call the runner
src/data/               data ingestion / preprocessing / encoders (provenance) + ranking data modules
src/cli/run_experiment.py     Hydra entrypoint + handler dispatch
configs/experiment/     one yaml per experiment (R0/R1/R-LOSS/R-CONF/R-EVAL)
data_contract/          frozen handoff artifacts + schemas + ranking metric contract + R-EVAL baseline
research_log/           decision ledger, figures, SCIENTIFIC_SYNTHESIS.md (canonical learnings)
docs/                   data manifests + PRUNED_INDEX + project_memory/ (AI working memory)
tests/                  unit + integration (keep green before any commit)
```

---

## Data flow

```
data/raw/feba.db ──(provenance: src/data ingestion/preprocessing/encoders)──▶
  data/derived/canonical/v0/fitness_experiment_long.parquet   (fit, t, condition fields)
  data/processed/ProtLM_embeddings_layer8/*.pt                (frozen ProteomeLM-L8, 1152-d)
  data_contract/preprocessing/de21504134c84a6c/               (S4 feature contract: 425-d multihot)

prepare_r1_data(orgs, split_seed)                              [src/ranking/pipeline.py]
  ├─ condition_key canonicalization + load_fitness            [src/data/datasets/conditions.py]
  ├─ R-LOCK-2 within-org condition-holdout split (frac 0.20)  [src/data/datasets/build_ranking_split.py]
  ├─ R-LOCK-1 eligibility: per-gene w_g (train) + val filter  [src/data/datasets/ranking_eligibility.py]
  ├─ chemistry features (multihot / fingerprints)             [src/data/datasets/condition_chemistry.py,
  │                                                            src/ranking/data/fingerprints.py]
  ├─ frozen gene embeddings                                   [src/data/datasets/build_s5_dataset.py]
  └─ linear inductive-MF baseline (cached)                    [src/ranking/eval/harness.py]
        ▼
train (per arm × seed)  →  AdapterResidualMLP                 [src/ranking/{train,models}.py]
        ▼
evaluate (denominator parity)  →  Spearman/NDCG@k + baselines [src/ranking/eval/harness.py]
        ▼
standardized report (CSV + side-by-side vs gate)             [src/ranking/runner.py]
```

`R-EVAL` is the runner on the locked-best arm, doubling as the regression gate.

---

## Locked design decisions (summary)

| Decision | Choice | Why |
|---|---|---|
| Split | within-org condition-holdout, frac 0.20, seed 0 | the realistic "novel conditions, known organism" setting (cold columns) |
| Eligibility | rank genes with spread `tail_g = p95−p5` over a per-org threshold; train weight `w_g` | ranking flat genes is meaningless; weight by discriminability |
| Metrics | **NDCG@5 (k=5) primary** + within-gene Spearman (secondary); both with hierarchical org→gene bootstrap CI; BH-FDR | NDCG matches "top stressors" so it outranks Spearman everywhere; clustered CI is honest |
| Gate | chem-kNN; promote at ΔNDCG@5 ≳ 0.026 + disjoint CIs | a learned model must beat gene-specific lookup to add value |
| Model | AdapterResidualMLP (adapter over frozen ProteomeLM-L8 ⊕ chemistry) | first real gain over frozen-only; more capacity doesn't help |
| Encoder | 425-d multihot chemistry | fingerprints did not beat it |
| Loss | pointwise Huber (carried-forward base) | robust to fit outliers; ranking losses didn't beat the gate |

Full rationale: [docs/project_memory/decisions.md](docs/project_memory/decisions.md)
and the decision ledger `research_log/decisions/**`.

---

## Conventions (hard rules)

- **Train-only preprocessing.** Vocab/scalers/eligibility thresholds fit on train
  rows only; val unseen categories → explicit `<UNK>`; log unknown-category rate.
- **Denominator parity.** Model and every baseline are scored on the identical
  eligible val gene set.
- **The gate.** chem-kNN (NDCG@5 ~0.485) is the baseline a learned model must
  beat; promotion needs ΔNDCG@5 ≳ 0.026 with disjoint hierarchical-bootstrap CIs.
- **Regression discipline.** After any change to ranking behavior, run `R-EVAL`;
  if a number moves beyond tolerance, **stop and investigate** (don't absorb it).
- **Tests green** before any commit.

### Authoritative data inputs

| Artifact | Path |
|---|---|
| Raw fitness DB | `data/raw/feba.db` |
| Canonical fitness | `data/derived/canonical/v0/fitness_experiment_long.parquet` |
| Gene embeddings | `data/processed/ProtLM_embeddings_layer8/*.pt` |
| Feature contract (S4, frozen) | `data_contract/preprocessing/de21504134c84a6c/` |

> `data/` is a machine-local symlink (gitignored) into the shared data root. It is
> **not** tracked by git — see `docs/project_memory/bugs.md` for the incident that
> established this. The canonical parquet is rebuildable from `feba.db` via
> `archive/data_processing/build_canonical_v0.py` (verified by manifest SHA256).

---

## Project memory & where to look

| Need | File |
|---|---|
| Standing AI instructions + Memory Protocol | [CLAUDE.md](CLAUDE.md) |
| **Exact current state / where we left off** | [docs/project_memory/progress.md](docs/project_memory/progress.md) |
| **Why we chose each architecture/approach** | [docs/project_memory/decisions.md](docs/project_memory/decisions.md) |
| **Quirks, recurring errors, proven fixes** | [docs/project_memory/bugs.md](docs/project_memory/bugs.md) |
| Canonical scientific narrative | [research_log/SCIENTIFIC_SYNTHESIS.md](research_log/SCIENTIFIC_SYNTHESIS.md) |
| Promotion-gate decision ledger | `research_log/decisions/**` |
| What was pruned & where its learning lives | [docs/PRUNED_INDEX.md](docs/PRUNED_INDEX.md) |
