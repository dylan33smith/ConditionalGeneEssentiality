# Architecture — Conditional Gene Essentiality (ranking)

Components, data flow, and the key design decisions (with rationale) behind the
within-organism conditional-ranking objective. Update this file only when the
component structure or data flow actually changes.

## 1. The task

Within a known organism, rank a gene's conditional essentiality across **novel
conditions** — "find the top stressors for this gene" — better than a chemistry-
similarity baseline that uses condition features but learns no gene-specific
interaction. Relevance for retrieval = `max(0, −fit)` (a stressor reduces fitness).

**Structural fact that shapes everything:** under condition-holdout the held-out
conditions are 100% disjoint from train (cold columns). So this is **inductive
(cold-start) matrix completion with chemistry as condition side-features** — not
warm-column collaborative filtering. Vanilla MF cannot be applied to the primary
split; chemistry features are what place a never-seen condition.

## 2. Data flow

```
data/raw/feba.db ──(provenance: src/data ingestion/preprocessing/encoders)──▶
  data/derived/canonical/v0/fitness_experiment_long.parquet   (fit, t, condition fields)
  data/processed/ProtLM_embeddings_layer8/*.pt                (frozen ProteomeLM-L8, 1152-d)
  data_contract/preprocessing/de21504134c84a6c/               (S4 feature contract: 425-d multihot)

prepare_r1_data(orgs, split_seed)                              [src/ranking/pipeline.py]
  ├─ condition_key canonicalization + load_fitness            [src/experiments/r0/analyses.py]
  ├─ R-LOCK-2 within-org condition-holdout split (frac 0.20)  [src/data/datasets/build_ranking_split.py]
  ├─ R-LOCK-1 eligibility: per-gene w_g (train) + val filter  [src/data/datasets/ranking_eligibility.py]
  ├─ chemistry features (multihot / fingerprints)             [src/data/datasets/condition_chemistry.py,
  │                                                            src/ranking/data/fingerprints.py]
  ├─ frozen gene embeddings                                   [src/data/datasets/build_s5_dataset.py]
  └─ linear inductive-MF baseline (cached, model-independent) [src/ranking/eval/harness.py]
        │
        ▼
train (per arm × seed)                                        [src/ranking/train.py, pipeline.train_r1_arm]
  AdapterResidualMLP (adapter over frozen emb ⊕ chemistry)    [src/ranking/models.py]
  pointwise (mse/huber) row-batched  |  ranking losses gene-batched   [src/ranking/losses/]
        │
        ▼
evaluate (denominator parity)                                [src/ranking/eval/harness.py]
  within-gene Spearman/Kendall + NDCG@1/3/5 + precision@5
  baselines: chem-kNN (gate) · chem-NULL · linear-MF · replicate noise floor
        │
        ▼
standardized report (CSV + side-by-side vs gate)             [src/ranking/runner.py]
```

The **runner** (`src/ranking/runner.py`) ties train→eval→report into one
declarative call (`ArmSpec` + `run_experiment`); the CLI handlers in
`src/experiments/<R*>/run.py` are thin specs over it. `R-EVAL` is the runner on
the locked-best arm, doubling as the regression gate.

## 3. Locked design decisions (and why)

| Decision | Choice | Rationale | Source |
|---|---|---|---|
| Split | within-org condition-holdout, frac 0.20, seed 0, replicate-grouped, expGroup-stratified | the realistic "novel conditions, known organism" setting; cold columns ⇒ needs side-features | R-LOCK-2 |
| Eligibility | rank only genes with spread `tail_g = p95−p5` over a per-org threshold; train weight `w_g`; hard val filter | ranking flat/near-constant genes is meaningless; weight by discriminability | R-LOCK-1 |
| Metrics | within-gene Spearman + NDCG@5 (k=5), hierarchical org→gene bootstrap, BH-FDR | NDCG matches "top stressors"; clustered CI is honest | R-LOCK-4 |
| Gate | chem-kNN; promote at ΔNDCG@5 ≳ 0.026 + disjoint CIs | a learned model must beat gene-specific lookup to add value | R1-DEC-001 |
| Model | AdapterResidualMLP (learnable adapter over frozen ProteomeLM-L8 ⊕ chemistry) | first real gain over frozen-only; capacity beyond this doesn't help | T5/T3 (archived) |
| Encoder | 425-d multihot chemistry | fingerprints did not beat it | R1-DEC-001 |
| Loss | pointwise Huber (carried-forward base) | robust to fit outliers; ranking losses didn't beat the gate | R-LOSS-DEC-001 |

Full rationale lives in `research_log/decisions/**` and the narrative in
`research_log/SCIENTIFIC_SYNTHESIS.md` (the canonical "what we learned" doc).

## 4. The central finding (context for next work)

The within-org task is **memorization-dominated**: a chemistry-similarity kNN
(NDCG@5 ~0.485) beats every global parametric model tried — across encoders
(R1), objectives (R-LOSS), capacity (linear-MF ≈ deep), and both static and
learned model+kNN hybrids (R-HYBRID-A/B). The signal is local and gene-
idiosyncratic; a global model averages it away. The negative is **noise-robust**
(R-CONF: kNN wins at every measurement-confidence stratum). The achievable
ceiling is modest (replicate agreement NDCG@5 ~0.66). The reusable substrate for
beating this — the loss family (incl. NDCG-direct losses), the `RankingBatch`
sampler contract, and the eval/baseline harness — is retained in `src/ranking`.

## 5. Regression gate

`data_contract/ranking/reval_baseline.json` stores the locked numbers; `R-EVAL`
recomputes and compares within tolerance (0.003). Training is deterministic on a
fixed device, so a behavior-preserving change reproduces bit-exactly. The gate is
how we keep cleanup/refactors honest: a moved number stops the work.
