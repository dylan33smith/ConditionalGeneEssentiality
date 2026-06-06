## Decision: R-HYBRID-DEC-001

### Header
- decision_id: R-HYBRID-DEC-001
- stage_or_tier: R-HYBRID
- regime: R
- date: 2026-06-06
- owner: project lead
- status: proposed
- related_experiments: [R-HYBRID-A ensemble α-curve]
- related_hypotheses: [H-R-HYBRID-01]

### Assumption Under Test
- assumption_statement: H-R-HYBRID-01 — the global parametric model carries
  signal COMPLEMENTARY to the local chem-kNN; combining them beats kNN alone
  (which R1+R-LOSS showed no global model beats by itself).
- assumption_type: architecture/fusion
- why_it_matters: First test of whether learned structure adds ANYTHING on top
  of the local lookup. Determines whether the project has a model contribution
  (path: R-HYBRID) or is purely memorization (path: characterization paper).

### Method
R-HYBRID-A: per-gene z-scored convex combination of the standalone model
(Huber, the R-LOSS winner) and chem-kNN; sweep α ∈ [0,1]. α=1 = pure kNN.
Honest α-selection: split val GENES 50/50 by hash, pick α* maximizing NDCG@5 on
the tune half, report on the test half (removes val-tuning leakage).

### Evidence (full val, 23 replicate orgs, 53,189 eligible val genes)

| α | knn+model NDCG@5 | knn+model Spearman |
|---|---|---|
| 0.0 (pure model) | 0.432 | 0.156 |
| 0.5 | 0.483 | 0.232 |
| 0.7 | 0.493 | 0.248 |
| **0.8 (peak)** | **0.493** | **0.248** |
| 0.9 | 0.492 | 0.245 |
| 1.0 (pure kNN) | 0.485 | 0.240 |

- **HONEST held-out selection (α*=0.8 on tune-half → test-half): hybrid NDCG@5
  0.494 vs kNN 0.486 = +0.008.** Holds out-of-sample (not curve-peeking).
- The curve peaks at α≈0.8 (mostly kNN + ~20% model), NOT at pure kNN — the
  model adds complementary signal. Spearman also improves (0.240 → 0.248).
- knn+linear-MF combo also beats kNN but by less than knn+model — the DEEP
  model's complement is larger than linear-MF's.

### Decision
- decision_outcome: **complementary signal CONFIRMED, but below the promotion
  delta.** The hybrid beats the chem-kNN gate directionally and out-of-sample
  (+0.008 NDCG@5, +0.008 Spearman), but the gain is BELOW the pre-registered
  promotion delta (≈0.023–0.026). So: H-R-HYBRID-01 directionally supported;
  NOT a promotion by the locked bar.
- rationale: This is the first positive signal that the learned model is not
  useless — it captures cross-gene / sparse-neighborhood structure the local
  lookup misses. But a crude z-score ensemble is the WEAKEST possible fusion;
  the +0.008 is a LOWER BOUND on what hybridization can achieve.
- next_action: **R-HYBRID-B** — a learned hybrid that should extract more of the
  complementary signal than the static ensemble:
  (a) residual model (train the global model on kNN leave-one-out residuals),
  (b) retrieval-augmented model (feed the model gene g's k nearest train-condition
      fits as input), or
  (c) learned per-example gating α(g,c) (trust kNN more where its neighborhood
      is dense, the model more where sparse).
  If R-HYBRID-B clears the promotion delta → real model contribution. If it also
  stalls near +0.01 → the complementary signal is genuinely small, and the
  characterization result is the honest scope.

### Reproducibility
- code: src/experiments/rhybrid/_rhybrid_common.py
- artifact: artifacts/runs/rhybrid_alpha_curve.csv
- standalone model: Huber, row-batched (R-LOSS winner), multihot.
