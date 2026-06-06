## Decision: R-HYBRID-DEC-002

### Header
- decision_id: R-HYBRID-DEC-002
- stage_or_tier: R-HYBRID
- regime: R
- date: 2026-06-06
- owner: project lead
- status: proposed
- related_experiments: [R-HYBRID-B residual / retrieval-augmented / learned-gating]
- related_hypotheses: [H-R-HYBRID-01]
- supersedes: none (extends R-HYBRID-DEC-001)
- metrics_primary: [within_gene_spearman_mean, ndcg_at_5] (R-LOCK-4 v2)
- baseline_gate: chemistry_knn (must_beat_competitive; R1-DEC-001)
- eligibility_filter_hash: r_lock_1_v1 (data_contract/ranking/eligibility_policy.yaml)

### Assumption Under Test
- assumption_statement: H-R-HYBRID-01 (round 2) — a LEARNED hybrid of the global
  parametric model + the local chem-kNN can extract MORE of the complementary
  signal than R-HYBRID-A's static z-score ensemble (+0.008 NDCG@5), ideally
  clearing the promotion delta (≈0.023 Spearman / ≈0.026 NDCG@5 over chem-kNN).
- assumption_type: architecture/fusion
- why_it_matters: R-HYBRID-A confirmed complementary signal exists but a static
  ensemble is the weakest fusion. If a learned hybrid clears the bar → the
  project has a real model contribution. If learned hybrids also stall near
  +0.01 (or worse) → the complementary signal is genuinely small and the
  characterization result is the honest scope.

### Method
Three learned hybrids, each combining the standalone model (AdapterResidualMLP,
row-batched weighted-Huber — the R-LOSS winner) with the chem-kNN gate
(NDCG@5 0.485 / Spearman 0.24 on full val), all sharing the R-HYBRID-A
evaluation protocol for comparability:

  (a) **Residual model** — `chemistry_knn_predict` gained an `exclude_self` flag
      enabling LEAVE-ONE-OUT kNN on train. Train AdapterResidualMLP to predict
      the residual r = fit − knn_loo (rows with NaN knn_loo dropped). Val pred =
      knn_val + model_residual_val.
  (b) **Retrieval-augmented model** — per (g,c) retrieve gene g's k=5 nearest
      TRAIN conditions by chemistry; build retrieval features [neighbor fits,
      similarities, weighted-mean fit, coverage]; feed concat(adapter(gene_emb),
      chem_c, retrieval_features); train end-to-end (Huber) to predict fit.
      Train rows use leave-one-out retrieval; val rows retrieve from TRAIN only.
  (c) **Learned gating** — α(g,c)=sigmoid(MLP(gate_features)) where gate_features
      = [dist to nearest train condition, neighborhood density, kNN coverage];
      final = α·z(knn) + (1−α)·z(model) per gene (per-gene z-score as in
      R-HYBRID-A). α's MLP trained on TRAIN (LOO kNN) to match per-gene-z-scored
      true fit.

Eval: eligible val gene set, denominator parity (SAME genes as chem-kNN), within
-gene Spearman (hierarchical org→gene bootstrap, n=1000) + NDCG@1/3/5 +
precision@5. HONEST held-out: split val genes 50/50 by hash, report the hybrid
on the test half (the residual/retrieval hybrids have no val-selected scalar, so
honest = test-half eval; gating likewise). Leakage discipline: all train-side
kNN/retrieval is leave-one-out; val-side neighbors come from TRAIN only.

### Evidence (full val, 23 replicate orgs, 3 seeds; ~52.9–53.2k common eligible val genes)

Seed-averaged (denominator parity vs chem-kNN on the SAME genes):

| model | hybrid NDCG@5 | kNN NDCG@5 | **Δ NDCG@5** | hybrid Spearman | kNN Spearman | **Δ Spearman** | hybrid NDCG@1 | hybrid NDCG@3 | hybrid prec@5 |
|---|---|---|---|---|---|---|---|---|---|
| **residual**  | 0.4900 | 0.4899 | **+0.0002** | 0.2136 | 0.2185 | **−0.0049** | 0.3707 | 0.4331 | 0.4595 |
| retrieval     | 0.4840 | 0.4899 | **−0.0059** | 0.2075 | 0.2185 | **−0.0109** | 0.3666 | 0.4269 | 0.4538 |
| gating        | 0.4817 | 0.4898 | **−0.0081** | 0.2003 | 0.2185 | **−0.0181** | 0.3634 | 0.4231 | 0.4553 |

HONEST held-out (test-half of a 50/50 gene split), seed-averaged:

| model | honest hybrid NDCG@5 | honest kNN NDCG@5 | **honest Δ NDCG@5** |
|---|---|---|---|
| residual  | 0.4912 | 0.4903 | **+0.0009** |
| retrieval | 0.4846 | 0.4903 | **−0.0057** |
| gating    | 0.4822 | 0.4903 | **−0.0081** |

Per-seed Δ NDCG@5 (note the residual sign-flips across seeds):

| model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| residual  | +0.0048 | −0.0076 | +0.0033 |
| retrieval | −0.0027 | −0.0093 | −0.0057 |
| gating    | −0.0082 | −0.0123 | −0.0038 |

Reference bars (full val, from R-LOCK-4 / SCIENTIFIC_SYNTHESIS):
- chem-kNN gate: NDCG@5 ≈ 0.485, Spearman ≈ 0.24.
- noise-floor ceiling (replicate agreement): NDCG@5 ≈ 0.658, Spearman ≈ 0.39.
- promotion delta: ≈ 0.026 NDCG@5 / ≈ 0.023 Spearman over the gate.

Dev (Keio single-org) gave residual a misleadingly large honest Δ NDCG@5 = +0.042;
this did NOT survive the full 23-org eval (residual collapses to ≈ 0). The
Keio-only number is a single-org-ceiling artifact — full-val multi-org is the
governing number, consistent with prior project findings about Keio being easy.

### Decision
- decision_outcome: **NO PROMOTION. H-R-HYBRID-01 round 2 NOT supported.** None of
  the three learned hybrids clears the promotion delta on either co-primary
  metric. The best (residual) is a statistical TIE with chem-kNN on NDCG@5
  (+0.0002, sign-flipping across seeds) and a small REGRESSION on Spearman
  (−0.0049). Retrieval and gating both regress on both metrics.
- rationale:
  - The learned hybrids extracted LESS complementary signal than R-HYBRID-A's
    crude static ensemble (+0.008 honest NDCG@5). The static ensemble's gain
    came from a fixed, conservative α≈0.8 (mostly kNN). The learned approaches
    either (residual) add a continuous model term that helps NDCG marginally
    but hurts Spearman, or (retrieval/gating) blend in too much of the weak
    global prior and regress.
  - Gating learned mean_alpha ≈ 0.46 (trusts kNN and model roughly equally),
    but the model half is weak enough that more weight on it HURTS — the data
    says "trust the lookup," and a learned gate that ever down-weights kNN loses.
  - The residual model's per-seed sign-flip (+0.0048 / −0.0076 / +0.0033) shows
    the complementary signal is within split-noise: it is not a stable, real,
    promotable gain. Honest held-out Δ = +0.0009 confirms ≈ zero.
  - This is the convergent, pre-registered outcome anticipated in
    R-HYBRID-DEC-001 / SCIENTIFIC_SYNTHESIS: learned hybrids "stall near +0.01"
    (here they came in BELOW that, ≤ +0.0009 honest). The complementary signal
    is genuinely small. The within-org conditional-essentiality ranking task is
    memorization-dominated; the local chem-kNN lookup is the honest ceiling among
    available methods, and learned global structure adds nothing promotable on
    top of it.
- next_action: **Conclude the modeling thread; pivot to the CHARACTERIZATION
  scope.** Two learned-hybrid rounds (A static, B residual/retrieval/gating) plus
  the full R1 (encoder) / R2-equivalent capacity / R-LOSS (objective) sweeps have
  now all failed to beat the chem-kNN lookup by a promotable margin. The honest,
  publishable result is the rigorous negative: a pre-registered benchmark showing
  the within-org task is memorization-dominated, with a simple chemistry-similarity
  lookup as the strong baseline that learned global models cannot surpass.
  Recommended follow-ups (characterization, not promotion gates):
    1. Run the cold-gene diagnostic split (RPLAN R3+) to quantify how far chem-kNN
       degrades when the gene is unseen — the one regime where a global model could
       in principle help — to bound the value of any future modeling.
    2. Fold these R-HYBRID-B results into SCIENTIFIC_SYNTHESIS as the closing
       modeling evidence.
  Do NOT pursue a fourth hybrid variant: three orthogonal fusions (residual /
  retrieval / gating) converging on ≈ 0 is strong evidence the ceiling is real,
  not a fusion-design artifact.

### Reproducibility
- code: src/experiments/rhybrid/_rhybrid_b.py (three hybrids),
  src/experiments/rhybrid/run_rhybrid_b.py (runner),
  src/evaluation/ranking_eval.py (chemistry_knn_predict exclude_self,
  chemistry_retrieval_features).
- tests: tests/unit/test_rhybrid_b.py (exclude_self LOO correctness, retrieval
  features, each hybrid runs end-to-end on synthetic data). `pytest tests/unit`
  green (156 passed, 1 skipped).
- artifacts: artifacts/runs/rhybrid_b/full_results.csv (per-model/seed metrics),
  artifacts/runs/rhybrid_b/full_run.log.
- config: full = 23 replicate orgs (eligibility_policy.yaml r_replicate_org),
  seeds 0/1/2, 8 epochs, k=5, weighted-Huber, multihot_425 chemistry, n_bootstrap
  1000. Standalone model = AdapterResidualMLP (T5-A locked).
- gate: chem-kNN (k=5, cosine on condition multihot), scored on the SAME eligible
  val gene set as each hybrid (denominator parity).
