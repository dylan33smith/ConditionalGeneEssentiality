## Decision: R-LOSS-DEC-001

### Header
- decision_id: R-LOSS-DEC-001
- stage_or_tier: R-LOSS
- regime: R
- date: 2026-06-06
- owner: project lead
- status: proposed
- related_experiments: [R-LOSS_loss_family]
- related_hypotheses: [H-R-LOSS-01]

### Assumption Under Test
- assumption_statement: H-R-LOSS-01 — changing the OBJECTIVE from pointwise MSE
  to a ranking loss (pairwise RankNet/LambdaRank, listwise ListMLE/ApproxNDCG),
  including losses that directly optimize NDCG, closes the gap to the chem-kNN
  gate (NDCG@5 0.485). Motivated by R1: encoder doesn't matter and linear-MF ≈
  deep model, so the objective (MSE for a ranking metric) was the suspected lever.
- assumption_type: optimization
- why_it_matters: If the objective is the bottleneck, a ranking loss should beat
  pointwise MSE and approach/exceed the chem-kNN gate. If not, the bottleneck is
  elsewhere (the local vs global nature of the signal).

### Pre-Registered Test Plan
- comparison: 6 losses × 3 seeds, 23 replicate orgs. Pointwise losses trained
  ROW-batched (natural/best form, = R1 control); pairwise/listwise GENE-batched
  (required). Encoder = multihot_425, architecture = T5-A, both held constant.
- metrics_primary: [ndcg_at_5, within_gene_spearman]
- promotion_rule: beat chem-kNN gate on NDCG@5 AND Δ-Spearman ≥ 0.023, BH-FDR
  across losses.

### Evidence Summary

**Per-loss (mean ± sd, 3 seeds, 23 orgs, common eligible val gene set):**

| loss | Spearman | NDCG@1 | NDCG@5 | prec@5 |
|---|---|---|---|---|
| pointwise_mse (control) | 0.130 ± .002 | 0.310 | 0.422 ± .001 | 0.392 |
| **pointwise_huber** | **0.151 ± .005** | **0.320** | **0.435 ± .003** | **0.403** |
| pairwise_ranknet | 0.142 ± .003 | 0.303 | 0.420 ± .003 | 0.394 |
| lambdarank | 0.080 ± .003 | 0.314 | 0.431 ± .005 | 0.389 |
| listmle | 0.128 ± .003 | 0.283 | 0.399 ± .003 | 0.379 |
| approxndcg | 0.011 ± .004 | 0.263 | 0.368 ± .005 | 0.343 |
| **chem-kNN (GATE)** | **0.240** | **0.377** | **0.485** | **0.446** |
| linear-MF | 0.143 | — | 0.429 | — |
| chem-NULL | 0.023 | 0.222 | 0.343 | 0.268 |
| noise-floor (ceiling) | 0.393 | 0.558 | 0.658 | 0.604 |

**Findings:**
1. **NO loss beats the chem-kNN gate (0.485).** Best is pointwise_huber at 0.435
   — still 0.05 below the non-parametric local baseline. H-R-LOSS-01 **rejected**.
2. **The best loss is POINTWISE (Huber), not a ranking loss.** Huber modestly
   beats MSE (ΔNDCG@5 +0.012, ΔSpearman +0.021, consistent across 3 seeds) —
   robustness to fit outliers helps more than switching to a ranking objective.
3. **The NDCG-direct ranking losses did not win.** lambdarank reaches NDCG@5
   0.431 (2nd) but TANKS Spearman to 0.080 — it is purely top-focused and
   sacrifices full-list order. approxndcg is the worst (0.368). listmle (0.399)
   is below MSE.
4. Every global parametric model — MSE, Huber, all ranking losses, AND the
   linear-MF (0.429) — clusters at NDCG@5 ≈ 0.40–0.44, well below chem-kNN.

### Decision
- decision_outcome: **no_winner** for the promotion gate (no loss beats chem-kNN).
  Minor within-tier result: **pointwise_huber > pointwise_mse** (carry Huber
  forward as the default pointwise loss — small but consistent gain). Ranking
  losses (pairwise/listwise) rejected — they do not help and lambdarank/approxndcg
  trade away full-list quality.
- rationale: Combined with R1, two independent levers (encoder, objective) have
  now BOTH failed to move a global parametric model past the chem-kNN gate, and
  capacity already failed (R1: linear-MF ≈ deep model). The consistent picture:
  **global parametric models plateau at NDCG@5 ≈ 0.42–0.44 regardless of encoder,
  objective, or capacity; a LOCAL non-parametric chemistry-kNN (0.485) wins.**
  The bottleneck is the local-vs-global nature of the signal — kNN exploits each
  gene's behavior at chemically-near conditions directly, which a single global
  mapping cannot reproduce.
- risks_remaining:
  - lambdarank's Spearman collapse (0.080) suggests the early-stop-on-NDCG@5
    criterion let it overfit the top at the expense of the list; a Spearman+NDCG
    composite stop might recover some, but it would not close the 0.05 gate gap.
  - Single split seed (R-LOCK-2 seed 0); 3 model seeds only vary init.
- next_action:
  1. **Do NOT pursue NeuralNDCG / differentiable-sorting** (the deferred option).
     lambdarank already directly optimizes NDCG and lost; a fancier NDCG surrogate
     will not beat a 0.05 gap that is about locality, not the loss surrogate.
  2. **Reconsider R2 (fusion).** R1+R-LOSS evidence says capacity/objective aren't
     the lever, so a fusion-topology sweep (also a capacity lever) is LOW
     expected value. Recommend DEMOTING R2.
  3. **Pivot to a LOCAL / HYBRID approach** — the real lever the data points to:
     (a) an explicit hybrid (parametric model + chem-kNN local correction /
     residual), (b) a model with local structure (attention/retrieval over
     chemically-near train conditions), or (c) R-CURRIC-style local refinement.
     This is the recommended next tier (propose **R-HYBRID**).
  4. Carry **pointwise_huber + multihot** forward as the parametric base.

### Reproducibility Attachments
- config_snapshot: configs/experiment/R-LOSS_loss_family.yaml
- artifacts: artifacts/runs/rloss/rloss_results.csv, rloss_metric_comparison.csv
- code_sha: <fill on commit>
- primary_metric_name: ndcg_at_5 + within_gene_spearman
