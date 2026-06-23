## Decision: R-TOPK-DEC-001

### Header
- decision_id: R-TOPK-DEC-001
- stage_or_tier: R-LOSS
- regime: R
- date: 2026-06-17
- owner: project lead
- status: approved
- approved_date: 2026-06-17
- related_experiments: [R-TOPK_loss]
- related_hypotheses: [H-R-TOPK-01]
- supersedes: none (focused follow-up to R-LOSS-DEC-001)

### Assumption Under Test
- assumption_statement: H-R-TOPK-01 — R-LOSS rejected the ranking losses, but its
  ApproxNDCG/LambdaRank optimized NDCG over the WHOLE list. Now that training runs
  through the RankingBatch listwise samplers (the proper [B,L]+mask substrate), a
  loss that truncates the NDCG gain to the TOP-k (k=5) — matching the headline
  metric exactly — closes the gap to the chem-kNN gate (NDCG@5 0.485).
- assumption_type: optimization
- why_it_matters: NDCG@5 is the promotion metric; if any objective could win, it
  would be one that optimizes precisely that quantity. This is the last
  objective-axis lever before concluding the objective is not the bottleneck.

### Pre-Registered Test Plan
- comparison: 5 arms × 3 seeds, 23 replicate orgs, 15 epochs, multihot_425 + T5-A
  held constant. Arms: pointwise_huber (carried-forward base), lambdarank,
  approxndcg, and the NEW list-truncated variants lambdarank_top5 (ΔNDCG truncated
  to top-5) + approxndcg_top5 (NDCG@5 surrogate with a top-k gate).
- metrics_primary: [ndcg_at_5, within_gene_spearman]
- promotion_rule: beat the chem-kNN gate on NDCG@5 (and not tank Spearman).

### Evidence Summary

**Per-arm (seed-mean, 3 seeds, 23 orgs, common eligible val gene set):**

| arm | Spearman | NDCG@5 | prec@5 | vs gate |
|---|---|---|---|---|
| **pointwise_huber** (base) | **0.1522** | **0.4319** | **0.4014** | −0.0533 |
| lambdarank | 0.0819 | 0.4289 | 0.3874 | −0.0563 |
| lambdarank_top5 (NEW) | 0.0726 | 0.4239 | 0.3828 | −0.0613 |
| approxndcg_top5 (NEW) | 0.0157 | 0.3695 | 0.3419 | −0.1157 |
| approxndcg | 0.0106 | 0.3620 | 0.3300 | −0.1232 |
| **chem-kNN (GATE)** | 0.2402 | **0.4852** | 0.446 | — |

**Findings:**
1. **No loss beats the gate.** Best learned arm remains pointwise_huber (0.4319),
   still −0.0533 below chem-kNN. H-R-TOPK-01 **rejected**.
2. **Top-k truncation did not help — it slightly HURT.** Each truncated variant
   scored BELOW its untruncated counterpart (lambdarank_top5 0.4239 < lambdarank
   0.4289; approxndcg_top5 0.3695 vs approxndcg 0.3620 is marginal and both far
   below base). Restricting the gradient to the top-5 removes the broader ordering
   signal without adding any top-end skill the local baseline doesn't already have.
3. **Consistent with R-LOSS-DEC-001 and the memorization-dominated finding:** the
   objective is not the bottleneck. A pointwise robust regression remains the best
   global objective; the gap to chem-kNN is structural (local vs global), not a
   matter of which ranking surrogate is optimized.

### Implementation note (bug found + fixed)
- approxndcg_top5 initially FROZE in training (val NDCG@5 flat ≈ 0.354, Spearman
  ≈ 0) because the top-k gate reused the score-softmax temperature (0.5), making
  the gate too sharp → vanishing gradient. **Fix:** a separate `gate_temp=2.0` in
  `approxndcg_topk` (`src/ranking/losses/family.py`). After the fix every arm
  trains and is evaluated fairly. Unit test `test_topk_truncation_engages_on_long_lists`
  guards that the top-k variants are finite, differ from their untruncated form,
  and prefer a stressor-first ordering.

### Decision
- decision_outcome: **reject H-R-TOPK-01.** Keep pointwise_huber as the locked
  objective. The top-k loss variants are retained in the loss registry (tested,
  cheap to keep) but are NOT promoted. The objective axis is now closed.
- reproduce: `python -m src.cli.run_experiment +experiment=R-TOPK_loss`
  (artifacts in `artifacts/runs/rloss/`, log `topk_full.log`).
