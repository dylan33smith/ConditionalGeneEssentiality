"""R1 — Representation retest under ranking (chemistry: fingerprints vs multihot).

First R-regime MODEL tier. See ARCHITECTURE.md + R-LOCK-1..4. Integrates:
  - R-LOCK-2 condition-holdout split (build_ranking_split)
  - R-LOCK-1 eligibility + w_g weighting (ranking_eligibility)
  - T5-A locked architecture (AdapterResidualMLP) trained with weighted pointwise MSE
  - R-LOCK-4 ranking eval harness (within-gene Spearman/Kendall/NDCG + chem baselines)
"""
