"""src.ranking.eval — the ranking evaluation layer (R-LOCK-4 metric contract).

Single import surface for ranking metrics, baselines, statistics, and the
promotion-gate helpers. Two modules:

  harness.py  — the CANONICAL, live evaluation path (produced the published
                numbers): retrieval metrics (NDCG@k / precision@k), within-gene
                Spearman/Kendall, hierarchical bootstrap + FDR, and the
                split-specific baselines (chem-kNN, chem-NULL, inductive-MF,
                retrieval features), plus per-organism breakdown.
  contract.py — the R-LOCK-4 promotion-gate helpers: BootstrapMetric, the
                per-condition-mean baseline (H-RANK-01), the Spearman
                task-relevant noise floor, and model_beats_baseline. Used for
                go/no-go decisions; complements (does not replace) the harness.

Import everything from here: `from src.ranking.eval import chemistry_knn_predict`.
"""
from src.ranking.eval.harness import (
    ndcg_at_k,
    precision_at_k,
    within_gene_retrieval,
    retrieval_noise_floor,
    paired_hierarchical_bootstrap_ci,
    nearest_train_condition_distance,
    similarity_stratified_report,
    per_gene_correlations,
    hierarchical_bootstrap_ci,
    benjamini_hochberg,
    bootstrap_pvalue_delta,
    chemistry_nearest_condition_profile,
    chemistry_knn_predict,
    chemistry_retrieval_features,
    inductive_mf_predict,
    per_organism_breakdown,
)
from src.ranking.eval.contract import (
    BootstrapMetric,
    within_gene_rank_metric,
    per_condition_train_mean_predictions,
    ranking_baseline_metrics,
    model_beats_baseline,
    task_relevant_noise_floor,
    cross_gene_within_condition_noise_proxy,
)

__all__ = [
    "ndcg_at_k", "precision_at_k", "within_gene_retrieval", "retrieval_noise_floor",
    "paired_hierarchical_bootstrap_ci",
    "nearest_train_condition_distance", "similarity_stratified_report",
    "per_gene_correlations", "hierarchical_bootstrap_ci", "benjamini_hochberg",
    "bootstrap_pvalue_delta", "chemistry_nearest_condition_profile",
    "chemistry_knn_predict", "chemistry_retrieval_features", "inductive_mf_predict",
    "per_organism_breakdown", "BootstrapMetric", "within_gene_rank_metric",
    "per_condition_train_mean_predictions", "ranking_baseline_metrics",
    "model_beats_baseline", "task_relevant_noise_floor",
    "cross_gene_within_condition_noise_proxy",
]
