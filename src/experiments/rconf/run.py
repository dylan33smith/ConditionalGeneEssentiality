"""R-CONF handler — confidence-stratified characterization (idea 1).

Trains the locked base twice (w_g baseline + t-confidence-weighted), then runs
the confidence-stratified evaluation and the cell-level high-|t| filter analysis,
emitting tables + a figure for the characterization report.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.ranking.pipeline import prepare_r1_data, _device
from src.experiments.rconf._rconf_common import (
    train_weighted_model, build_eval_frame, assign_gene_strata,
    stratified_metrics, cell_filter_analysis)

log = logging.getLogger(__name__)
OUT = Path("artifacts/runs/rconf")
FIGDIR = Path("research_log/figures/r_conf")


def _figure(strat_base: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(strat_base))
    labels = [f"{s}\n(|t|~{c:.1f})" for s, c in
              zip(strat_base["stratum"], strat_base["gene_conf_median"])]
    for ax, metric, title in (
            (axes[0], "ndcg5", "NDCG@5 by measurement-confidence stratum"),
            (axes[1], "spearman", "within-gene Spearman by confidence stratum")):
        ax.plot(x, strat_base[f"model_{metric}"], "o-", label="deep model")
        ax.plot(x, strat_base[f"knn_{metric}"], "s-", label="chem-kNN (gate)")
        ax.plot(x, strat_base[f"null_{metric}"], "^-", label="chem-NULL")
        if metric == "ndcg5":
            ax.plot(x, strat_base["ceiling_ndcg5"], "D--", color="gray",
                    label="replicate ceiling")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("confidence stratum (low -> high |t|)")
        ax.set_ylabel(metric); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    strat_base.to_csv(path.with_suffix(".csv"), index=False)
    log.info("    figure -> %s", path)


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    seeds = list(exp.get("seeds", [0]))
    orgs = exp.get("orgs", None)
    orgs = list(orgs) if orgs is not None else None
    epochs = int(exp.get("epochs", 8))
    n_strata = int(exp.get("n_strata", 4))
    OUT.mkdir(parents=True, exist_ok=True)
    dev = _device()

    log.info("=" * 64)
    log.info("R-CONF — confidence-stratified characterization | seeds=%s orgs=%s",
             seeds, orgs if orgs else "ALL")
    log.info("=" * 64)

    log.info("[1/5] preparing data (+abs_t confidence carry)")
    data = prepare_r1_data(orgs, seed=seeds[0])
    log.info("    train=%d val=%d eligible val genes=%d",
             len(data.train), len(data.val), len(data.eligible_val_genes))

    strat_all, cell_all, overall = [], [], []
    for seed in seeds:
        log.info("[2/5] seed=%d — training baseline (w_g) + conf-weighted models", seed)
        m_base = train_weighted_model(data, weight_col_extra=None, seed=seed, epochs=epochs)
        m_conf = train_weighted_model(data, weight_col_extra="conf", seed=seed, epochs=epochs)

        log.info("[3/5] seed=%d — building eval frames + strata", seed)
        ev_base = build_eval_frame(m_base, data, dev)
        ev_conf = build_eval_frame(m_conf, data, dev)
        strata = assign_gene_strata(ev_base, n_strata=n_strata)

        log.info("[4/5] seed=%d — stratified metrics (baseline model)", seed)
        sb = stratified_metrics(ev_base, data.val_raw, strata); sb["seed"] = seed
        strat_all.append(sb)
        log.info("    STRATIFIED (seed=%d):\n%s", seed,
                 sb[["stratum", "n_genes", "gene_conf_median", "model_ndcg5",
                     "knn_ndcg5", "null_ndcg5", "ceiling_ndcg5"]].to_string(index=False))

        cf = cell_filter_analysis(ev_base); cf["seed"] = seed
        cell_all.append(cf)
        log.info("    CELL-FILTER (seed=%d):\n%s", seed,
                 cf[["abs_t_threshold", "n_genes", "model_ndcg5",
                     "knn_ndcg5"]].to_string(index=False))

        # overall: baseline-model vs conf-weighted-model (the t-weighting lever)
        from src.ranking.pipeline import _metrics_for_pred
        mb = _metrics_for_pred(ev_base, "model_pred")
        mc = _metrics_for_pred(ev_conf, "model_pred")
        kn = _metrics_for_pred(ev_base, "knn_pred")
        overall.append({"seed": seed,
                        "model_base_ndcg5": mb["ndcg_at_5"], "model_base_spearman": mb["spearman"],
                        "model_conf_ndcg5": mc["ndcg_at_5"], "model_conf_spearman": mc["spearman"],
                        "knn_ndcg5": kn["ndcg_at_5"], "knn_spearman": kn["spearman"],
                        "n_genes": mb["n_genes"]})
        log.info("    OVERALL t-weighting (seed=%d): base NDCG@5=%.4f conf NDCG@5=%.4f "
                 "(Δ=%.4f) | kNN=%.4f", seed, mb["ndcg_at_5"], mc["ndcg_at_5"],
                 mc["ndcg_at_5"] - mb["ndcg_at_5"], kn["ndcg_at_5"])

        pd.concat(strat_all).to_csv(OUT / "stratified.csv", index=False)
        pd.concat(cell_all).to_csv(OUT / "cell_filter.csv", index=False)
        pd.DataFrame(overall).to_csv(OUT / "tweight_overall.csv", index=False)

    log.info("[5/5] aggregating + figure")
    strat_df = pd.concat(strat_all)
    # seed-mean per stratum for the figure
    num = strat_df.select_dtypes(include=[np.number]).columns
    strat_mean = (strat_df.groupby("stratum", observed=True)[list(num)].mean()
                  .reset_index().sort_values("gene_conf_median"))
    _figure(strat_mean, FIGDIR / "01_confidence_stratified.png")
    log.info("R-CONF done. artifacts in %s", OUT)
