"""R-COLD handler — the cold-gene (inductive-over-genes) diagnostic.

The primary split is transductive over genes: every val gene also has TRAIN rows,
so chem-kNN can retrieve a gene's OWN history and reliably beats the global model
(the standing R1 gate). The cold_gene split instead holds out WHOLE genes per org —
val genes have ZERO training rows. That breaks the per-gene retrieval baselines by
construction:

  - chem-kNN     : predicts a gene's held-out conditions from that gene's own TRAIN
                   conditions → no train rows → NaN (coverage ~0).
  - inductive-MF : needs a learned free per-gene latent U[g] → unlearnable for an
                   unseen gene → skipped.
  - chem-NULL    : population condition-profile (train-gene-mean fit at the nearest
                   train condition, gene-identity-free) → STILL APPLIES → the gate.

So this is the one regime a frozen-embedding global model could win: it can place
an unseen gene via its ProteomeLM embedding, while the retrieval baselines cannot.
The question: does the embedding carry gene-specific conditional-response signal
beyond the population average (chem-NULL)?

Single locked arm (pointwise_huber + multihot_425), scored against the chem-NULL
gate on the identical eligible cold val genes (denominator parity). chem-kNN
coverage on the eligible cold val genes is logged explicitly to make its
inapplicability quantitative rather than implied.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.ranking.pipeline import prepare_cold_gene_data
from src.ranking.eval import chemistry_knn_predict
from src.ranking.runner import ArmSpec, run_arm, standardized_report, _ci_disjoint

log = logging.getLogger(__name__)
OUT = Path("artifacts/runs/rcold")
LOCKED_LOSS = "pointwise_huber"          # R-LOSS-DEC-001 carried-forward arm
GATE = "chem_null"                       # chem-kNN is structurally inapplicable here


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    orgs = exp.get("orgs", None)
    orgs = list(orgs) if orgs is not None else None
    split_seed = int(exp.get("split_seed", 0))
    model_seeds = list(exp.get("model_seeds", exp.get("seeds", [0])))
    epochs = int(exp.get("epochs", 8))
    tag = str(exp.get("tag", "rcold"))
    OUT.mkdir(parents=True, exist_ok=True)

    # determinism aids (match R-EVAL / R-AUG — keep baselines reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log.info("=" * 72)
    log.info("R-COLD — cold-gene diagnostic | orgs=%s seeds=%s epochs=%d",
             orgs if orgs else "ALL", model_seeds, epochs)
    log.info("=" * 72)

    data = prepare_cold_gene_data(orgs, seed=split_seed)
    n_elig = len(data.eligible_val_genes)
    log.info("[cold_gene] train=%d val=%d eligible cold val genes=%d",
             len(data.train), len(data.val), n_elig)

    # Make chem-kNN's inapplicability quantitative: fraction of eligible cold val
    # rows for which the per-gene retrieval baseline can produce ANY prediction.
    elig_val = data.val[data.val["eligible"]]
    knn = chemistry_knn_predict(data.train, elig_val, data.cond_features, k=5)
    knn_cov = float(knn.notna().mean()) if len(knn) else float("nan")
    log.info("[cold_gene] chem-kNN coverage on eligible cold val rows = %.4f "
             "(≈0 expected — held-out genes have no own-gene train history)", knn_cov)

    res = run_arm(ArmSpec("cold_gene", loss=LOCKED_LOSS, epochs=epochs),
                  data, model_seeds=model_seeds, gate=GATE)
    standardized_report([res], out_dir=OUT, tag=tag, gate=GATE)

    m, g = res["agg"]["model"], res["agg"][GATE]
    d_ndcg = m["ndcg_at_5"] - g["ndcg_at_5"]
    d_spear = m["spearman"] - g["spearman"]
    nd_disjoint = _ci_disjoint(m, g, "ndcg_at_5")
    sp_disjoint = _ci_disjoint(m, g, "spearman")
    log.info("-" * 72)
    log.info("R-COLD result (seed-mean over %d seed(s), n_genes=%d):",
             len(model_seeds), int(m["n_genes"]))
    # NDCG@5 is the PRIMARY metric — report it first, with its CI + disjointness.
    log.info("    NDCG@5 (PRIMARY): model %.4f [%.4f, %.4f]  vs chem-NULL %.4f [%.4f, %.4f]",
             m["ndcg_at_5"], m["ndcg_at_5_ci_low"], m["ndcg_at_5_ci_high"],
             g["ndcg_at_5"], g["ndcg_at_5_ci_low"], g["ndcg_at_5_ci_high"])
    log.info("        Δ=%+.4f  CI-disjoint? %s", d_ndcg,
             "—" if nd_disjoint is None else ("YES" if nd_disjoint else "no (overlap)"))
    log.info("    Spearman (secondary): model %.4f [%.4f, %.4f]  vs chem-NULL %.4f [%.4f, %.4f]",
             m["spearman"], m["spearman_ci_low"], m["spearman_ci_high"],
             g["spearman"], g["spearman_ci_low"], g["spearman_ci_high"])
    log.info("        Δ=%+.4f  CI-disjoint? %s", d_spear,
             "—" if sp_disjoint is None else ("YES" if sp_disjoint else "no (overlap)"))
    verdict = ("model BEATS the population baseline on unseen genes (NDCG@5)"
               if d_ndcg > 0 else "model does NOT beat the population baseline (NDCG@5)")
    log.info("    verdict: %s", verdict)
    log.info("R-COLD done — see %s/%s_metrics.csv", OUT, tag)
