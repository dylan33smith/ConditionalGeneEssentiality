"""Run R-HYBRID-B (residual / retrieval / gating) on an org set, write CSVs.

Usage:
    python -m src.experiments.rhybrid.run_rhybrid_b --orgs Keio --seeds 0
    python -m src.experiments.rhybrid.run_rhybrid_b --orgs FULL --seeds 0 1 2

Writes per-(model, seed) rows to artifacts/runs/rhybrid_b/<tag>_results.csv with
the hybrid vs chem-kNN headline metrics + the honest held-out numbers.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.r1._r1_common import prepare_r1_data
from src.experiments.rhybrid import _rhybrid_b as rb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rhybrid_b")

OUT = Path("artifacts/runs/rhybrid_b")


def _full_orgs() -> list[str]:
    p = yaml.safe_load(open("data_contract/ranking/eligibility_policy.yaml"))
    return list(p["r_replicate_org"].keys())


def _flatten(block: dict) -> dict:
    h, kn = block["hybrid"], block["chem_knn"]
    honest = block.get("honest", {})
    row = {
        "model": block["model"], "seed": block["seed"],
        "n_common_genes": block["n_common_genes"],
        "hybrid_spearman": h["spearman"],
        "hybrid_spearman_ci_low": h["spearman_ci_low"],
        "hybrid_spearman_ci_high": h["spearman_ci_high"],
        "hybrid_ndcg1": h["ndcg_at_1"], "hybrid_ndcg3": h["ndcg_at_3"],
        "hybrid_ndcg5": h["ndcg_at_5"], "hybrid_precision5": h["precision_at_5"],
        "knn_spearman": kn["spearman"],
        "knn_spearman_ci_low": kn["spearman_ci_low"],
        "knn_spearman_ci_high": kn["spearman_ci_high"],
        "knn_ndcg1": kn["ndcg_at_1"], "knn_ndcg3": kn["ndcg_at_3"],
        "knn_ndcg5": kn["ndcg_at_5"], "knn_precision5": kn["precision_at_5"],
        "delta_spearman": h["spearman"] - kn["spearman"],
        "delta_ndcg5": h["ndcg_at_5"] - kn["ndcg_at_5"],
        "honest_hybrid_ndcg5": honest.get("hybrid_ndcg5"),
        "honest_knn_ndcg5": honest.get("knn_ndcg5"),
        "honest_delta_ndcg5": (honest.get("hybrid_ndcg5", float("nan"))
                               - honest.get("knn_ndcg5", float("nan"))),
        "honest_n_test_genes": honest.get("n_test_genes"),
        "mean_alpha": block.get("mean_alpha"),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orgs", nargs="+", default=["Keio"],
                    help="org ids, or the literal FULL for all 23 replicate orgs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--models", nargs="+",
                    default=["residual", "retrieval", "gating"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    orgs = _full_orgs() if args.orgs == ["FULL"] else args.orgs
    tag = args.tag or ("full" if args.orgs == ["FULL"] else "_".join(orgs))
    OUT.mkdir(parents=True, exist_ok=True)
    log.info("R-HYBRID-B tag=%s orgs=%d seeds=%s models=%s", tag, len(orgs),
             args.seeds, args.models)

    rows = []
    for seed in args.seeds:
        log.info("==== preparing data (seed=%d) ====", seed)
        data = prepare_r1_data(orgs, seed=seed)
        log.info("    train=%d val=%d eligible_val_genes=%d", len(data.train),
                 len(data.val), len(data.eligible_val_genes))
        for m in args.models:
            log.info("---- model=%s seed=%d ----", m, seed)
            if m == "residual":
                block = rb.run_residual(data, seed=seed, epochs=args.epochs,
                                        k=args.k, n_bootstrap=args.n_bootstrap)
            elif m == "retrieval":
                block = rb.run_retrieval(data, seed=seed, epochs=args.epochs,
                                         k=args.k, n_bootstrap=args.n_bootstrap)
            elif m == "gating":
                block = rb.run_gating(data, seed=seed, epochs_model=args.epochs,
                                      k=args.k, n_bootstrap=args.n_bootstrap)
            else:
                raise ValueError(m)
            row = _flatten(block)
            rows.append(row)
            # write incrementally so a crash mid-run keeps prior results
            pd.DataFrame(rows).to_csv(OUT / f"{tag}_results.csv", index=False)
            log.info("    -> Δspearman=%.4f Δndcg5=%.4f honest Δndcg5=%.4f",
                     row["delta_spearman"], row["delta_ndcg5"],
                     row["honest_delta_ndcg5"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{tag}_results.csv", index=False)
    log.info("wrote %s", OUT / f"{tag}_results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
