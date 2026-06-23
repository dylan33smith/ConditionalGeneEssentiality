"""R-AUG handler — does training on MORE organisms help the global model?

Trains the locked arm (pointwise_huber + multihot_425) twice on the SAME locked
evaluation — val set, eligibility, and chem-kNN gate all fixed to `eval_orgs`:

  base_<n>org : model trained on eval_orgs only       (reproduces the locked baseline)
  aug_<m>org  : model trained on eval_orgs ∪ extra_orgs

Both arms are scored on the identical eligible val genes against the identical
chem-kNN gate (R-LOCK-4) — a clean controlled A/B whose single manipulated
variable is training-organism breadth. chem-kNN is per-organism-local, so its
NDCG@5 is bit-identical across the two arms (asserted in the report); only the
global model can exploit the extra organisms. See the R-AUG decision for the
result and its bearing on the memorization-dominated finding.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.ranking.pipeline import prepare_r1_data, prepare_r_aug_data, EMB_DIR
from src.ranking.runner import ArmSpec, run_arm, standardized_report

log = logging.getLogger(__name__)
OUT = Path("artifacts/runs/raug")
LOCKED_LOSS = "pointwise_huber"          # R-LOSS-DEC-001 carried-forward arm


def _all_embedded_orgs() -> list[str]:
    """The organisms we have ProteomeLM embeddings for (authoritative + instant)."""
    return sorted(p.name[:-len("_proteomelm.pt")]
                  for p in EMB_DIR.glob("*_proteomelm.pt"))


def main(cfg: DictConfig) -> None:
    exp = cfg.get("experiment", {})
    eval_orgs = list(exp["eval_orgs"])
    extra = exp.get("extra_orgs", None)
    if extra is None or (isinstance(extra, str) and extra == "auto"):
        extra_orgs = [o for o in _all_embedded_orgs() if o not in set(eval_orgs)]
    else:
        extra_orgs = list(extra)
    split_seed = int(exp.get("split_seed", 0))
    model_seeds = list(exp.get("model_seeds", exp.get("seeds", [0])))
    epochs = int(exp.get("epochs", 8))
    OUT.mkdir(parents=True, exist_ok=True)

    # determinism aids (match R-EVAL — keep the gate reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log.info("=" * 72)
    log.info("R-AUG — train-org augmentation | eval_orgs=%d extra_orgs=%d seeds=%s epochs=%d",
             len(eval_orgs), len(extra_orgs), model_seeds, epochs)
    log.info("    extra: %s", ", ".join(extra_orgs))
    log.info("=" * 72)

    # Arm A — model trained on eval_orgs only (locked-baseline reproduction)
    base = prepare_r1_data(eval_orgs, seed=split_seed)
    log.info("[base] train=%d val=%d eligible val genes=%d",
             len(base.train), len(base.val), len(base.eligible_val_genes))
    base_res = run_arm(
        ArmSpec(f"base_{len(eval_orgs)}org", loss=LOCKED_LOSS, epochs=epochs),
        base, model_seeds=model_seeds)

    # Arm B — model trained on eval_orgs ∪ extra_orgs (eval IDENTICAL to base)
    aug = prepare_r_aug_data(eval_orgs, extra_orgs, seed=split_seed)
    n_total = len(eval_orgs) + len(extra_orgs)
    log.info("[aug] train=%d (baseline_train=%d) val=%d eligible val genes=%d",
             len(aug.train), len(aug.baseline_train), len(aug.val),
             len(aug.eligible_val_genes))
    aug_res = run_arm(
        ArmSpec(f"aug_{n_total}org", loss=LOCKED_LOSS, epochs=epochs),
        aug, model_seeds=model_seeds)

    standardized_report([base_res, aug_res], out_dir=OUT, tag="raug")

    # explicit A/B + gate-parity assertion
    b, a = base_res["agg"], aug_res["agg"]
    d_ndcg = a["model"]["ndcg_at_5"] - b["model"]["ndcg_at_5"]
    d_spear = a["model"]["spearman"] - b["model"]["spearman"]
    gate_drift = a["chem_knn"]["ndcg_at_5"] - b["chem_knn"]["ndcg_at_5"]
    log.info("-" * 72)
    log.info("R-AUG A/B (seed-mean over %d seed(s)):", len(model_seeds))
    log.info("    base (%d org) : model NDCG@5=%.4f  Spearman=%.4f",
             len(eval_orgs), b["model"]["ndcg_at_5"], b["model"]["spearman"])
    log.info("    aug  (%d org) : model NDCG@5=%.4f  Spearman=%.4f",
             n_total, a["model"]["ndcg_at_5"], a["model"]["spearman"])
    log.info("    Δ(aug-base)   : NDCG@5=%+.4f  Spearman=%+.4f", d_ndcg, d_spear)
    log.info("    chem-kNN gate : base=%.4f  aug=%.4f  drift=%+.6f (must be ~0)",
             b["chem_knn"]["ndcg_at_5"], a["chem_knn"]["ndcg_at_5"], gate_drift)
    log.info("    model below gate by: base %.4f  ->  aug %.4f",
             b["chem_knn"]["ndcg_at_5"] - b["model"]["ndcg_at_5"],
             a["chem_knn"]["ndcg_at_5"] - a["model"]["ndcg_at_5"])
    if abs(gate_drift) > 1e-4:
        log.error("    [parity] chem-kNN gate drifted %.6f between arms — the eval is "
                  "NOT identical; investigate before trusting the A/B.", gate_drift)
    log.info("R-AUG done — see %s/raug_metrics.csv", OUT)
