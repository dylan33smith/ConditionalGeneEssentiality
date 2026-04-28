"""S2 — Evaluation Trustworthiness handler.

For each candidate protocol from S1:
  - Compute the 5 required null baselines.
  - Run the power report (eligibility, bootstrap CI, permutation null).
  - Compute heteroscedastic-noise diagnostics on additive residuals.
Emits:
  - artifacts/baselines/baselines_per_protocol.json
  - data_contract/policy/eval_policy.yaml (locked)
  - research_log/figures/stage2/ (7 figures)

Per REFACTORPLAN §7 S2.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from omegaconf import DictConfig

from src.experiments.stage1 import analyses as A_S1
from src.experiments.stage2 import baselines as B
from src.experiments.stage2 import figures as F


log = logging.getLogger(__name__)

CANDIDATE_PATH = Path("data_contract/splits/candidate_protocols.yaml")
BASELINES_OUT = Path("artifacts/baselines/baselines_per_protocol.json")
EVAL_POLICY_OUT = Path("data_contract/policy/eval_policy.yaml")


def _to_jsonable(obj):
    """Recursively convert numpy / pandas / dataframe / array types to plain Python."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main(cfg: DictConfig) -> None:
    log.info("=" * 60)
    log.info("S2 Evaluation Trustworthiness")
    log.info("=" * 60)

    # ---- Load candidate protocols ----
    log.info("[1/5] loading candidate protocols")
    cands = yaml.safe_load(CANDIDATE_PATH.read_text())
    candidates = cands["candidates"]
    log.info("    %d candidates", len(candidates))

    # ---- Load canonical fitness ----
    log.info("[2/5] loading canonical fitness (need orgId, gene_key, expName, media, fit)")
    fit_df = A_S1.load_fitness(cols=["orgId", "locusId", "expName", "fit",
                                       "media", "gene_key"])
    fit_df = fit_df.dropna(subset=["fit", "gene_key", "expName"])
    log.info("    fit_df rows=%d", len(fit_df))

    # ---- Per-candidate evaluation ----
    log.info("[3/5] running baselines + power on each candidate")
    n_bootstrap = int(cfg.stage.s2.get("bootstrap_samples", 1000))
    n_permutations = int(cfg.stage.s2.get("permutation_samples", 200))

    per_protocol: dict = {}
    for c in candidates:
        result = B.evaluate_candidate(
            fit_df, c,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
        )
        per_protocol[c["protocol_id"]] = result

    # ---- Emit baselines_per_protocol.json (numpy stripped) ----
    log.info("[4/5] writing %s", BASELINES_OUT)
    BASELINES_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v1",
        "emitted_by": "stage2",
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
        "candidates": _to_jsonable(per_protocol),
    }
    BASELINES_OUT.write_text(json.dumps(payload, indent=2))

    # ---- Emit eval_policy.yaml ----
    log.info("    writing %s", EVAL_POLICY_OUT)
    spearman_role_per_protocol = {
        pid: info["power"]["spearman_role"] for pid, info in per_protocol.items()
    }
    v_min_per_protocol = {
        pid: float(info["power"]["v_min"]) for pid, info in per_protocol.items()
    }
    gain_thresholds = {
        pid: info["meaningful_gain_thresholds"] for pid, info in per_protocol.items()
    }
    eval_policy = {
        "status": "locked",
        "emitted_by": "stage2",
        "spearman_eligibility": {
            "m": 5,
            "v_min_method": "cross_gene_iqr_p25",
            "v_min_value_per_protocol": v_min_per_protocol,
            "policy_id": "spearman_eligibility_v1",
        },
        "co_primary_metrics": ["rmse", "mae"],
        "spearman_role_per_protocol": spearman_role_per_protocol,
        "gain_thresholds_per_protocol": gain_thresholds,
        "null_baselines_required": [
            "global_train_mean", "per_condition_mean", "per_organism_mean",
            "additive_baseline", "embedding_nn",
        ],
    }
    EVAL_POLICY_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_POLICY_OUT.write_text(yaml.safe_dump(eval_policy, sort_keys=False,
                                               default_flow_style=False))

    # ---- Generate figures ----
    log.info("[5/5] generating 7 figures")
    F.fig_01_baseline_metrics_per_protocol(per_protocol)
    F.fig_02_difficulty_ladder(per_protocol)
    F.fig_03_spearman_eligibility_at_m(per_protocol)
    F.fig_04_bootstrap_ci_per_protocol(per_protocol)
    F.fig_05_permutation_null_spearman(per_protocol)
    F.fig_06_residual_quantile_profile(per_protocol)
    F.fig_07_per_organism_residual_spread(per_protocol)
    log.info("    7 figures written under research_log/figures/stage2/")

    log.info("S2 complete.")
    log.info("  baselines  → %s", BASELINES_OUT)
    log.info("  eval_policy→ %s", EVAL_POLICY_OUT)
    log.info("  figures    → %s", F.FIG_DIR)
    log.info("  next: write tier report + S2-DEC-001")
