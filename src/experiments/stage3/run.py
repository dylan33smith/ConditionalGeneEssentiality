"""S3 — Split Protocol Lock handler.

Select one primary protocol from S1 candidates using S2 evidence, then emit:
  - data_contract/splits/locked_protocol.yaml
  - data_contract/splits/diagnostic_protocols.yaml
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig


log = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _index_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["protocol_id"]: c for c in candidates}


def select_primary_protocol(
    candidates_payload: dict[str, Any],
    baselines_payload: dict[str, Any],
    eval_policy: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically select primary protocol and return score table.

    Rule id: s3_v1_power_driven
    """
    cands = candidates_payload["candidates"]
    by_id = _index_candidates(cands)
    s2 = baselines_payload["candidates"]
    m = int(eval_policy["spearman_eligibility"]["m"])
    val_rows_floor = m * 200

    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for pid, s2_info in s2.items():
        if pid not in by_id:
            continue
        c = by_id[pid]
        n_genes = int(s2_info["power"]["bootstrap_ci"]["n_genes_used"])
        val_rows = int(c["support"]["val_rows"])
        role = str(s2_info["power"]["spearman_role"])
        seen_rate = float(c["chemistry_overlap"]["val_canonical_id_seen_rate"])
        nn_rmse = float(s2_info["baselines"]["embedding_nn"]["rmse"])
        global_rmse = float(s2_info["baselines"]["global_train_mean"]["rmse"])
        nn_beats_global = nn_rmse < global_rmse

        checks = {
            "spearman_primary": role == "primary",
            "n_genes_eligible_ge_200": n_genes >= 200,
            "val_rows_floor_pass": val_rows >= val_rows_floor,
            "canonical_seen_rate_ge_0_90": seen_rate >= 0.90,
        }
        is_eligible = all(checks.values())
        row = {
            "protocol_id": pid,
            "eligible": is_eligible,
            "checks": checks,
            "n_genes_used": n_genes,
            "val_rows": val_rows,
            "val_canonical_seen_rate": seen_rate,
            "nn_beats_global_rmse": nn_beats_global,
            "nn_rmse": nn_rmse,
            "global_rmse": global_rmse,
        }
        rows.append(row)
        if is_eligible:
            eligible.append(row)

    if not eligible:
        raise ValueError("No S3 primary protocol satisfies eligibility criteria.")

    def _rank_key(r: dict[str, Any]) -> tuple[int, int, int]:
        return (1 if r["nn_beats_global_rmse"] else 0, r["n_genes_used"], r["val_rows"])

    winner = max(eligible, key=lambda r: _rank_key(r))
    winner_id = winner["protocol_id"]
    return winner_id, sorted(rows, key=lambda r: r["protocol_id"])


def build_locked_protocol_payload(
    *,
    winner: dict[str, Any],
    selection_table: list[dict[str, Any]],
    candidate_sha256: str,
    baselines_sha256: str,
    eval_policy_sha256: str,
) -> dict[str, Any]:
    seed_set = [0, 1, 2]
    manifest_payload = {
        "protocol_id": winner["protocol_id"],
        "val_org_ids": sorted(winner["val_org_ids"]),
        "test_org_ids": sorted(winner["test_org_ids"]),
        "seed_set": seed_set,
        "candidate_protocols_sha256": candidate_sha256,
    }
    split_manifest_sha256 = _sha256_json_payload(manifest_payload)
    return {
        "status": "locked",
        "emitted_by": "stage3",
        "selection_rule_id": "s3_v1_power_driven",
        "protocol_id": winner["protocol_id"],
        "val_org_ids": list(winner["val_org_ids"]),
        "test_org_ids": list(winner["test_org_ids"]),
        "seed": 0,
        "seed_set": seed_set,
        "candidate_protocols_sha256": candidate_sha256,
        "baselines_sha256": baselines_sha256,
        "eval_policy_sha256": eval_policy_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "selection_rationale": {
            "summary": (
                "Primary selected using s3_v1_power_driven: eligible protocols are "
                "ranked by (NN beats global on RMSE, n_genes_used, val_rows)."
            ),
            "table": selection_table,
        },
    }


def _pick_secondary_stress_protocol(
    candidates: list[dict[str, Any]], primary_id: str
) -> dict[str, Any]:
    non_primary = [c for c in candidates if c["protocol_id"] != primary_id]
    preferred = [c for c in non_primary if c["protocol_id"] == "low_overlap_stress"]
    if preferred:
        chosen = preferred[0]
    else:
        chosen = min(non_primary, key=lambda c: c["chemistry_overlap"]["val_seen_rate"])
    return {
        "id": "secondary_stress",
        "protocol_id": chosen["protocol_id"],
        "role": "reported_not_gating",
        "reason": "Lowest media-level val seen-rate among non-primary candidates.",
    }


def _build_homology_diagnostic(primary_id: str) -> dict[str, Any]:
    return {
        "id": "homology_diagnostic",
        "mode": "primary_bin_stratified_plus_secondary_masked_subset",
        "cosine_threshold": 0.85,
        "bin_edges": [0.0, 0.5, 0.85, 1.0],
        "applies_to_protocol": primary_id,
        "evidence": "research_log/figures/stage1/18_embedding_cosine_to_nearest_train_per_protocol.csv",
        "decision_ref": "research_log/decisions/stage3/S3-DEC-001.md",
    }


def build_diagnostic_protocols_payload(
    *,
    candidates_payload: dict[str, Any],
    primary_id: str,
    enable_homology_diagnostic: bool,
) -> dict[str, Any]:
    protocols = [_pick_secondary_stress_protocol(candidates_payload["candidates"], primary_id)]
    if enable_homology_diagnostic:
        protocols.append(_build_homology_diagnostic(primary_id))
    return {
        "status": "populated",
        "emitted_by": "stage3",
        "protocols": protocols,
    }


def main(cfg: DictConfig) -> None:
    s3_cfg = cfg.stage.s3
    candidate_path = Path(str(s3_cfg.candidate_protocols_path))
    baselines_path = Path(str(s3_cfg.baselines_path))
    eval_policy_path = Path("data_contract/policy/eval_policy.yaml")
    out_locked = Path(str(s3_cfg.output_locked_protocol))
    out_diag = Path(str(s3_cfg.output_diagnostic_protocols))

    log.info("=" * 60)
    log.info("S3 Split Protocol Lock")
    log.info("=" * 60)
    log.info("Loading inputs")

    candidates_payload = yaml.safe_load(candidate_path.read_text())
    baselines_payload = json.loads(baselines_path.read_text())
    eval_policy = yaml.safe_load(eval_policy_path.read_text())

    winner_id, selection_table = select_primary_protocol(
        candidates_payload, baselines_payload, eval_policy
    )
    by_id = _index_candidates(candidates_payload["candidates"])
    winner = by_id[winner_id]

    candidate_sha = _sha256_file(candidate_path)
    baselines_sha = _sha256_file(baselines_path)
    eval_policy_sha = _sha256_file(eval_policy_path)

    locked_payload = build_locked_protocol_payload(
        winner=winner,
        selection_table=selection_table,
        candidate_sha256=candidate_sha,
        baselines_sha256=baselines_sha,
        eval_policy_sha256=eval_policy_sha,
    )

    h_homo_triggered = bool(candidates_payload.get("h_homo_01_triggered", False))
    cfg_enable_homology = bool(s3_cfg.get("enable_homology_diagnostic", False))
    enable_homology = cfg_enable_homology or h_homo_triggered
    diag_payload = build_diagnostic_protocols_payload(
        candidates_payload=candidates_payload,
        primary_id=winner_id,
        enable_homology_diagnostic=enable_homology,
    )

    out_locked.parent.mkdir(parents=True, exist_ok=True)
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    out_locked.write_text(yaml.safe_dump(locked_payload, sort_keys=False, default_flow_style=False))
    out_diag.write_text(yaml.safe_dump(diag_payload, sort_keys=False, default_flow_style=False))

    log.info("Locked protocol: %s", winner_id)
    log.info("  wrote %s", out_locked)
    log.info("  wrote %s", out_diag)
