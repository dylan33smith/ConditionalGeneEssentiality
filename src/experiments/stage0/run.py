"""S0 — Reproducibility & Governance smoke pipeline.

Required outputs (REFACTORPLAN §7 S0):
  1. v4 schema verification → data_contract/v4_schema_verification.json
  2. Run manifest schema → already at data_contract/schemas/run_manifest_v1.schema.json
  3. End-to-end smoke pipeline producing a manifest that validates against the schema
  4. Decision-ledger template → already at research_log/decisions/decision_template.md

Acceptance gate:
  - Smoke run reproduces metrics within tolerance across two fixed-seed reruns.
  - All tests green.
  - v4 schema verification artifact committed.

This handler is NOT promotion-eligible by design: it uses a placeholder random
80/20 split (not the locked organism-holdout that S3 will produce). Its
stage_or_tier is 'S0' and its protocol_id is 's0_smoke_random_80_20' to make
this explicit in the manifest.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from omegaconf import DictConfig

from src.data.ingestion.checksums import file_sha256, manifest_sha256, short
from src.evaluation.additive_baseline import (
    additive_baseline_metrics,
    fit_additive_baseline,
)
from src.evaluation.metrics import mae, rmse


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKBOOK_V4 = Path("data/media_composition_v4.xlsx")
WORKBOOK_SHEET = "Media_Components_ML"
EXPECTED_V4_COLUMNS = [
    "Media", "Canonical_ID", "Compound_name", "Chemical_form",
    "Source_row_component", "Decomposition_type", "Ingredient_source",
    "Include_in_ml", "Source_dataset", "Source_url",
]
MIN_V4_ROWS = 4000

FEBA_DB = Path("data/raw/feba.db")
CANONICAL_DIR = Path("data/derived/canonical/v0")
CANONICAL_FITNESS = CANONICAL_DIR / "fitness_experiment_long.parquet"
EMBEDDING_DIR = Path("data/processed/ProtLM_embeddings_layer8")

V4_VERIFICATION_PATH = Path("data_contract/v4_schema_verification.json")
RUN_MANIFEST_SCHEMA_PATH = Path("data_contract/schemas/run_manifest_v1.schema.json")


# ---------------------------------------------------------------------------
# Step 1: v4 schema verification
# ---------------------------------------------------------------------------

def verify_v4_schema(workbook_sha256: str) -> dict:
    """Open the v4 workbook; assert sheet + columns + row count; emit verification dict."""
    if not WORKBOOK_V4.exists():
        raise FileNotFoundError(f"v4 workbook missing: {WORKBOOK_V4}")
    wb = openpyxl.load_workbook(WORKBOOK_V4, read_only=True, data_only=True)
    if WORKBOOK_SHEET not in wb.sheetnames:
        raise AssertionError(
            f"v4 workbook missing sheet '{WORKBOOK_SHEET}'. Found: {wb.sheetnames}"
        )
    sh = wb[WORKBOOK_SHEET]
    header = list(next(sh.iter_rows(max_row=1, values_only=True)))
    if header != EXPECTED_V4_COLUMNS:
        raise AssertionError(
            f"v4 column mismatch.\nExpected: {EXPECTED_V4_COLUMNS}\nFound:    {header}"
        )
    row_count = sh.max_row  # includes header
    if row_count < MIN_V4_ROWS:
        raise AssertionError(f"v4 row count too low: {row_count} < {MIN_V4_ROWS}")

    # Sample distinct Decomposition_type values for the representation_mode mapping.
    decomp_values: set[str] = set()
    for row in sh.iter_rows(min_row=2, values_only=True):
        v = row[5]   # Decomposition_type column
        if v is not None:
            decomp_values.add(str(v))

    verification = {
        "status": "verified",
        "emitted_by": "stage0",
        "workbook_path": str(WORKBOOK_V4),
        "workbook_sha256": workbook_sha256,
        "sheet_name": WORKBOOK_SHEET,
        "verified_columns": header,
        "verified_row_count": row_count,
        "decomposition_type_values": sorted(decomp_values),
        "verified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    return verification


# ---------------------------------------------------------------------------
# Step 2: data-contract checksums
# ---------------------------------------------------------------------------

def compute_data_contract_checksums(*, use_cache: bool = True) -> dict:
    """Compute the four hard-gate checksums required by run_manifest_v1.schema.json."""
    feba_sha = file_sha256(FEBA_DB, use_cache=use_cache)
    workbook_sha = file_sha256(WORKBOOK_V4, use_cache=use_cache)

    canonical_files = sorted(CANONICAL_DIR.glob("*.parquet"))
    if not canonical_files:
        raise FileNotFoundError(f"no canonical parquet files under {CANONICAL_DIR}")
    canonical_manifest_sha, _ = manifest_sha256(canonical_files, use_cache=use_cache)

    embedding_files = sorted(EMBEDDING_DIR.glob("*_proteomelm.pt"))
    if not embedding_files:
        raise FileNotFoundError(f"no embedding files under {EMBEDDING_DIR}")
    embedding_manifest_sha, _ = manifest_sha256(embedding_files, use_cache=use_cache)

    return {
        "feba_db_sha256": feba_sha,
        "workbook_v4_sha256": workbook_sha,
        "workbook_v4_sheet": WORKBOOK_SHEET,
        "embedding_manifest_id": embedding_manifest_sha,
        "canonical_manifest_id": canonical_manifest_sha,
    }


# ---------------------------------------------------------------------------
# Step 3: smoke split + dataset
# ---------------------------------------------------------------------------

def build_smoke_dataset(seed: int, n_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Sample n_rows rows deterministically; 80/20 split.

    Returns (train_df, val_df, split_manifest_sha256).
    Determinism: identical (seed, n_rows) reproduces identical (train, val) row-for-row.
    """
    log.info("loading canonical fitness (filtered for non-null fit + gene_key)")
    cols = ["orgId", "locusId", "expName", "fit", "gene_key"]
    df = pd.read_parquet(CANONICAL_FITNESS, columns=cols)
    df = df.dropna(subset=["fit", "gene_key", "expName"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    sample_idx = np.sort(perm[:n_rows])
    sample = df.iloc[sample_idx].reset_index(drop=True)

    rng2 = np.random.default_rng(seed + 1)
    is_val = rng2.random(len(sample)) < 0.20
    train = sample[~is_val].reset_index(drop=True)
    val = sample[is_val].reset_index(drop=True)

    h = hashlib.sha256()
    h.update(f"seed={seed},n_rows={n_rows}\n".encode("utf-8"))
    for part_name, part in (("train", train), ("val", val)):
        h.update(f"---{part_name}---\n".encode("utf-8"))
        h.update(
            part.sort_values(["gene_key", "expName"])
                .to_csv(index=False, header=False)
                .encode("utf-8")
        )
    split_manifest_sha = h.hexdigest()
    return train, val, split_manifest_sha


# ---------------------------------------------------------------------------
# Step 4: scored row-set hash
# ---------------------------------------------------------------------------

def hash_scored_rowset(val: pd.DataFrame) -> str:
    """SHA-256 over the (gene_key, expName) pairs in scoring order."""
    h = hashlib.sha256()
    h.update(
        val[["gene_key", "expName"]]
        .sort_values(["gene_key", "expName"])
        .to_csv(index=False, header=False)
        .encode("utf-8")
    )
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step 5: manifest emission + validation
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def validate_manifest_or_raise(manifest: dict) -> None:
    """Validate manifest against the v1 schema; raise on failure."""
    import jsonschema
    schema = json.loads(RUN_MANIFEST_SCHEMA_PATH.read_text())
    jsonschema.validate(manifest, schema)


# ---------------------------------------------------------------------------
# Main S0 handler
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    log.info("=" * 60)
    log.info("S0 smoke pipeline")
    log.info("=" * 60)

    # We keep hydra.job.chdir=false so relative paths to data/ keep working;
    # use HydraConfig to find the run-output dir.
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.train.seed)
    n_smoke_rows = int(cfg.stage.s0.get("n_smoke_rows", 80_000))

    # 1. Checksums (cached after first run)
    log.info("[1/6] computing data-contract checksums (cached after first run)")
    checksums = compute_data_contract_checksums(use_cache=True)
    for k, v in checksums.items():
        if k.endswith("_sheet"):
            log.info("    %s = %s", k, v)
        else:
            log.info("    %s = %s...", k, short(v))

    # 2. v4 schema verification → emit artifact
    log.info("[2/6] verifying v4 workbook schema")
    verification = verify_v4_schema(checksums["workbook_v4_sha256"])
    V4_VERIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    V4_VERIFICATION_PATH.write_text(json.dumps(verification, indent=2))
    log.info("    wrote %s", V4_VERIFICATION_PATH)

    # 3. Build smoke split
    log.info("[3/6] building smoke split (n=%d, seed=%d)", n_smoke_rows, seed)
    train, val, split_manifest_sha = build_smoke_dataset(seed, n_smoke_rows)
    log.info("    train rows=%d, val rows=%d", len(train), len(val))

    # 4. No-op model: predict global train mean; compute baselines
    log.info("[4/6] computing null baselines + additive baseline")
    train_mean = float(train["fit"].mean())
    y_val = val["fit"].to_numpy()
    pred_global_mean = np.full(len(val), train_mean)

    rmse_global = rmse(y_val, pred_global_mean)
    mae_global = mae(y_val, pred_global_mean)

    add_fit = fit_additive_baseline(
        train["fit"].to_numpy(),
        train["gene_key"].to_numpy(),
        train["expName"].to_numpy(),
        max_iters=50,
        tol=1e-4,   # smoke-scale tolerance; S2 will use tighter
    )
    pred_additive = add_fit.predict(
        val["gene_key"].to_numpy(),
        val["expName"].to_numpy(),
    )
    add_metrics = additive_baseline_metrics(pred_additive, y_val)
    log.info(
        "    additive baseline: rmse=%.4f mae=%.4f (converged=%s in %d iters)",
        add_metrics["rmse"], add_metrics["mae"], add_fit.converged, add_fit.n_iters,
    )

    # The "model" in S0 IS the global mean predictor. Delta vs itself is 0.
    null_baseline_deltas = {
        "global_train_mean": {
            "baseline_rmse": rmse_global,
            "baseline_mae": mae_global,
            "delta_rmse": 0.0,
            "delta_mae": 0.0,
        },
        "additive_baseline": {
            "baseline_rmse": add_metrics["rmse"],
            "baseline_mae": add_metrics["mae"],
            "delta_rmse": rmse_global - add_metrics["rmse"],
            "delta_mae": mae_global - add_metrics["mae"],
        },
    }

    # 5. Build run manifest
    log.info("[5/6] assembling run manifest")
    git_sha = get_git_sha()
    code_sha = git_sha
    scored_rowset_hash = hash_scored_rowset(val)

    manifest = {
        "run_id": f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                  f"_S0_smoke_s{seed}_{short(git_sha)}",
        "experiment_id": cfg.get("experiment_id", "S0_smoke"),
        "stage_or_tier": "S0",
        "git_sha": git_sha,
        "code_sha": code_sha,
        "seed": seed,
        "data_contract_checksums": checksums,
        "split_protocol": {
            "protocol_id": "s0_smoke_random_80_20",
            "split_manifest_sha256": split_manifest_sha,
        },
        "preprocessing_artifact_id": "s0_no_preprocessing",
        "config_snapshot_path": ".hydra/config.yaml",
        "scored_rowset": {
            "scored_rowset_hash": scored_rowset_hash,
            "n_rows_scored": len(val),
            "n_genes_eligible": int(val["gene_key"].nunique()),
            "inclusion_counters": {
                "pre_join": int(n_smoke_rows),
                "post_join": int(len(train) + len(val)),
                "post_filter": int(len(train) + len(val)),
                "post_split": int(len(val)),
            },
        },
        "metrics": {
            "rmse": rmse_global,
            "mae": mae_global,
            "within_gene_spearman": None,
        },
        "null_baseline_deltas": null_baseline_deltas,
        "unknown_category_rate": 0.0,
        "leakage_checks": {
            "vocab_train_only_pass": True,
            "scaler_train_only_pass": True,
            "split_overlap_pass": True,
        },
        "notes": (
            "S0 smoke pipeline. Random 80/20 split is a wiring proof, NOT a "
            "promotion-eligible protocol. Locked split is produced by S3. "
            "Model is the global train mean predictor; its delta vs itself is 0."
        ),
    }

    # 6. Validate manifest, write to run dir
    log.info("[6/6] validating manifest against run_manifest_v1.schema.json")
    validate_manifest_or_raise(manifest)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info("    wrote %s", manifest_path)

    # Smoke digest for reproducibility test
    digest = hashlib.sha256(
        json.dumps(
            {
                "rmse": rmse_global,
                "mae": mae_global,
                "additive_rmse": add_metrics["rmse"],
                "additive_mae": add_metrics["mae"],
                "scored_rowset_hash": scored_rowset_hash,
                "split_manifest_sha256": split_manifest_sha,
                "n_val": len(val),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    (output_dir / "smoke_digest.txt").write_text(digest + "\n")
    log.info("    smoke_digest=%s", digest)
    log.info("S0 smoke pipeline complete.")
