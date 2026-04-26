#!/usr/bin/env python3
"""Training harness: frozen ProteomeLM + versioned condition encoding table.

Run from repo root:
  python modeling/train.py --protocol splits/organism_single_holdout_largest_v0/protocol.json

Smoke:
  python modeling/train.py --protocol splits/organism_single_holdout_largest_v0/protocol.json \\
    --epochs 1 --max-train-rows 8000 --max-val-rows 4000 --shuffle-buffer 4000 \\
    --skip-full-row-counts

Chemistry OOD (val rows vs train-organism media components) uses ``--media-workbook`` (default
``data/media_composition_v3.xlsx``) and ``--experiments-parquet``; disable with ``--skip-chemistry-audit``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from condition_store import ExperimentConditionEncoding
from data import ArmName, count_split_row_stats, iter_val_batches, shuffled_training_batches
from embedding_store import EmbeddingStore
from fast_data import MaterializedSplit
from metrics import mean_within_gene_spearman_with_diagnostics, rmse_numpy
from model import GeneConditionMLP
from paths import EMBEDDING_LAYER8_DIR, REPO_ROOT, RUNS_ROOT, resolve_parquet_path
from split_diagnostics import compute_split_chemistry_report
from split_protocol import load_split_protocol

DEFAULT_CONDITION_MANIFEST = REPO_ROOT / "docs" / "condition_encoding_manifest_v0.json"


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s.strip("_") or "run"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gene × condition regression: frozen ProteomeLM + versioned condition encoding table."
    )
    p.add_argument(
        "--protocol",
        type=str,
        default="splits/organism_single_holdout_largest_v0/protocol.json",
        help="Path to split protocol JSON (repo-relative or absolute).",
    )
    p.add_argument(
        "--condition-manifest",
        type=str,
        default=str(DEFAULT_CONDITION_MANIFEST),
        help="JSON manifest for per-experiment condition encoding.",
    )
    p.add_argument("--arm", type=str, choices=("weighted_full", "strict_slice"), default="weighted_full")
    p.add_argument("--run-id", type=str, default="", help="Optional run id (default: derived stamp + uuid).")
    p.add_argument("--run-tag", type=str, default="", help="Tag included in default run folder name.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--cat-emb-dim", type=int, default=16)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--num-hidden", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--shuffle-buffer", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-rows", type=int, default=0, help="Cap train rows per epoch (0 = no cap).")
    p.add_argument("--max-val-rows", type=int, default=0, help="Cap val rows for eval (0 = no cap).")
    p.add_argument("--skip-full-row-counts", action="store_true")
    p.add_argument("--strict-min-cor12", type=float, default=0.4)
    p.add_argument("--strict-min-abs-t", type=float, default=2.0)
    p.add_argument("--cor12-floor", type=float, default=0.05)
    p.add_argument("--weight-t-scale", type=float, default=4.0)
    p.add_argument("--min-val-conditions-per-gene", type=int, default=2)
    p.add_argument("--embed-dir", type=str, default=str(EMBEDDING_LAYER8_DIR))
    p.add_argument("--parquet", type=str, default="", help="Override canonical long Parquet path.")
    p.add_argument(
        "--experiments-parquet",
        type=str,
        default="data/derived/canonical/v0/experiments.parquet",
        help="Canonical experiments.parquet (train media → component vocabulary for chemistry audit).",
    )
    p.add_argument(
        "--media-workbook",
        type=str,
        default="data/media_composition_v3.xlsx",
        help="Excel workbook with Media_Components sheet (chemistry OOD audit vs train organisms).",
    )
    p.add_argument(
        "--skip-chemistry-audit",
        action="store_true",
        help="Do not compute val chemistry-overlap stats (no extra Parquet / Excel read).",
    )
    p.add_argument("--output-dir", type=str, default=str(RUNS_ROOT))
    p.add_argument(
        "--materialized-dir",
        type=str,
        default="",
        help=(
            "Path to directory produced by materialize_training_data.py "
            "(contains train/ and val/ subdirs with meta.json + .npy arrays). "
            "When set, skips all Parquet streaming and Python row filtering each epoch."
        ),
    )
    p.add_argument(
        "--log-every-n-batches",
        type=int,
        default=0,
        help="Print train heartbeat every N batches (0 = only end-of-epoch logs).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arm: ArmName = args.arm  # type: ignore[assignment]
    device = torch.device(args.device)
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = REPO_ROOT / protocol_path
    protocol = load_split_protocol(protocol_path)
    rel_pq = args.parquet.strip() or protocol.canonical_fitness_parquet
    parquet_path = Path(rel_pq) if Path(rel_pq).is_absolute() else resolve_parquet_path(rel_pq)
    if not parquet_path.is_file():
        print(f"Missing canonical Parquet: {parquet_path}", file=sys.stderr)
        return 1

    cond_manifest = Path(args.condition_manifest)
    if not cond_manifest.is_absolute():
        cond_manifest = REPO_ROOT / cond_manifest
    if not cond_manifest.is_file():
        print(f"Missing condition manifest: {cond_manifest}", file=sys.stderr)
        return 1

    print("Loading experiment condition encoding…", flush=True)
    condition_store = ExperimentConditionEncoding(cond_manifest, repo_root=REPO_ROOT)
    summ = condition_store.manifest_summary()
    print(f"  {summ['encoding_id']}  indexed_experiments={summ['n_experiments_indexed']}", flush=True)

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = generated_utc.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    if args.run_id.strip():
        run_id = args.run_id.strip()
    else:
        tag = _slug(args.run_tag) if args.run_tag.strip() else "baseline"
        run_id = _slug(f"{stamp}_{protocol.protocol_id}_{arm}_{tag}_s{args.seed}_{uuid.uuid4().hex[:8]}")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_orgs = set(protocol.train_org_ids)
    val_orgs = set(protocol.val_org_ids)
    test_orgs = set(protocol.test_org_ids)

    # --- Decide: fast materialized path vs streaming Parquet path ---
    mat_dir_str = getattr(args, "materialized_dir", "").strip()
    mat_dir: Path | None = None
    if mat_dir_str:
        mat_dir = Path(mat_dir_str) if Path(mat_dir_str).is_absolute() else REPO_ROOT / mat_dir_str
    using_materialized = (
        mat_dir is not None
        and (mat_dir / "train" / "meta.json").is_file()
        and (mat_dir / "val" / "meta.json").is_file()
    )

    mat_train: MaterializedSplit | None = None
    mat_val: MaterializedSplit | None = None
    embed_store: EmbeddingStore | None = None

    if using_materialized:
        assert mat_dir is not None
        print(f"Loading pre-materialized data from {mat_dir}…", flush=True)
        mat_train = MaterializedSplit(mat_dir / "train", device)
        mat_val = MaterializedSplit(mat_dir / "val", device)
        gene_dim = mat_train.gene_dim
        print(
            f"  train: {mat_train.n_rows:,} rows | val: {mat_val.n_rows:,} rows",
            flush=True,
        )
    else:
        orgs_needed = protocol.train_org_ids | protocol.val_org_ids
        embed_dir = Path(args.embed_dir)
        if not embed_dir.is_absolute():
            embed_dir = REPO_ROOT / embed_dir
        print(f"Loading embeddings from {embed_dir} ({len(orgs_needed)} organisms)…", flush=True)
        embed_store = EmbeddingStore(embed_dir, set(orgs_needed), device=device)
        gene_dim = embed_store.gene_embedding_dim

    experiments_pq = Path(args.experiments_parquet)
    if not experiments_pq.is_absolute():
        experiments_pq = REPO_ROOT / experiments_pq
    media_workbook = Path(args.media_workbook)
    if not media_workbook.is_absolute():
        media_workbook = REPO_ROOT / media_workbook

    max_train = args.max_train_rows or None
    max_val = args.max_val_rows or None

    chem_audit: dict[str, object] | None = None
    if not args.skip_chemistry_audit and not using_materialized:
        chem_audit = compute_split_chemistry_report(
            experiments_parquet=experiments_pq,
            media_workbook=media_workbook,
            fitness_parquet=parquet_path,
            train_orgs=train_orgs,
            val_orgs=val_orgs,
            arm=arm,
            embed_store=embed_store,  # type: ignore[arg-type]
            condition_store=condition_store,
            strict_min_cor12=args.strict_min_cor12,
            strict_min_abs_t=args.strict_min_abs_t,
            cor12_floor=args.cor12_floor,
            weight_t_scale=args.weight_t_scale,
            max_val_rows=max_val,
        )

    cat_field_order = condition_store.cat_field_order
    n_cont = condition_store.n_cont_fields
    model = GeneConditionMLP(
        gene_dim=gene_dim,
        cat_field_max_ids=condition_store.cat_field_max_ids,
        cat_field_order=cat_field_order,
        cat_emb_dim=args.cat_emb_dim,
        n_cont=n_cont,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    huber = nn.HuberLoss(delta=args.huber_delta, reduction="none")

    row_stats: dict[str, int | float] = {}
    if using_materialized:
        assert mat_train is not None and mat_val is not None
        row_stats = {
            "n_train_rows_used_by_model_under_arm": mat_train.n_rows,
            "n_val_rows_used_by_model_under_arm": mat_val.n_rows,
        }
    elif not args.skip_full_row_counts:
        print("Counting train/val rows (full Parquet pass)…", flush=True)
        t1 = time.perf_counter()
        row_stats = count_split_row_stats(
            parquet_path,
            train_orgs,
            val_orgs,
            embed_store,  # type: ignore[arg-type]
            arm,
            condition_store.key_set(),
            strict_min_cor12=args.strict_min_cor12,
            strict_min_abs_t=args.strict_min_abs_t,
        )
        print(f"  row stats done ({time.perf_counter() - t1:.1f}s)", flush=True)

    log_every = max(0, int(args.log_every_n_batches))

    config = {
        "run_id": run_id,
        "generated_utc": generated_utc,
        "run_tag": args.run_tag.strip() or None,
        "protocol_id": protocol.protocol_id,
        "protocol_path": str(protocol_path.relative_to(REPO_ROOT)),
        "arm": arm,
        "parquet": str(parquet_path.relative_to(REPO_ROOT)),
        "condition_encoding_manifest": str(cond_manifest.relative_to(REPO_ROOT)),
        "condition_encoding_id": summ["encoding_id"],
        "condition_encoding_parquet": str(Path(summ["parquet_path"]).relative_to(REPO_ROOT)),
        "canonical_experiments_sha256_expected_in_manifest": summ.get("canonical_experiments_sha256_expected"),
        "epochs": args.epochs,
        "log_every_n_batches": log_every,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_hidden": args.num_hidden,
        "cat_emb_dim": args.cat_emb_dim,
        "huber_delta": args.huber_delta,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "device": str(device),
        "max_train_rows": max_train,
        "max_val_rows": max_val,
        "strict_min_cor12": args.strict_min_cor12,
        "strict_min_abs_t": args.strict_min_abs_t,
        "cor12_floor": args.cor12_floor,
        "weight_t_scale": args.weight_t_scale,
        "min_val_conditions_per_gene": args.min_val_conditions_per_gene,
        "cat_field_max_index": condition_store.cat_field_max_ids,
        "gene_dim": gene_dim,
        "train_org_ids": sorted(train_orgs),
        "val_org_ids": sorted(val_orgs),
        "test_org_ids": sorted(test_orgs),
        "split_axis_orgId": True,
        "experiments_parquet_chemistry_audit": str(experiments_pq.relative_to(REPO_ROOT))
        if experiments_pq.is_relative_to(REPO_ROOT)
        else str(experiments_pq),
        "media_workbook_chemistry_audit": str(media_workbook.relative_to(REPO_ROOT))
        if media_workbook.is_relative_to(REPO_ROOT)
        else str(media_workbook),
        "chemistry_audit_skipped": bool(args.skip_chemistry_audit),
        "row_weighting_note": "Arm weighted_full: Huber weighted by cor12 (floored) × min(1, abs_t/weight_t_scale).",
        "modular_inputs_note": "Gene: ProteomeLM .pt; Condition: table keyed by (orgId, expName).",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, object]] = []

    def _chem_epoch_fields() -> dict[str, object]:
        if not chem_audit:
            return {}
        out: dict[str, object] = {}
        for key in ("val_frac_rows_any_unseen_component", "val_frac_weight_on_unseen_component_rows"):
            v = chem_audit.get(key)
            if isinstance(v, float) and not math.isnan(v):
                out[key] = v
            else:
                out[key] = None
        n = chem_audit.get("n_val_rows_chemistry_audit")
        if isinstance(n, int):
            out["n_val_rows_chemistry_audit"] = n
        return out

    null_path = REPO_ROOT / "evaluation" / "outputs" / "null_baselines_m35.json"
    null_ref: float | None = None
    if null_path.is_file():
        nb = json.loads(null_path.read_text(encoding="utf-8"))
        block = nb.get(protocol.protocol_id, {}).get("val", {})
        null_ref = block.get("rmse_global_train_mean")

    def _json_float(x: float) -> float | None:
        if isinstance(x, float) and math.isnan(x):
            return None
        return x

    def _eval_pass(*, detailed: bool = False) -> dict[str, object]:
        model.eval()
        y_parts: list[np.ndarray] = []
        p_parts: list[np.ndarray] = []
        g_parts: list[str] = []
        n_val_batches = 0
        val_loss_num = 0.0
        val_loss_den = 0.0
        with torch.no_grad():
            if using_materialized:
                assert mat_val is not None
                _val_iter = mat_val.iter_val_batches(batch_size=args.batch_size)
            else:
                _val_iter = iter_val_batches(
                    parquet_path,
                    val_orgs,
                    arm,
                    embed_store,  # type: ignore[arg-type]
                    condition_store,
                    device,
                    batch_size=args.batch_size,
                    cat_field_order=cat_field_order,
                    n_cont=n_cont,
                    strict_min_cor12=args.strict_min_cor12,
                    strict_min_abs_t=args.strict_min_abs_t,
                    cor12_floor=args.cor12_floor,
                    weight_t_scale=args.weight_t_scale,
                    max_rows=max_val,
                )
            for x_g, cat_ids, x_cont, y, w, gks in _val_iter:
                pred = model(x_g, cat_ids, x_cont)
                loss_vec = huber(pred, y) * w
                val_loss_num += float(loss_vec.sum().detach().cpu())
                val_loss_den += float(w.sum().detach().cpu())
                y_parts.append(y.detach().cpu().numpy())
                p_parts.append(pred.detach().cpu().numpy())
                g_parts.extend(gks)
                n_val_batches += 1

        if not y_parts:
            raise RuntimeError("No val rows after filters / joins — check data paths and arm.")

        y_true = np.concatenate(y_parts)
        y_pred = np.concatenate(p_parts)
        g_keys = np.array(g_parts, dtype=object)
        val_rmse = rmse_numpy(y_true, y_pred)
        rho, n_genes, spearman_diag = mean_within_gene_spearman_with_diagnostics(
            y_true,
            y_pred,
            g_keys,
            min_conditions=args.min_val_conditions_per_gene,
        )
        val_loss = (val_loss_num / max(1e-8, val_loss_den)) if val_loss_den > 0 else float("nan")
        out: dict[str, object] = {
            "val_rmse": val_rmse,
            "val_loss_huber_weighted": val_loss,
            "mean_within_gene_spearman": rho,
            "n_genes_used_for_spearman": n_genes,
            "spearman_diagnostics": spearman_diag,
            "n_val_rows_scored": int(len(y_true)),
            "n_val_batches": n_val_batches,
        }
        if detailed:
            out["y_true"] = y_true
            out["y_pred"] = y_pred
            out["gene_keys"] = g_keys
        return out

    for epoch in range(args.epochs):
        model.train()
        n_batches = 0
        losses = []
        t_ep = time.perf_counter()
        if log_every:
            print(
                f"epoch {epoch + 1}/{args.epochs}  train started "
                f"(heartbeat every {log_every} batches)",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch + 1}/{args.epochs}  train started "
                f"(no per-batch logs; use --log-every-n-batches N for progress)",
                flush=True,
            )
        if using_materialized:
            assert mat_train is not None
            _train_iter = mat_train.iter_train_batches(
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=epoch,
            )
        else:
            _train_iter = shuffled_training_batches(
                parquet_path,
                train_orgs,
                arm,
                embed_store,  # type: ignore[arg-type]
                condition_store,
                device,
                batch_size=args.batch_size,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.seed + epoch * 10_000,
                cat_field_order=cat_field_order,
                n_cont=n_cont,
                strict_min_cor12=args.strict_min_cor12,
                strict_min_abs_t=args.strict_min_abs_t,
                cor12_floor=args.cor12_floor,
                weight_t_scale=args.weight_t_scale,
                max_rows=max_train,
            )
        for x_g, cat_ids, x_cont, y, w in _train_iter:
            opt.zero_grad(set_to_none=True)
            pred = model(x_g, cat_ids, x_cont)
            loss_vec = huber(pred, y) * w
            denom = w.sum().clamp_min(1e-8)
            loss = loss_vec.sum() / denom
            loss.backward()
            opt.step()
            n_batches += 1
            losses.append(float(loss.detach().cpu()))
            if log_every and n_batches % log_every == 0:
                tail = float(np.mean(losses[-log_every:])) if len(losses) >= log_every else float(losses[-1])
                print(
                    f"epoch {epoch + 1}/{args.epochs}  batch={n_batches}  "
                    f"recent_mean_loss={tail:.6f}  wall_s={time.perf_counter() - t_ep:.1f}",
                    flush=True,
                )
        train_loss = float(np.mean(losses)) if losses else float("nan")
        is_last_epoch = (epoch == args.epochs - 1)
        eval_stats = _eval_pass(detailed=is_last_epoch)
        row_hist: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss_huber_weighted": train_loss,
            "val_loss_huber_weighted": float(eval_stats["val_loss_huber_weighted"]),
            "val_rmse": float(eval_stats["val_rmse"]),
            "mean_within_gene_spearman": float(eval_stats["mean_within_gene_spearman"]),
            "n_genes_used_for_spearman": int(eval_stats["n_genes_used_for_spearman"]),
            "n_val_rows_scored": int(eval_stats["n_val_rows_scored"]),
        }
        row_hist.update(_chem_epoch_fields())
        history.append(row_hist)
        print(
            f"epoch {epoch + 1}/{args.epochs}  batches={n_batches}  "
            f"train_loss={train_loss:.6f}  val_loss={float(eval_stats['val_loss_huber_weighted']):.6f}  "
            f"val_rmse={float(eval_stats['val_rmse']):.6f}  "
            f"val_spearman={float(eval_stats['mean_within_gene_spearman']):.6f}  "
            f"wall_s={time.perf_counter() - t_ep:.1f}",
            flush=True,
        )

    if not history:
        # epochs=0 path: run one eval without training
        eval_stats = _eval_pass(detailed=True)
        row_hist0: dict[str, object] = {
            "epoch": 0,
            "train_loss_huber_weighted": float("nan"),
            "val_loss_huber_weighted": float(eval_stats["val_loss_huber_weighted"]),
            "val_rmse": float(eval_stats["val_rmse"]),
            "mean_within_gene_spearman": float(eval_stats["mean_within_gene_spearman"]),
            "n_genes_used_for_spearman": int(eval_stats["n_genes_used_for_spearman"]),
            "n_val_rows_scored": int(eval_stats["n_val_rows_scored"]),
        }
        row_hist0.update(_chem_epoch_fields())
        history.append(row_hist0)

    # Reuse the last epoch's detailed eval — no extra val pass needed.
    val_rmse = float(eval_stats["val_rmse"])
    rho = float(eval_stats["mean_within_gene_spearman"])
    n_genes = int(eval_stats["n_genes_used_for_spearman"])
    n_val_rows_scored = int(eval_stats["n_val_rows_scored"])
    n_val_batches = int(eval_stats["n_val_batches"])
    spearman_diag = eval_stats["spearman_diagnostics"]
    y_true = eval_stats["y_true"]
    y_pred = eval_stats["y_pred"]
    gene_keys = eval_stats["gene_keys"]

    null_ref_intersection: float | None = None
    if null_ref is not None and not (isinstance(null_ref, float) and math.isnan(null_ref)):
        null_pred = np.full_like(y_true, fill_value=float(null_ref), dtype=np.float64)
        null_ref_intersection = rmse_numpy(y_true.astype(np.float64), null_pred)

    org_ids = np.array([str(gk).split(":", 1)[0] for gk in gene_keys], dtype=object)
    per_org_rows: list[dict[str, object]] = []
    for org in sorted({str(o) for o in org_ids.tolist()}):
        m = org_ids == org
        if not np.any(m):
            continue
        rmse_org = rmse_numpy(y_true[m], y_pred[m])
        rho_org, n_genes_org, diag_org = mean_within_gene_spearman_with_diagnostics(
            y_true[m],
            y_pred[m],
            gene_keys[m],
            min_conditions=args.min_val_conditions_per_gene,
        )
        per_org_rows.append(
            {
                "orgId": org,
                "n_rows": int(np.sum(m)),
                "val_rmse": rmse_org,
                "mean_within_gene_spearman": rho_org,
                "n_genes_used_for_spearman": n_genes_org,
                "n_genes_total_in_val": int(diag_org["n_genes_total_in_val"]),
            }
        )

    metrics: dict[str, object] = {
        "val_rmse": _json_float(val_rmse),
        "mean_within_gene_spearman": _json_float(rho),
        "n_genes_used_for_spearman": n_genes,
        "n_val_rows_scored": n_val_rows_scored,
        "n_val_batches": n_val_batches,
        "reference_null_rmse_val_global_train_mean": null_ref,
        "reference_null_rmse_val_global_train_mean_on_scored_rows": _json_float(null_ref_intersection)
        if null_ref_intersection is not None
        else None,
        "spearman_diagnostics": spearman_diag,
        "per_org_metrics_file": "per_org_metrics.json",
        "history_file": "history.json",
        "curves_file": "learning_curves.png",
        "per_org_plot_file": "per_org_metrics.png",
        "note_null_compare": (
            "M3.5 null is on all val rows with fit; model RMSE is on val rows after embedding join "
            "and arm policy (strict_slice filters cor12/abs_t). "
            "See row_count_* for how many val rows have fit vs embedding vs scored."
        ),
    }
    if max_val is not None:
        metrics["note_eval_subset"] = f"--max-val-rows={max_val} capped evaluation; row_count_* are full-protocol counts."
    if row_stats:
        for k, v in row_stats.items():
            metrics[f"row_count_{k}"] = v
    else:
        metrics["note_row_counts"] = "Skipped full-file row counts (--skip-full-row-counts)."

    if chem_audit:
        for k, v in chem_audit.items():
            if k == "chemistry_audit_note":
                metrics["split_chemistry_audit_note"] = v
            elif isinstance(v, float):
                metrics[k] = _json_float(v)
            else:
                metrics[k] = v
    else:
        metrics["split_chemistry_audit"] = None
        if args.skip_chemistry_audit:
            metrics["split_chemistry_audit_skip_reason"] = "--skip-chemistry-audit"
        elif not media_workbook.is_file():
            metrics["split_chemistry_audit_skip_reason"] = f"missing_workbook:{media_workbook}"
        elif not experiments_pq.is_file():
            metrics["split_chemistry_audit_skip_reason"] = f"missing_experiments_parquet:{experiments_pq}"
        else:
            metrics["split_chemistry_audit_skip_reason"] = "unknown"

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "per_org_metrics.json").write_text(json.dumps(per_org_rows, indent=2), encoding="utf-8")

    epochs = [int(h["epoch"]) for h in history]
    train_loss_curve = [float(h["train_loss_huber_weighted"]) for h in history]
    val_loss_curve = [float(h["val_loss_huber_weighted"]) for h in history]
    val_rmse_curve = [float(h["val_rmse"]) for h in history]
    val_spearman_curve = [float(h["mean_within_gene_spearman"]) for h in history]

    split_title = (
        f"{protocol.protocol_id} | train orgs={len(train_orgs)} | "
        f"val={','.join(sorted(val_orgs))} | test={','.join(sorted(test_orgs))}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax0, ax1), (ax2, ax3) = axes
    ax0.plot(epochs, train_loss_curve, marker="o", label="train_loss_huber_weighted")
    ax0.plot(epochs, val_loss_curve, marker="o", label="val_loss_huber_weighted")
    ax0.set_title("Training vs evaluation loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Huber loss (weighted)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(epochs, val_rmse_curve, marker="o", color="tab:orange", label="val_rmse")
    if null_ref is not None and not (isinstance(null_ref, float) and math.isnan(null_ref)):
        ax1.axhline(float(null_ref), color="tab:red", linestyle="--", label="null_baseline_rmse")
    ax1.set_title("Validation RMSE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMSE")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, val_spearman_curve, marker="o", color="tab:green", label="val_spearman")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_title("Within-gene Spearman (val)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Spearman")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.set_title("Val rows: unseen chemistry vs train media")
    ax3.set_ylabel("Fraction")
    ax3.set_ylim(0.0, 1.05)
    n_chem = int(chem_audit["n_val_rows_chemistry_audit"]) if chem_audit else 0
    if chem_audit and n_chem > 0:
        fr = chem_audit["val_frac_rows_any_unseen_component"]
        fw = chem_audit["val_frac_weight_on_unseen_component_rows"]
        fr_f = float(fr) if isinstance(fr, (int, float)) and not (isinstance(fr, float) and math.isnan(fr)) else 0.0
        fw_f = float(fw) if isinstance(fw, (int, float)) and not (isinstance(fw, float) and math.isnan(fw)) else 0.0
        xpos = [0, 1]
        ax3.bar(
            xpos,
            [fr_f, fw_f],
            color=["tab:blue", "tab:purple"],
            alpha=0.85,
            tick_label=[
                "Row fraction\n(any unseen component)",
                "Huber weight fraction\n(on those rows)",
            ],
        )
        ax3.axhline(0.0, color="gray", linewidth=0.8)
        n_train_comp = chem_audit.get("n_distinct_components_in_train_media", "?")
        ax3.text(
            0.5,
            1.02,
            f"n_val_rows(audit)={n_chem} | train component types={n_train_comp}",
            transform=ax3.transAxes,
            ha="center",
            fontsize=9,
        )
    else:
        reason = metrics.get("split_chemistry_audit_skip_reason")
        if reason is None and chem_audit is not None:
            reason = f"n_val_rows_chemistry_audit={chem_audit.get('n_val_rows_chemistry_audit', 0)}"
        if reason is None:
            reason = "no val rows or audit off"
        ax3.text(
            0.5,
            0.55,
            "Chemistry OOD audit unavailable\n(same filters as training val iterator).",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=10,
        )
        ax3.text(0.5, 0.35, str(reason), ha="center", va="center", transform=ax3.transAxes, fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([0, 0.5, 1.0])

    fig.suptitle(split_title, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if per_org_rows:
        per_org_rows_sorted = sorted(per_org_rows, key=lambda r: float(r["val_rmse"]), reverse=True)
        org_names = [str(r["orgId"]) for r in per_org_rows_sorted]
        org_rmse = [float(r["val_rmse"]) for r in per_org_rows_sorted]
        org_spear = [float(r["mean_within_gene_spearman"]) for r in per_org_rows_sorted]

        fig2, axes2 = plt.subplots(2, 1, figsize=(max(10, 0.35 * len(org_names)), 8), sharex=True)
        bx0, bx1 = axes2
        bx0.bar(org_names, org_rmse, color="tab:orange", alpha=0.85)
        if null_ref_intersection is not None:
            bx0.axhline(
                float(null_ref_intersection),
                color="tab:red",
                linestyle="--",
                label="null_on_scored_rows_rmse",
            )
            bx0.legend()
        bx0.set_ylabel("RMSE")
        bx0.set_title("Per-organism validation RMSE (sorted; outliers at left)")
        bx0.grid(True, axis="y", alpha=0.3)

        bx1.bar(org_names, org_spear, color="tab:green", alpha=0.85)
        bx1.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        bx1.set_ylabel("Within-gene Spearman")
        bx1.set_title("Per-organism validation Spearman")
        bx1.grid(True, axis="y", alpha=0.3)
        bx1.set_xticks(range(len(org_names)))
        bx1.set_xticklabels(org_names, rotation=90)
        bx1.set_xlabel("orgId")
        fig2.tight_layout()
        fig2.savefig(run_dir / "per_org_metrics.png", dpi=150)
        plt.close(fig2)

    summary_lines = [
        f"# Experiment summary: {run_id}",
        "",
        f"- protocol: `{config['protocol_id']}`",
        f"- organism split: train={len(train_orgs)} orgs | val={sorted(val_orgs)} | test={sorted(test_orgs)}",
        f"- arm: `{arm}`",
        f"- run_tag: `{args.run_tag.strip() or 'None'}`",
        f"- epochs: `{args.epochs}`",
        f"- model: `GeneConditionMLP` (hidden_dim={args.hidden_dim}, num_hidden={args.num_hidden}, dropout={args.dropout})",
        f"- val_rmse: `{_json_float(val_rmse)}`",
        f"- val_spearman: `{_json_float(rho)}`",
        f"- null_rmse_global_val: `{_json_float(null_ref) if null_ref is not None else None}`",
        (
            f"- null_rmse_on_scored_rows: `{_json_float(null_ref_intersection)}`"
            if null_ref_intersection is not None
            else "- null_rmse_on_scored_rows: `None`"
        ),
        f"- n_val_rows_scored: `{n_val_rows_scored}`",
        f"- per_org_metrics: `per_org_metrics.json`",
        f"- per_org_plot: `per_org_metrics.png`",
        f"- learning_curves: `learning_curves.png`",
    ]
    if chem_audit and int(chem_audit.get("n_val_rows_chemistry_audit", 0) or 0) > 0:
        summary_lines.extend(
            [
                f"- val_frac_rows_any_unseen_component: `{chem_audit.get('val_frac_rows_any_unseen_component')}`",
                f"- val_frac_weight_on_unseen_component_rows: `{chem_audit.get('val_frac_weight_on_unseen_component_rows')}`",
                f"- (unseen = medium has no Media_Components row, or any component not in train-organism media union)",
            ]
        )
    elif not args.skip_chemistry_audit:
        summary_lines.append("- chemistry OOD audit: skipped or unavailable (see `metrics.json` split_chemistry_audit_*)")
    (run_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Wrote {run_dir.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
