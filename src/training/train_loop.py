"""Training utilities for S5 quality-policy comparisons."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainLoopConfig:
    """Hyperparameters for a fixed S5 training run."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8192
    epochs: int = 8
    device: str = "cpu"
    loss_fn_name: str = "mse"
    huber_delta: float = 1.0


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _within_gene_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_keys: np.ndarray,
    *,
    min_conditions: int,
    min_iqr: float,
) -> tuple[float, int]:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "gene": gene_keys})
    vals: list[float] = []
    for _gene, sub in df.groupby("gene"):
        if len(sub) < min_conditions:
            continue
        iqr = float(np.percentile(sub["y_true"], 75) - np.percentile(sub["y_true"], 25))
        if iqr < min_iqr:
            continue
        r, _ = spearmanr(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        if not np.isnan(r):
            vals.append(float(r))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), int(len(vals))


def _per_org_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, org_ids: np.ndarray
) -> dict[str, float]:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "orgId": org_ids})
    out: dict[str, float] = {}
    for org, sub in df.groupby("orgId"):
        out[str(org)] = _rmse(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
    return out


def _residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_true - y_pred
    return {
        "p1": float(np.percentile(resid, 1)),
        "p5": float(np.percentile(resid, 5)),
        "p25": float(np.percentile(resid, 25)),
        "p50": float(np.percentile(resid, 50)),
        "p75": float(np.percentile(resid, 75)),
        "p95": float(np.percentile(resid, 95)),
        "p99": float(np.percentile(resid, 99)),
    }


def _batch_index_iterator(n_rows: int, batch_size: int, *, shuffle: bool, seed: int):
    idx = np.arange(n_rows, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n_rows, batch_size):
        yield idx[start : start + batch_size]


def train_one_arm(
    *,
    arm_name: str,
    seed: int,
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    val_gene_keys: np.ndarray,
    val_org_ids: np.ndarray,
    spearman_min_conditions: int,
    spearman_min_iqr: float,
    config: TrainLoopConfig,
) -> tuple[pd.DataFrame, dict]:
    """Run fixed-length training and return per-epoch metrics and summary."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    device = _resolve_device(config.device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.loss_fn_name == "huber":
        loss_fn = torch.nn.HuberLoss(reduction="none", delta=config.huber_delta)
    else:
        loss_fn = torch.nn.MSELoss(reduction="none")

    fast_path = all(
        hasattr(train_dataset, attr) for attr in ["row_batch", "embedding_matrix", "chemistry_matrix", "weights"]
    ) and all(hasattr(val_dataset, attr) for attr in ["row_batch", "embedding_matrix", "chemistry_matrix", "weights"])
    if not fast_path:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    rows: list[dict] = []
    best_val_rmse = float("inf")
    best_epoch = -1
    best_summary: dict = {}

    for epoch in range(1, int(config.epochs) + 1):
        t0 = time.time()
        model.train()
        train_pred_parts: list[np.ndarray] = []
        train_true_parts: list[np.ndarray] = []
        grad_norm_acc = 0.0
        grad_norm_n = 0
        if fast_path:
            rb = train_dataset.row_batch
            for b_idx in _batch_index_iterator(
                len(rb.y), config.batch_size, shuffle=True, seed=seed * 1000 + epoch
            ):
                gene_emb = torch.from_numpy(train_dataset.embedding_matrix[rb.gene_idx[b_idx]]).to(device)
                chem = torch.from_numpy(train_dataset.chemistry_matrix[rb.exp_idx[b_idx]]).to(device)
                y = torch.from_numpy(rb.y[b_idx]).to(device)
                w = torch.from_numpy(train_dataset.weights[b_idx]).to(device)
                pred = model(gene_emb, chem)
                loss_vec = loss_fn(pred, y)
                loss = (loss_vec * w).sum() / torch.clamp(w.sum(), min=1e-9)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                batch_grad_vals = []
                for p in model.parameters():
                    if p.grad is not None:
                        batch_grad_vals.append(float(p.grad.detach().norm().item()))
                if batch_grad_vals:
                    grad_norm_acc += float(np.mean(batch_grad_vals))
                    grad_norm_n += 1
                opt.step()
                train_pred_parts.append(pred.detach().cpu().numpy())
                train_true_parts.append(y.detach().cpu().numpy())
        else:
            for batch in train_loader:
                gene_emb = batch["gene_emb"].to(device)
                chem = batch["chem_multihot"].to(device)
                y = batch["y"].to(device)
                w = batch["weight"].to(device)
                pred = model(gene_emb, chem)
                loss_vec = loss_fn(pred, y)
                loss = (loss_vec * w).sum() / torch.clamp(w.sum(), min=1e-9)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Mean grad-norm proxy over parameter tensors.
                batch_grad_vals = []
                for p in model.parameters():
                    if p.grad is not None:
                        batch_grad_vals.append(float(p.grad.detach().norm().item()))
                if batch_grad_vals:
                    grad_norm_acc += float(np.mean(batch_grad_vals))
                    grad_norm_n += 1
                opt.step()
                train_pred_parts.append(pred.detach().cpu().numpy())
                train_true_parts.append(y.detach().cpu().numpy())

        train_pred = np.concatenate(train_pred_parts).astype(np.float64, copy=False)
        train_true = np.concatenate(train_true_parts).astype(np.float64, copy=False)
        train_rmse = _rmse(train_true, train_pred)
        train_mae = _mae(train_true, train_pred)

        model.eval()
        val_pred_parts: list[np.ndarray] = []
        val_true_parts: list[np.ndarray] = []
        with torch.no_grad():
            if fast_path:
                rbv = val_dataset.row_batch
                for b_idx in _batch_index_iterator(
                    len(rbv.y), config.batch_size, shuffle=False, seed=0
                ):
                    gene_emb = torch.from_numpy(val_dataset.embedding_matrix[rbv.gene_idx[b_idx]]).to(device)
                    chem = torch.from_numpy(val_dataset.chemistry_matrix[rbv.exp_idx[b_idx]]).to(device)
                    y = torch.from_numpy(rbv.y[b_idx]).to(device)
                    pred = model(gene_emb, chem)
                    val_pred_parts.append(pred.detach().cpu().numpy())
                    val_true_parts.append(y.detach().cpu().numpy())
            else:
                for batch in val_loader:
                    gene_emb = batch["gene_emb"].to(device)
                    chem = batch["chem_multihot"].to(device)
                    y = batch["y"].to(device)
                    pred = model(gene_emb, chem)
                    val_pred_parts.append(pred.detach().cpu().numpy())
                    val_true_parts.append(y.detach().cpu().numpy())
        val_pred = np.concatenate(val_pred_parts).astype(np.float64, copy=False)
        val_true = np.concatenate(val_true_parts).astype(np.float64, copy=False)
        val_rmse = _rmse(val_true, val_pred)
        val_mae = _mae(val_true, val_pred)

        ws, n_eligible = _within_gene_spearman(
            val_true,
            val_pred,
            val_gene_keys,
            min_conditions=spearman_min_conditions,
            min_iqr=float(spearman_min_iqr),
        )
        per_org = _per_org_rmse(val_true, val_pred, val_org_ids)
        param_norm = float(np.mean([p.detach().norm().item() for p in model.parameters()]))
        wallclock = float(time.time() - t0)

        row = {
            "arm": arm_name,
            "seed": int(seed),
            "epoch": int(epoch),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_within_gene_spearman": ws,
            "val_n_genes_eligible": int(n_eligible),
            "grad_norm_mean": float(grad_norm_acc / max(grad_norm_n, 1)),
            "param_norm": param_norm,
            "wallclock_seconds": wallclock,
            "per_org_val_rmse_json": pd.Series(per_org).to_json(),
        }
        rows.append(row)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_summary = {
                "best_epoch": int(epoch),
                "best_val_rmse": float(val_rmse),
                "best_val_mae": float(val_mae),
                "train_val_gap_rmse": float(train_rmse - val_rmse),
                "train_val_gap_mae": float(train_mae - val_mae),
                "val_residual_quantiles_json": pd.Series(_residual_quantiles(val_true, val_pred)).to_json(),
                # T1+ uses these for bootstrap CIs and homology-bin breakdowns.
                # Numpy arrays not JSON-serializable; downstream code must strip
                # them before saving the summary to disk (S5's flow drops them
                # implicitly via pd.DataFrame coercion).
                "_best_val_pred": val_pred.copy(),
                "_best_val_true": val_true.copy(),
            }

    metrics_df = pd.DataFrame(rows)
    summary = {
        "arm": arm_name,
        "seed": int(seed),
        "n_train_rows": int(len(train_dataset)),
        "n_val_rows": int(len(val_dataset)),
        **best_summary,
    }
    return metrics_df, summary

