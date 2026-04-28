"""Figure generation for S2 (REFACTORPLAN §11).

Each function generates one numbered figure (PNG + sibling CSV) at
research_log/figures/stage2/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import reporting as R

log = logging.getLogger(__name__)

FIG_DIR = Path("research_log/figures/stage2")


def fig_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASELINES = ["global_train_mean", "per_condition_mean", "per_organism_mean",
              "additive_baseline", "embedding_nn"]


def _baseline_metrics_long(per_protocol: dict) -> pd.DataFrame:
    """Long-form table: protocol × baseline × metric → value."""
    rows = []
    for pid, info in per_protocol.items():
        for b in _BASELINES:
            data = info["baselines"][b]
            rows.append({"protocol": pid, "baseline": b, "metric": "rmse",
                         "value": float(data["rmse"])})
            rows.append({"protocol": pid, "baseline": b, "metric": "mae",
                         "value": float(data["mae"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 01 — baseline RMSE+MAE per protocol
# ---------------------------------------------------------------------------

def fig_01_baseline_metrics_per_protocol(per_protocol: dict) -> None:
    """Grouped bar chart: each protocol gets 5 baselines × 2 metrics (RMSE, MAE)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    long_df = _baseline_metrics_long(per_protocol)
    protocols = list(per_protocol.keys())
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.6 * len(protocols) * 2), 5),
                              sharey=False)
    metrics = ["rmse", "mae"]
    for ax, metric in zip(axes, metrics):
        sub = long_df[long_df["metric"] == metric]
        # pivot to protocol × baseline
        pivot = sub.pivot(index="protocol", columns="baseline", values="value")
        pivot = pivot[_BASELINES].loc[protocols]
        x = np.arange(len(protocols))
        bar_w = 0.16
        for i, b in enumerate(_BASELINES):
            ax.bar(x + (i - 2) * bar_w, pivot[b].to_numpy(), bar_w, label=b)
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} per baseline × protocol")
        ax.grid(axis="y", alpha=0.3)
        if metric == "rmse":
            ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    out = fig_dir() / "01_baseline_metrics_per_protocol.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    long_df.to_csv(out.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 02 — difficulty ladder
# ---------------------------------------------------------------------------

def fig_02_difficulty_ladder(per_protocol: dict) -> None:
    """Sort protocols by additive_baseline RMSE; show alongside global_mean RMSE."""
    rows = []
    for pid, info in per_protocol.items():
        rows.append({
            "protocol": pid,
            "additive_rmse": info["baselines"]["additive_baseline"]["rmse"],
            "global_mean_rmse": info["baselines"]["global_train_mean"]["rmse"],
            "embedding_nn_rmse": info["baselines"]["embedding_nn"]["rmse"],
            "gap_global_minus_additive":
                info["baselines"]["global_train_mean"]["rmse"]
                - info["baselines"]["additive_baseline"]["rmse"],
        })
    df = pd.DataFrame(rows).sort_values("additive_rmse", ascending=True)
    R.save_coverage_curve(
        df.assign(_idx=np.arange(len(df))),
        primary_col="additive_rmse",
        secondary_col="global_mean_rmse",
        sort_descending=False,
        log_secondary=False,
        title="Difficulty ladder: protocols sorted by additive-baseline RMSE",
        path=fig_dir() / "02_difficulty_ladder.png",
        primary_label="additive RMSE",
        secondary_label="global-mean RMSE",
    )
    df.to_csv((fig_dir() / "02_difficulty_ladder.png").with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 03 — Spearman eligibility curves
# ---------------------------------------------------------------------------

def fig_03_spearman_eligibility_at_m(per_protocol: dict) -> None:
    """For each protocol: n_genes_eligible vs m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for pid, info in per_protocol.items():
        elig = info["power"]["eligibility_curve"]
        ms = [e["m"] for e in elig]
        ns = [e["n_genes_eligible"] for e in elig]
        ax.plot(ms, ns, marker="o", label=pid)
        for m, n in zip(ms, ns):
            rows.append({"protocol": pid, "m": m, "n_genes_eligible": n})
    ax.set_xlabel("min conditions per gene (m)")
    ax.set_ylabel("n_genes_eligible")
    ax.set_title("Spearman eligibility curves — n eligible genes vs m, per protocol")
    ax.axvline(5, color="red", lw=1, ls="--", alpha=0.6, label="locked m=5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = fig_dir() / "03_spearman_eligibility_at_m.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 04 — Bootstrap CI per protocol (additive baseline Spearman)
# ---------------------------------------------------------------------------

def fig_04_bootstrap_ci_per_protocol(per_protocol: dict) -> None:
    """Mean ± 95% CI of within-gene Spearman for the additive baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    protocols = list(per_protocol.keys())
    means, lows, highs, ns, roles = [], [], [], [], []
    for pid in protocols:
        boot = per_protocol[pid]["power"]["bootstrap_ci"]
        means.append(boot["mean"]); lows.append(boot["ci_low"]); highs.append(boot["ci_high"])
        ns.append(boot["n_genes_used"])
        roles.append(per_protocol[pid]["power"]["spearman_role"])
        rows.append({"protocol": pid, **boot,
                     "spearman_role": per_protocol[pid]["power"]["spearman_role"]})
    means_a = np.array(means); lows_a = np.array(lows); highs_a = np.array(highs)
    err_low = means_a - lows_a; err_high = highs_a - means_a
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(protocols)), 5))
    x = np.arange(len(protocols))
    colors = ["C0" if r == "primary" else "C7" for r in roles]
    ax.errorbar(x, means_a, yerr=[err_low, err_high], fmt="o", capsize=5,
                ecolor="gray", color="black", markersize=8)
    for i, (xi, mi, role, n) in enumerate(zip(x, means_a, roles, ns)):
        ax.scatter(xi, mi, s=80, color=colors[i], zorder=3,
                   label=role if i == 0 or roles[i] != roles[i - 1] else None)
        ax.text(xi, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] != 0 else 0.95,
                f"n={n}", ha="center", fontsize=7)
    ax.axhline(0, color="red", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(protocols, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean within-gene Spearman (additive baseline)")
    ax.set_title("Bootstrap 95% CI of within-gene Spearman (color = role decision)")
    ax.grid(axis="y", alpha=0.3)
    out = fig_dir() / "04_bootstrap_ci_per_protocol.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 05 — Permutation null distribution per protocol
# ---------------------------------------------------------------------------

def fig_05_permutation_null_spearman(per_protocol: dict) -> None:
    """Show null permutation Spearman p95/p99 vs the actual mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    protocols = list(per_protocol.keys())
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(protocols)), 5))
    x = np.arange(len(protocols))
    actual = []
    null_p95 = []
    null_p99 = []
    for pid in protocols:
        perm = per_protocol[pid]["power"]["permutation_null"]
        boot = per_protocol[pid]["power"]["bootstrap_ci"]
        actual.append(boot["mean"])
        null_p95.append(perm["null_p95"])
        null_p99.append(perm["null_p99"])
        rows.append({"protocol": pid, "actual_mean_spearman": boot["mean"],
                     **perm})
    actual_a = np.array(actual); p95_a = np.array(null_p95); p99_a = np.array(null_p99)
    ax.bar(x - 0.2, actual_a, 0.4, label="actual mean", color="C0")
    ax.bar(x + 0.2, p95_a, 0.4, label="null p95", color="C7", alpha=0.7)
    ax.scatter(x + 0.2, p99_a, s=40, color="red", marker="_", label="null p99")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(protocols, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Spearman")
    ax.set_title("Actual vs permutation-null mean within-gene Spearman")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    out = fig_dir() / "05_permutation_null_spearman.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 06 — Residual quantile profile per protocol
# ---------------------------------------------------------------------------

def fig_06_residual_quantile_profile(per_protocol: dict) -> None:
    """Heteroscedastic-noise diagnostic: residual quantiles per protocol."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qs = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for pid, info in per_protocol.items():
        q = info["noise_diagnostics"]["additive_residual_quantiles"]
        ys = [q[k] for k in qs]
        ax.plot(range(len(qs)), ys, marker="o", label=pid)
        for k in qs:
            rows.append({"protocol": pid, "quantile": k, "value": q[k]})
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(len(qs))); ax.set_xticklabels(qs)
    ax.set_xlabel("residual quantile")
    ax.set_ylabel("residual (val - pred)")
    ax.set_title("Additive-baseline residual quantile profile per protocol")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = fig_dir() / "06_residual_quantile_profile.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Figure 07 — Per-organism residual spread (additive baseline)
# ---------------------------------------------------------------------------

def fig_07_per_organism_residual_spread(per_protocol: dict) -> None:
    """For each protocol: per-org IQR of additive-baseline residuals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(per_protocol),
                              figsize=(max(10, 3 * len(per_protocol)), 4),
                              sharey=False)
    if len(per_protocol) == 1:
        axes = [axes]
    rows = []
    for ax, (pid, info) in zip(axes, per_protocol.items()):
        per_org = pd.DataFrame(info["noise_diagnostics"]["per_organism_residual"])
        per_org = per_org.sort_values("std", ascending=False)
        x = np.arange(len(per_org))
        ax.bar(x, per_org["std"].to_numpy(), label="std", color="C0", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(per_org["orgId"].tolist(), rotation=90, fontsize=7)
        ax.set_title(pid, fontsize=9)
        ax.set_ylabel("residual std")
        ax.grid(axis="y", alpha=0.3)
        for _, r in per_org.iterrows():
            rows.append({"protocol": pid, **r})
    fig.suptitle("Per-organism additive-baseline residual std (heteroscedastic noise)")
    fig.tight_layout()
    out = fig_dir() / "07_per_organism_residual_spread.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
