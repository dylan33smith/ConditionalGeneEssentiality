"""R0 figure generation. Each function writes a PNG + sibling CSV.

Mirrors the convention in src/experiments/stage1/figures.py: matplotlib only,
no seaborn, fixed figure size, agg backend, deterministic output.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FIGURES_DIR = Path("research_log/figures/r0_data")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str, df: pd.DataFrame | None = None) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    if df is not None:
        csv_path = FIGURES_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        log.info("wrote %s", csv_path)


def _grid_dims(n: int) -> tuple[int, int]:
    cols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / cols))
    return rows, cols


# ---------------------------------------------------------------------------

def fig_01_m_per_gene(cond_per_gene: pd.DataFrame) -> None:
    orgs = sorted(cond_per_gene["orgId"].unique())
    rows, cols = _grid_dims(len(orgs))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6),
                             sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()
    for ax, org in zip(axes, orgs):
        sub = cond_per_gene[cond_per_gene["orgId"] == org]
        ax.hist(sub["m_g"], bins=40, color="#3a7", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{org} (n_genes={len(sub)})", fontsize=8)
        ax.set_xlabel("m_g")
        ax.set_ylabel("genes")
    for ax in axes[len(orgs):]:
        ax.set_visible(False)
    fig.suptitle("R0-A.01 — Conditions per gene (m_g), per organism", fontsize=11)
    fig.tight_layout()
    _save(fig, "01_m_per_gene_per_org", cond_per_gene)


def fig_02_iqr_per_gene(iqr_df: pd.DataFrame) -> None:
    """Per-org distribution of tail_g (PRIMARY) overlaid with iqr_g (DIAGNOSTIC).

    Locked R-LOCK-1 gate is `tail_g = p95 − p5`. IQR is kept on the same
    axis to show how much sparse-conditional signal IQR misses: a gene
    with tail_g >> iqr_g is essential in a few conditions but not most.
    """
    orgs = sorted(iqr_df["orgId"].unique())
    rows, cols = _grid_dims(len(orgs))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.6))
    axes = np.atleast_1d(axes).flatten()
    for ax, org in zip(axes, orgs):
        sub = iqr_df[iqr_df["orgId"] == org]
        ax.hist(sub["tail_g"], bins=50, color="#37a", alpha=0.7,
                edgecolor="black", linewidth=0.2, label=f"tail (p95−p5)")
        ax.hist(sub["iqr_g"],  bins=50, color="#a73", alpha=0.55,
                edgecolor="black", linewidth=0.2, label=f"IQR (q75−q25)")
        ax.axvline(float(sub["tail_g"].median()), color="#04a",
                   linestyle="--", linewidth=0.7)
        ax.axvline(float(sub["iqr_g"].median()),  color="#a30",
                   linestyle="--", linewidth=0.7)
        ax.set_title(f"{org}  tail med={sub['tail_g'].median():.2f}  "
                     f"iqr med={sub['iqr_g'].median():.2f}", fontsize=7)
        ax.set_xlabel("spread", fontsize=7)
        if org == orgs[0]:
            ax.legend(fontsize=6, loc="upper right")
    for ax in axes[len(orgs):]:
        ax.set_visible(False)
    fig.suptitle("R0-A.02 — Spread per gene per organism (PRIMARY: tail_g = p95−p5; DIAGNOSTIC: IQR_g)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "02_spread_per_gene_per_org", iqr_df)


def fig_03_joint_m_iqr(iqr_df: pd.DataFrame) -> None:
    orgs = sorted(iqr_df["orgId"].unique())
    rows, cols = _grid_dims(len(orgs))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_1d(axes).flatten()
    for ax, org in zip(axes, orgs):
        sub = iqr_df[iqr_df["orgId"] == org]
        ax.hexbin(sub["m_g"], sub["iqr_g"], gridsize=24, cmap="viridis", mincnt=1)
        ax.set_title(org, fontsize=8)
        ax.set_xlabel("m_g")
        ax.set_ylabel("IQR_g")
    for ax in axes[len(orgs):]:
        ax.set_visible(False)
    fig.suptitle("R0-A.03 — Joint (m_g, IQR_g) per organism", fontsize=11)
    fig.tight_layout()
    _save(fig, "03_joint_m_iqr", iqr_df[["orgId", "gene_key", "m_g", "iqr_g"]])


def fig_04_replicate_noise(noise_df: pd.DataFrame,
                            noise_summary: pd.DataFrame,
                            all_orgs: list[str] | None = None) -> None:
    if noise_df.empty:
        log.warning("noise_df empty — skipping fig 04")
        return
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    orgs = sorted(noise_df["orgId"].unique())
    data = [noise_df[noise_df["orgId"] == o]["spearman"].to_numpy() for o in orgs]
    ax.boxplot(data, labels=orgs, showfliers=False)
    ax.set_xticklabels(orgs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("cross-replicate within-gene Spearman")
    title = "R0-A.04 — Replicate noise floor (per org)"
    if all_orgs is not None:
        missing = sorted(set(all_orgs) - set(orgs))
        if missing:
            title += (f"\n{len(missing)} of {len(all_orgs)} orgs absent "
                      f"(no replicate (expDesc, media) groups; default to IQR floor 0.10 in R-LOCK-1)")
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    _save(fig, "04_replicate_noise_floor", noise_summary)


def fig_05_snr(snr_df: pd.DataFrame, all_orgs: list[str] | None = None) -> None:
    """Per-org distribution of tail-based SNR (PRIMARY per R-LOCK-1)."""
    if "snr_tail" not in snr_df.columns or snr_df["snr_tail"].isna().all():
        log.warning("snr_tail empty — skipping fig 05")
        return
    orgs_with_snr = sorted(snr_df.loc[snr_df["snr_tail"].notna(), "orgId"].unique())
    rows, cols = _grid_dims(len(orgs_with_snr))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_1d(axes).flatten()
    for ax, org in zip(axes, orgs_with_snr):
        sub = snr_df[(snr_df["orgId"] == org) & snr_df["snr_tail"].notna()]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.hist(np.clip(sub["snr_tail"], 0, np.quantile(sub["snr_tail"], 0.99)),
                bins=40, color="#a73", edgecolor="black", linewidth=0.3)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=0.7)
        ax.set_title(f"{org} (frac>1={float((sub['snr_tail'] > 1).mean()):.2f})", fontsize=8)
        ax.set_xlabel("tail_g / (1−r_rep)")
    for ax in axes[len(orgs_with_snr):]:
        ax.set_visible(False)
    suptitle = "R0-A.05 — Signal-to-noise (tail_g basis) per gene, per organism"
    if all_orgs is not None:
        missing = sorted(set(all_orgs) - set(orgs_with_snr))
        if missing:
            suptitle += (f"  ({len(missing)} of {len(all_orgs)} orgs absent — "
                         f"no replicate data; default to tail floor in R-LOCK-1)")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    _save(fig, "05_signal_to_noise",
          snr_df[["orgId", "gene_key", "tail_g", "iqr_g", "snr_tail", "snr_iqr"]]
            .dropna(subset=["snr_tail"]))


def fig_06_eligibility_frontier(frontier: pd.DataFrame) -> None:
    """Heatmap: rows=(m_min, threshold) pairs, cols=orgs, cell=frac eligible.

    The `threshold` is on the metric column (tail_g for primary; iqr_g if
    diagnostic frontier was generated).
    """
    if frontier.empty:
        log.warning("frontier empty — skipping fig 06")
        return
    metric_label = frontier["metric"].iloc[0] if "metric" in frontier.columns else "iqr_g"
    pivot = (frontier.assign(_lab=lambda d: d["m_min"].astype(str)
                              + ", " + d["threshold"].map(lambda x: f"{x:.2f}"))
             .pivot(index="_lab", columns="orgId", values="frac_eligible"))
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.4 * len(pivot.columns)),
                                          max(6, 0.25 * len(pivot.index))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("orgId")
    ax.set_ylabel(f"(m_min, {metric_label}_min)")
    ax.set_title(f"R0-A.06 — Eligibility frontier on {metric_label} (fraction eligible)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="frac_eligible")
    fig.tight_layout()
    _save(fig, "06_eligibility_frontier", frontier)


def fig_07_low_iqr_overlap(jaccard: pd.DataFrame) -> None:
    if jaccard.empty:
        log.warning("jaccard empty — skipping fig 07")
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(jaccard.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(jaccard.columns)))
    ax.set_xticklabels(jaccard.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(jaccard.index)))
    ax.set_yticklabels(jaccard.index, fontsize=7)
    ax.set_title("R0-A.07 — Jaccard of bottom-decile-IQR gene sets across orgs")
    fig.colorbar(im, ax=ax, shrink=0.7, label="jaccard")
    fig.tight_layout()
    _save(fig, "07_low_iqr_overlap",
          jaccard.reset_index().melt(id_vars=jaccard.index.name or "index",
                                      var_name="other_org", value_name="jaccard"))


def fig_08_experiment_structure(exp_struct: pd.DataFrame) -> None:
    if exp_struct.empty:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    df = exp_struct.sort_values("n_experiments", ascending=True)
    ax1.barh(df["orgId"], df["n_experiments"], color="#37a", label="expNames")
    ax1.barh(df["orgId"], df["n_conditions"], color="#7a3", alpha=0.7, label="conditions")
    ax1.set_xlabel("count")
    ax1.set_title("Experiments (blue) vs distinct conditions (green) per org")
    ax1.tick_params(axis="y", labelsize=6)
    ax1.legend()
    ax2.barh(df["orgId"], df["exp_per_cond_p50"], color="#a73")
    ax2.set_xlabel("median expNames per condition (replicate depth, p50)")
    ax2.set_title("Replicate depth (p50)")
    ax2.tick_params(axis="y", labelsize=6)
    ax3.barh(df["orgId"], df["frac_cond_with_replicates"], color="#73a")
    ax3.set_xlabel("frac conditions with ≥2 expName replicates")
    ax3.set_xlim(0, 1)
    ax3.set_title("Replicate coverage")
    ax3.tick_params(axis="y", labelsize=6)
    fig.suptitle("R0-A.08 — Experiment structure per org")
    fig.tight_layout()
    _save(fig, "08_experiment_structure", exp_struct)


def fig_09_condition_discriminability(disc: pd.DataFrame) -> None:
    if disc.empty:
        return
    orgs = sorted(disc["orgId"].unique())
    rows, cols = _grid_dims(len(orgs))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_1d(axes).flatten()
    for ax, org in zip(axes, orgs):
        sub = disc[disc["orgId"] == org]
        ax.hist(sub["cond_iqr"], bins=30, color="#73a", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{org} (n_cond={len(sub)})", fontsize=8)
        ax.set_xlabel("cross-gene IQR of fit")
    for ax in axes[len(orgs):]:
        ax.set_visible(False)
    fig.suptitle("R0-A.09 — Condition discriminability (IQR of fit across genes)", fontsize=11)
    fig.tight_layout()
    _save(fig, "09_condition_discriminability", disc)


def fig_10_split_feasibility(split_df: pd.DataFrame) -> None:
    if split_df.empty:
        return
    orgs = sorted(split_df["orgId"].unique())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, 0.25 * len(orgs)), 9), sharex=True)
    for frac, sub in split_df.groupby("holdout_fraction"):
        sub = sub.set_index("orgId").reindex(orgs)
        ax1.plot(orgs, sub["n_genes_val_m_ge_5"],
                 marker="o", linestyle="-", label=f"frac={frac:.2f}")
        ax2.plot(orgs, sub["median_val_m_per_gene"],
                 marker="o", linestyle="-", label=f"frac={frac:.2f}")
    ax1.set_ylabel("# genes with val m ≥ 5")
    ax1.set_title("R0-A.10 — Split feasibility (fraction-based holdout, replicate-grouped)")
    ax1.axhline(100, color="red", linestyle="--", linewidth=0.7, label="threshold=100")
    ax1.legend(title="holdout frac", fontsize=7)
    ax2.set_ylabel("median val m per gene")
    ax2.set_xlabel("orgId")
    ax2.tick_params(axis="x", rotation=45, labelsize=6)
    for lbl in ax2.get_xticklabels():
        lbl.set_horizontalalignment("right")
    fig.tight_layout()
    _save(fig, "10_split_feasibility", split_df)


def fig_11_expgroup_coverage(coverage: pd.DataFrame,
                              summary: pd.DataFrame) -> None:
    if coverage.empty or summary.empty:
        log.warning("expgroup_coverage empty — skipping fig 11")
        return
    # Two panels: (1) distinct groups per org, (2) # groups passing strat threshold
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, 0.25 * len(summary))))
    df = summary.sort_values("n_expgroups", ascending=True)
    # Color encodes the actual feasibility verdict:
    #   blue  = feasible (n_groups ≥ 2 AND ≥ 2 groups have ≥ 10 exps)
    #   red   = NOT feasible (fails one of the two)
    colors = ["#37a" if f else "#a33" for f in df["stratification_feasible"]]
    ax1.barh(df["orgId"], df["n_expgroups"], color=colors)
    ax1.set_xlabel("# distinct expGroups")
    ax1.set_title("expGroups per org\n(blue=feasible, red=NOT feasible at 20% holdout)")
    ax1.tick_params(axis="y", labelsize=6)
    ax1.axvline(2, color="black", linestyle=":", linewidth=0.7, label="≥ 2 needed (gate 1)")
    ax1.legend(loc="lower right", fontsize=7)
    # Right panel: # groups that themselves pass the size threshold (gate 2).
    # This matches the feasibility rule exactly — colors and threshold now agree.
    ax2.barh(df["orgId"], df["n_groups_passing_strat_threshold"], color=colors)
    ax2.set_xlabel("# expGroups with ≥ 10 experiments")
    ax2.set_title("Groups passing size threshold per org\n"
                  "(blue=feasible, red=NOT feasible; line at 2 = the feasibility gate)")
    ax2.tick_params(axis="y", labelsize=6)
    ax2.axvline(2, color="black", linestyle=":", linewidth=0.7, label="≥ 2 needed (gate 2)")
    ax2.legend(loc="lower right", fontsize=7)
    fig.suptitle("R0-A.11 — expGroup coverage per org (R-LOCK-2 stratification feasibility)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "11_expgroup_coverage", summary)
    # also save the long-form coverage csv
    coverage.to_csv(FIGURES_DIR / "11_expgroup_coverage_long.csv", index=False)
