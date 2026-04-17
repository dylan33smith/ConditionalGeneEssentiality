#!/usr/bin/env python3
"""Phase 0 elucidation: figures + SQL summaries + media workbook audit.

Run from repo root:
  python data_analysis/run_phase0.py

Outputs:
  figures/phase0/*.png  (01–06 from DB sample; 07+ from canonical Parquet + media_composition_v2.xlsx)
  data_analysis/outputs/media_composition_audit.md
  data_analysis/outputs/phase0_summary.json
"""

from __future__ import annotations

from pathlib import Path

import json
import sqlite3
import sys
import warnings
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

from paths import (
    EMBEDDINGS_DIR,
    FEBA_DB,
    FIGURES_PHASE0,
    OUTPUT_DIR,
    REPO_ROOT,
)

SAMPLE_MOD = 313  # ~88k rows from ~27.4M

CANON_EXPERIMENTS_PARQUET = REPO_ROOT / "data" / "derived" / "canonical" / "v0" / "experiments.parquet"
CANON_FITNESS_LONG_PARQUET = REPO_ROOT / "data" / "derived" / "canonical" / "v0" / "fitness_experiment_long.parquet"
MEDIA_XLSX_V2 = REPO_ROOT / "data" / "media_composition_v2.xlsx"
COND_ENCODING_PARQUET = REPO_ROOT / "data" / "derived" / "condition_encoding" / "v0" / "experiments_condition.parquet"


def _norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_dirs() -> None:
    FIGURES_PHASE0.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    if not FEBA_DB.is_file():
        raise FileNotFoundError(f"Missing database: {FEBA_DB}")
    conn = sqlite3.connect(f"file:{FEBA_DB.as_posix()}?mode=ro", uri=True)
    return conn


def load_experiments(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM Experiment", conn)


def load_fitness_sample(conn: sqlite3.Connection) -> pd.DataFrame:
    q = f"""
    SELECT gf.fit AS fit, gf.t AS t, gf.orgId AS orgId, gf.locusId AS locusId,
           gf.expName AS expName, e.media AS media, e.cor12 AS cor12, e.expGroup AS expGroup
    FROM GeneFitness gf
    INNER JOIN Experiment e ON gf.orgId = e.orgId AND gf.expName = e.expName
    WHERE (gf.rowid % {SAMPLE_MOD}) = 0
    """
    return pd.read_sql_query(q, conn)


def load_n_experiments_per_gene(conn: sqlite3.Connection) -> pd.Series:
    q = """
    SELECT n_experiments
    FROM (
        SELECT orgId, locusId, COUNT(DISTINCT expName) AS n_experiments
        FROM GeneFitness
        GROUP BY orgId, locusId
    )
    """
    df = pd.read_sql_query(q, conn)
    return df["n_experiments"]


def plot_histograms(sample: pd.DataFrame, experiments: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(sample["fit"], bins=80, kde=False, ax=axes[0], color="steelblue")
    axes[0].set_title("fit (systematic sample)")
    axes[0].set_xlabel("fit")
    sns.histplot(sample["t"].abs(), bins=80, kde=False, ax=axes[1], color="coral")
    axes[1].set_title("|t| (same sample)")
    axes[1].set_xlabel("|t|")
    fig.tight_layout()
    fig.savefig(FIGURES_PHASE0 / "01_histogram_fit_and_abs_t.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(experiments["cor12"].dropna(), bins=60, kde=False, ax=ax, color="seagreen")
    ax.set_title("cor12 (one value per experiment)")
    ax.set_xlabel("cor12")
    fig.tight_layout()
    fig.savefig(FIGURES_PHASE0 / "02_histogram_cor12_experiments.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(
        sample["cor12"],
        sample["fit"].abs(),
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("cor12 (row-level, from experiment)")
    ax.set_ylabel("|fit|")
    ax.set_title("cor12 vs |fit| (sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_PHASE0 / "03_hexbin_cor12_vs_abs_fit.png", dpi=150)
    plt.close(fig)


def plot_rows_per_org(sample: pd.DataFrame) -> None:
    counts = sample.groupby("orgId").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.15)))
    counts.plot(kind="barh", ax=ax, color="slategray")
    ax.set_title("Rows in systematic fitness sample by orgId")
    ax.set_xlabel("row count")
    fig.tight_layout()
    fig.savefig(FIGURES_PHASE0 / "04_rows_per_org_sample.png", dpi=150)
    plt.close(fig)


def plot_cdf_n_experiments_per_gene(n_per: pd.Series) -> None:
    x = np.sort(n_per.to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="darkviolet")
    ax.set_xlabel("# distinct experiments per gene (orgId, locusId)")
    ax.set_ylabel("CDF")
    ax.set_title("Sparsity: experiments per gene")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_PHASE0 / "05_cdf_experiments_per_gene.png", dpi=150)
    plt.close(fig)


def benchmark_mask(experiments: pd.DataFrame) -> pd.Series:
    """v1 heuristic from PROJECT_RESTART_PLAN §12.1 (starting point)."""
    m = experiments["media"].fillna("")
    prefix_ok = m.str.startswith("LB") | m.str.startswith("RCH2") | m.str.startswith("M9")
    cor_ok = experiments["cor12"].astype(float) >= 0.2
    not_plant = experiments["expGroup"].fillna("") != "plant"
    not_pdb = m != "Potato Dextrose Broth"
    return prefix_ok & cor_ok & not_plant & not_pdb


def connected_media_table(experiments: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    sub = experiments.loc[mask, ["orgId", "media"]].drop_duplicates()
    deg = sub.groupby("media")["orgId"].nunique().reset_index(name="n_organisms")
    return deg.sort_values("n_organisms", ascending=False)


def plot_media_org_bipartite(experiments: pd.DataFrame, mask: pd.Series, max_media: int = 40) -> None:
    sub = experiments.loc[mask, ["orgId", "media"]].drop_duplicates()
    deg = sub.groupby("media")["orgId"].nunique().sort_values(ascending=False)
    top_media = set(deg.head(max_media).index)
    edges = sub[sub["media"].isin(top_media)]
    B = nx.Graph()
    for _, r in edges.iterrows():
        o, m = r["orgId"], r["media"]
        B.add_edge(f"org:{o}", f"med:{m}")
    pos = nx.spring_layout(B, seed=42, k=0.35)
    fig, ax = plt.subplots(figsize=(14, 10))
    org_nodes = [n for n in B if n.startswith("org:")]
    med_nodes = [n for n in B if n.startswith("med:")]
    nx.draw_networkx_nodes(B, pos, nodelist=org_nodes, node_color="tab:blue", node_size=80, ax=ax, alpha=0.85)
    nx.draw_networkx_nodes(B, pos, nodelist=med_nodes, node_color="tab:orange", node_size=120, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(B, pos, alpha=0.25, width=0.5, ax=ax)
    ax.set_title(f"Organism–media graph (benchmark mask, top {max_media} media by # organisms)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_PHASE0 / "06_bipartite_org_media_benchmark.png", dpi=150)
    plt.close(fig)


def variance_decomposition_table(sample: pd.DataFrame) -> dict:
    """Sequential sum-of-squares on systematic sample (exploratory, not full mixed model)."""
    y = sample["fit"].to_numpy(dtype=float)
    mu = float(np.mean(y))
    ss_tot = float(np.sum((y - mu) ** 2))

    g_org = sample.groupby("orgId")["fit"]
    mean_org = g_org.mean()
    cnt_org = g_org.count()
    ss_org = float(np.sum(cnt_org * (mean_org - mu) ** 2))

    pred_org = sample["orgId"].map(mean_org).to_numpy()
    resid = y - pred_org

    top_exp = sample["expName"].value_counts().head(400).index
    sample2 = sample.copy()
    sample2["exp_bucket"] = np.where(sample2["expName"].isin(top_exp), sample2["expName"], "__other_exp__")
    g_exp = sample2.groupby("exp_bucket")["fit"]
    mean_exp = g_exp.mean()
    cnt_exp = g_exp.count()
    mu_r = float(np.mean(resid))
    ss_exp_on_resid = float(np.sum(cnt_exp * (mean_exp - mu_r) ** 2))

    return {
        "n_rows_sample": int(len(sample)),
        "ss_total": ss_tot,
        "ss_orgId_vs_grand_mean": ss_org,
        "frac_ss_orgId": ss_org / ss_tot if ss_tot else None,
        "ss_top400_exp_on_org_residuals": ss_exp_on_resid,
        "frac_ss_top400_exp_on_residuals_of_ss_tot": ss_exp_on_resid / ss_tot if ss_tot else None,
        "note": "Sequential exploratory decomposition on systematic sample; exp buckets top-400 + other.",
    }


def _load_media_components_v2() -> pd.DataFrame | None:
    if not MEDIA_XLSX_V2.is_file():
        print(f"Skipping Parquet/v2 figures: missing {MEDIA_XLSX_V2}", file=sys.stderr)
        return None
    mc = pd.read_excel(MEDIA_XLSX_V2, sheet_name="Media_Components")
    mc["Media_n"] = mc["Media"].map(_norm_text)
    mc["Component_n"] = mc["Component"].map(_norm_text)
    return mc[(mc["Media_n"] != "") & (mc["Component_n"] != "")][["Media_n", "Component_n"]].drop_duplicates()


def _experiment_component_table() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Canonical experiments expanded to (experiment × component) rows."""
    if not CANON_EXPERIMENTS_PARQUET.is_file():
        print(f"Skipping Parquet/v2 figures: missing {CANON_EXPERIMENTS_PARQUET}", file=sys.stderr)
        return None
    mc = _load_media_components_v2()
    if mc is None:
        return None
    canon = pd.read_parquet(CANON_EXPERIMENTS_PARQUET, columns=["orgId", "expName", "media"])
    canon["orgId_n"] = canon["orgId"].map(_norm_text)
    canon["expName_n"] = canon["expName"].map(_norm_text)
    canon["media_n"] = canon["media"].map(_norm_text)
    exp_comp = canon[["orgId_n", "expName_n", "media_n"]].copy()
    exp_comp = exp_comp[exp_comp["media_n"] != ""]
    media_with_components = set(mc["Media_n"].unique())
    exp_comp = exp_comp[exp_comp["media_n"].isin(media_with_components)]
    exp_comp = exp_comp.merge(mc, left_on="media_n", right_on="Media_n", how="inner")
    exp_comp = exp_comp[exp_comp["Component_n"].notna()].copy()
    component_coverage = (
        exp_comp.groupby("Component_n", as_index=False)
        .agg(
            n_organisms=("orgId_n", "nunique"),
            n_experiments=("expName_n", "count"),
            n_media=("media_n", "nunique"),
        )
        .sort_values(["n_organisms", "n_experiments"], ascending=False)
    )
    return exp_comp, component_coverage


def plot_chemical_overlap_figures() -> list[str]:
    """07–09: all chemical components on x-axis; tick labels omitted for density."""
    out: list[str] = []
    tbl = _experiment_component_table()
    if tbl is None:
        return out
    exp_comp, component_coverage = tbl
    component_coverage.to_csv(OUTPUT_DIR / "chemical_component_cross_species_coverage.csv", index=False)
    out.append(str(OUTPUT_DIR / "chemical_component_cross_species_coverage.csv"))

    plot_df = component_coverage.sort_values(["n_organisms", "n_experiments"], ascending=False).reset_index(drop=True)
    n_comp = len(plot_df)
    fig_w = max(14.0, 0.06 * n_comp)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    ax.bar(range(n_comp), plot_df["n_organisms"], color="tab:cyan", alpha=0.85, width=1.0)
    ax.set_ylabel("Number of organisms")
    ax.set_xlabel("Chemical component (all; sorted by organism count; labels omitted)")
    ax.set_title("Cross-species overlap per chemical (by organisms)")
    ax.set_xticks(range(n_comp))
    ax.set_xticklabels([])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p07 = FIGURES_PHASE0 / "07_component_organism_overlap_all.png"
    fig.savefig(p07, dpi=150)
    plt.close(fig)
    out.append(str(p07))

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        component_coverage["n_organisms"],
        component_coverage["n_experiments"],
        alpha=0.7,
        s=25,
        color="tab:blue",
    )
    ax.set_xlabel("# organisms containing component")
    ax.set_ylabel("# experiments containing component")
    ax.set_title("Chemical component support across organisms/experiments")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p09 = FIGURES_PHASE0 / "09_component_support_scatter.png"
    fig.savefig(p09, dpi=150)
    plt.close(fig)
    out.append(str(p09))

    co = (
        exp_comp.groupby(["Component_n", "orgId_n"], as_index=False)
        .agg(n_experiments=("expName_n", "count"))
    )
    totals = co.groupby("Component_n", as_index=False)["n_experiments"].sum().sort_values(
        "n_experiments", ascending=False
    )
    component_order = totals["Component_n"].tolist()
    pivot = co.pivot_table(
        index="Component_n", columns="orgId_n", values="n_experiments", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(component_order).fillna(0)
    org_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[org_order]
    top_orgs = 12
    if pivot.shape[1] > top_orgs:
        keep = org_order[:top_orgs]
        other = [c for c in org_order if c not in keep]
        pivot_plot = pivot[keep].copy()
        pivot_plot["__Other_organisms__"] = pivot[other].sum(axis=1)
    else:
        pivot_plot = pivot.copy()
    pivot_plot.to_csv(OUTPUT_DIR / "component_by_organism_experiment_counts_all.csv")
    out.append(str(OUTPUT_DIR / "component_by_organism_experiment_counts_all.csv"))

    n_stack = pivot_plot.shape[0]
    fig_w2 = max(16.0, 0.06 * n_stack)
    fig2, ax2 = plt.subplots(figsize=(fig_w2, 6.5))
    pivot_plot.plot(kind="bar", stacked=True, ax=ax2, width=1.0, linewidth=0)
    ax2.set_xlabel("Chemical component (all; sorted by experiment count; labels omitted)")
    ax2.set_ylabel("# experiments containing component")
    ax2.set_title("Cross-species overlap: experiment support per chemical (stacked by organism)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(title="orgId", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    ax2.set_xticklabels([])
    fig2.tight_layout()
    p08 = FIGURES_PHASE0 / "08_component_overlap_stacked_by_organism_all.png"
    fig2.savefig(p08, dpi=150)
    plt.close(fig2)
    out.append(str(p08))
    return out


def plot_media_and_condition_diversity_figures() -> list[str]:
    """10–12: media richness, encoding coverage by org, condition diversity."""
    out: list[str] = []
    mc = _load_media_components_v2()
    if mc is None or not CANON_EXPERIMENTS_PARQUET.is_file():
        return out

    richness = (
        mc.groupby("Media_n", as_index=False)
        .agg(n_components=("Component_n", "nunique"))
        .sort_values("n_components", ascending=False)
    )
    richness.to_csv(OUTPUT_DIR / "media_component_richness.csv", index=False)
    out.append(str(OUTPUT_DIR / "media_component_richness.csv"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        richness["n_components"],
        bins=min(40, max(10, int(np.sqrt(len(richness))))),
        color="tab:blue",
        alpha=0.85,
    )
    ax.set_xlabel("Number of components in media")
    ax.set_ylabel("Count of media")
    ax.set_title("Component richness distribution across media")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p10 = FIGURES_PHASE0 / "10_media_component_richness_hist.png"
    fig.savefig(p10, dpi=150)
    plt.close(fig)
    out.append(str(p10))

    canon = pd.read_parquet(CANON_EXPERIMENTS_PARQUET, columns=["orgId", "expName", "media"])
    canon["orgId_n"] = canon["orgId"].map(_norm_text)
    canon["expName_n"] = canon["expName"].map(_norm_text)
    canon["media_n"] = canon["media"].map(_norm_text)
    media_with_components = set(mc["Media_n"].unique())
    canon["has_media_label"] = canon["media_n"] != ""
    canon["has_component_mapping"] = canon["media_n"].isin(media_with_components)
    canon["covered_for_chem_encoding"] = canon["has_media_label"] & canon["has_component_mapping"]
    coverage_by_org = (
        canon.groupby("orgId_n", as_index=False)
        .agg(
            n_experiments=("expName_n", "count"),
            n_has_media_label=("has_media_label", "sum"),
            n_covered_for_chem_encoding=("covered_for_chem_encoding", "sum"),
        )
    )
    coverage_by_org["pct_covered_for_chem_encoding"] = (
        100.0 * coverage_by_org["n_covered_for_chem_encoding"] / coverage_by_org["n_experiments"].clip(lower=1)
    )
    coverage_by_org = coverage_by_org.sort_values("pct_covered_for_chem_encoding", ascending=True)
    coverage_by_org.to_csv(OUTPUT_DIR / "chemical_encoding_coverage_by_org.csv", index=False)
    out.append(str(OUTPUT_DIR / "chemical_encoding_coverage_by_org.csv"))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        coverage_by_org["orgId_n"],
        coverage_by_org["pct_covered_for_chem_encoding"],
        color="tab:green",
        alpha=0.85,
    )
    ax.set_ylabel("% experiments mappable to components")
    ax.set_xlabel("orgId")
    ax.set_title("Chemical encoding coverage by organism (canonical experiments)")
    ax.set_xticks(range(len(coverage_by_org)))
    ax.set_xticklabels(coverage_by_org["orgId_n"], rotation=80, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p11 = FIGURES_PHASE0 / "11_chemical_encoding_coverage_by_org.png"
    fig.savefig(p11, dpi=150)
    plt.close(fig)
    out.append(str(p11))

    if not COND_ENCODING_PARQUET.is_file():
        return out
    ce = pd.read_parquet(COND_ENCODING_PARQUET)
    cat_cols = [c for c in ce.columns if c.startswith("ce_cat_")]
    cont_cols = [c for c in ce.columns if c.startswith("ce_cont_")]
    cond_cols = cat_cols + cont_cols
    ce["orgId_n"] = ce["orgId"].map(_norm_text)
    for c in cont_cols:
        ce[c] = pd.to_numeric(ce[c], errors="coerce").fillna(0.0).round(6)
    ce["condition_signature"] = ce[cond_cols].astype(str).agg("|".join, axis=1)
    cond_div = (
        ce.groupby("orgId_n", as_index=False)
        .agg(
            n_experiments=("expName", "count"),
            n_unique_condition_signatures=("condition_signature", "nunique"),
            n_unique_media_codes=("ce_cat_media", "nunique"),
            n_unique_condition1_codes=("ce_cat_condition_1", "nunique"),
        )
    )
    cond_div["avg_experiments_per_condition_signature"] = (
        cond_div["n_experiments"] / cond_div["n_unique_condition_signatures"].clip(lower=1)
    )
    cond_div = cond_div.sort_values("n_unique_condition_signatures", ascending=False)
    cond_div.to_csv(OUTPUT_DIR / "condition_diversity_by_org.csv", index=False)
    out.append(str(OUTPUT_DIR / "condition_diversity_by_org.csv"))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cond_div["orgId_n"], cond_div["n_unique_condition_signatures"], color="tab:orange", alpha=0.85)
    ax.set_ylabel("Unique condition signatures")
    ax.set_xlabel("orgId")
    ax.set_title("Condition diversity by organism (encoded table)")
    ax.set_xticks(range(len(cond_div)))
    ax.set_xticklabels(cond_div["orgId_n"], rotation=80, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p12 = FIGURES_PHASE0 / "12_condition_diversity_by_org.png"
    fig.savefig(p12, dpi=150)
    plt.close(fig)
    out.append(str(p12))
    return out


def plot_embedding_coverage_figures() -> list[str]:
    """13–14: ProteomeLM embedding coverage vs canonical fitness long."""
    out: list[str] = []
    try:
        import torch
    except ImportError:
        print("Skipping embedding coverage figures: torch not installed", file=sys.stderr)
        return out
    if not CANON_FITNESS_LONG_PARQUET.is_file() or not EMBEDDINGS_DIR.is_dir():
        print("Skipping embedding coverage figures: missing Parquet or embeddings dir", file=sys.stderr)
        return out

    fit = pd.read_parquet(CANON_FITNESS_LONG_PARQUET, columns=["orgId", "gene_key", "fit"])
    fit = fit[fit["fit"].notna()].copy()
    fit["orgId"] = fit["orgId"].astype(str)
    fit["gene_key"] = fit["gene_key"].astype(str)

    emb_by_org: dict[str, set[str]] = {}
    for pt in sorted(EMBEDDINGS_DIR.glob("*_proteomelm.pt")):
        org = pt.name.replace("_proteomelm.pt", "")
        try:
            blob = torch.load(pt, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(pt, map_location="cpu")
        labels = blob.get("group_labels")
        if labels is None:
            continue
        emb_by_org[org] = {str(x) for x in labels}

    fit["has_embedding"] = [
        gk in emb_by_org.get(org, set()) for org, gk in zip(fit["orgId"], fit["gene_key"], strict=False)
    ]
    row_summary = {
        "n_rows_with_fit": int(len(fit)),
        "n_rows_with_embedding": int(fit["has_embedding"].sum()),
        "n_rows_missing_embedding": int((~fit["has_embedding"]).sum()),
    }
    row_summary["pct_rows_missing_embedding"] = (
        100.0 * row_summary["n_rows_missing_embedding"] / max(1, row_summary["n_rows_with_fit"])
    )

    genes = fit[["orgId", "gene_key"]].drop_duplicates().copy()
    genes["has_embedding"] = [
        gk in emb_by_org.get(org, set()) for org, gk in zip(genes["orgId"], genes["gene_key"], strict=False)
    ]
    gene_summary = {
        "n_unique_org_gene_keys": int(len(genes)),
        "n_unique_with_embedding": int(genes["has_embedding"].sum()),
        "n_unique_missing_embedding": int((~genes["has_embedding"]).sum()),
    }
    gene_summary["pct_unique_missing_embedding"] = (
        100.0 * gene_summary["n_unique_missing_embedding"] / max(1, gene_summary["n_unique_org_gene_keys"])
    )

    per_org = (
        genes.groupby("orgId", as_index=False)
        .agg(
            n_unique_gene_keys=("gene_key", "count"),
            n_unique_with_embedding=("has_embedding", "sum"),
        )
    )
    per_org["n_unique_missing_embedding"] = per_org["n_unique_gene_keys"] - per_org["n_unique_with_embedding"]
    per_org["pct_unique_missing_embedding"] = (
        100.0 * per_org["n_unique_missing_embedding"] / per_org["n_unique_gene_keys"].clip(lower=1)
    )
    per_org = per_org.sort_values("pct_unique_missing_embedding", ascending=False)

    summary = {"row_level": row_summary, "gene_level": gene_summary}
    (OUTPUT_DIR / "embedding_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out.append(str(OUTPUT_DIR / "embedding_coverage_summary.json"))
    per_org.to_csv(OUTPUT_DIR / "embedding_coverage_by_org.csv", index=False)
    out.append(str(OUTPUT_DIR / "embedding_coverage_by_org.csv"))
    genes.loc[~genes["has_embedding"]].sort_values(["orgId", "gene_key"]).to_csv(
        OUTPUT_DIR / "missing_embedding_gene_keys.csv", index=False
    )
    out.append(str(OUTPUT_DIR / "missing_embedding_gene_keys.csv"))

    plot_pct = per_org.head(20).copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_pct["orgId"], plot_pct["pct_unique_missing_embedding"], color="tab:purple", alpha=0.85)
    ax.set_ylabel("% unique gene keys missing embedding")
    ax.set_xlabel("orgId (top 20 by missing %)")
    ax.set_title("Embedding coverage gap by organism")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(range(len(plot_pct)))
    ax.set_xticklabels(plot_pct["orgId"], rotation=70, ha="right")
    fig.tight_layout()
    p13 = FIGURES_PHASE0 / "13_embedding_coverage_gap_top20.png"
    fig.savefig(p13, dpi=150)
    plt.close(fig)
    out.append(str(p13))

    top_abs = per_org.sort_values("n_unique_missing_embedding", ascending=False).head(20)
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.bar(top_abs["orgId"], top_abs["n_unique_missing_embedding"], color="tab:red", alpha=0.85)
    ax2.set_ylabel("Missing unique (orgId, gene_key) embeddings")
    ax2.set_xlabel("orgId (top 20 by absolute missing count)")
    ax2.set_title("Absolute missing embedding counts by organism")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks(range(len(top_abs)))
    ax2.set_xticklabels(top_abs["orgId"], rotation=70, ha="right")
    fig2.tight_layout()
    p14 = FIGURES_PHASE0 / "14_embedding_missing_counts_top20.png"
    fig2.savefig(p14, dpi=150)
    plt.close(fig2)
    out.append(str(p14))
    return out


def excel_experiment_coverage(
    experiments: pd.DataFrame, workbook: Path = MEDIA_XLSX_V2
) -> dict:
    """How many DB experiments appear in the workbook `Experiments` sheet (orgId + name)."""
    if not workbook.is_file():
        return {
            "workbook": str(workbook),
            "error": "workbook_missing",
            "n_experiments_db": int(len(experiments[["orgId", "expName"]].drop_duplicates())),
            "n_rows_excel_experiments": 0,
            "matched_on_orgId_expName": 0,
            "unmatched_db_experiments": 0,
            "media_string_mismatches_on_matches": 0,
        }
    ex = pd.read_excel(workbook, sheet_name="Experiments", header=0)
    ex = ex.rename(columns={"name": "expName"})
    key = ["orgId", "expName"]
    db = experiments[key + ["media"]].drop_duplicates()
    merged = db.merge(ex[key + ["Media"]], on=key, how="left")
    matched = int(merged["Media"].notna().sum())
    mismatch = merged.dropna(subset=["Media"])
    mismatch = mismatch[mismatch["media"].astype(str) != mismatch["Media"].astype(str)]
    try:
        wb_rel = str(workbook.relative_to(REPO_ROOT))
    except ValueError:
        wb_rel = str(workbook)
    return {
        "workbook": wb_rel,
        "n_experiments_db": int(len(db)),
        "n_rows_excel_experiments": int(len(ex)),
        "matched_on_orgId_expName": matched,
        "unmatched_db_experiments": int(len(db) - matched),
        "media_string_mismatches_on_matches": int(len(mismatch)),
    }


def write_media_audit_md() -> str:
    audit_wb = MEDIA_XLSX_V2
    if not audit_wb.is_file():
        out = OUTPUT_DIR / "media_composition_audit.md"
        out.write_text(
            f"# Media composition workbook audit (Phase 0)\n\n"
            f"**Missing file:** `{audit_wb.relative_to(REPO_ROOT)}` — audit not generated.\n",
            encoding="utf-8",
        )
        return str(out)
    wb_rel = str(audit_wb.relative_to(REPO_ROOT))
    xl = pd.ExcelFile(audit_wb)
    lines = [
        "# Media composition workbook audit (Phase 0)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} UTC",
        f"File: `{wb_rel}` (authoritative for this audit; v2 expanded workbook).",
        "",
        "## Sheet inventory",
        "",
        "| Index | Sheet name | Rows | Cols |",
        "| ----- | ---------- | ---- | ---- |",
    ]
    for i, name in enumerate(xl.sheet_names):
        df = pd.read_excel(audit_wb, sheet_name=name, header=None)
        lines.append(f"| {i} | {name} | {df.shape[0]} | {df.shape[1]} |")
    conn_x = connect()
    try:
        exp_db = pd.read_sql_query("SELECT orgId, expName, media FROM Experiment", conn_x)
    finally:
        conn_x.close()
    cov = excel_experiment_coverage(exp_db, audit_wb)
    lines.extend(
        [
            "",
            "## DB vs `Experiments` sheet (join keys)",
            "",
            f"- Matched `(orgId, expName)` rows: **{cov['matched_on_orgId_expName']}** / {cov['n_experiments_db']} DB experiments.",
            f"- Unmatched DB experiments (no Excel row): **{cov['unmatched_db_experiments']}**.",
            f"- Media string mismatches where both present: **{cov['media_string_mismatches_on_matches']}**.",
            f"- Excel `Experiments` sheet rows: {cov['n_rows_excel_experiments']}.",
        ]
    )
    lines.extend(
        [
            "",
            "## Authoritative mapping",
            "",
            "- **Medium → components:** sheet **`Media_Components`**: columns *Media*, *Component*, *Concentration*, *Units*, …",
            "- **Experiment ↔ medium / metadata:** sheet **`Experiments`**: join to `Experiment` table on **`(orgId, name)`** ↔ **`(orgId, expName)`**; medium string in column *Media*.",
            "- **Composition** is on **`Media_Components`**, not derived from the `Experiments` row alone (that sheet names the medium; components are looked up by medium name).",
            "",
            "## `Media_Components` header row (row 0)",
            "",
        ]
    )
    mc = pd.read_excel(audit_wb, sheet_name="Media_Components", header=0)
    lines.append("| " + " | ".join(str(c) for c in mc.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(mc.columns)) + " |")
    for _, row in mc.head(15).iterrows():
        lines.append("| " + " | ".join(str(row[c])[:80] for c in mc.columns) + " |")
    lines.append("")
    lines.append(f"*({len(mc)} component rows total.)*")
    lines.append("")
    lines.append("## `Experiments` sheet columns (header row)")
    lines.append("")
    ex = pd.read_excel(audit_wb, sheet_name="Experiments", header=0)
    lines.append("- " + "\n- ".join(f"`{c}`" for c in ex.columns))
    lines.append("")
    text = "\n".join(lines)
    out = OUTPUT_DIR / "media_composition_audit.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def embedding_spotcheck() -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    if not EMBEDDINGS_DIR.is_dir():
        return None
    pts = sorted(EMBEDDINGS_DIR.glob("*.pt"))
    if not pts:
        return None
    p = pts[0]
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(p, map_location="cpu")
    emb = d.get("embeddings")
    labels = d.get("group_labels")
    if emb is None:
        return {"file": p.name, "error": "no embeddings key"}
    return {
        "file": p.name,
        "embeddings_shape": list(emb.shape),
        "n_labels": len(labels) if labels is not None else None,
        "l2_mean": float(torch.linalg.norm(emb.float(), dim=1).mean()),
    }


def main() -> int:
    ensure_dirs()
    conn = connect()
    try:
        experiments = load_experiments(conn)
        sample = load_fitness_sample(conn)
        n_per = load_n_experiments_per_gene(conn)
    finally:
        conn.close()

    plot_histograms(sample, experiments)
    plot_rows_per_org(sample)
    plot_cdf_n_experiments_per_gene(n_per)

    mask = benchmark_mask(experiments)
    deg = connected_media_table(experiments, mask)
    deg.to_csv(OUTPUT_DIR / "connected_media_degree_benchmark.csv", index=False)
    plot_media_org_bipartite(experiments, mask)

    parquet_derived_outputs: list[str] = []
    parquet_derived_outputs.extend(plot_chemical_overlap_figures())
    parquet_derived_outputs.extend(plot_media_and_condition_diversity_figures())
    parquet_derived_outputs.extend(plot_embedding_coverage_figures())

    vd = variance_decomposition_table(sample)
    audit_path = write_media_audit_md()
    emb = embedding_spotcheck()
    xl_cov = excel_experiment_coverage(experiments, MEDIA_XLSX_V2)

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_experiments": int(len(experiments)),
        "n_fitness_sample_rows": int(len(sample)),
        "sample_mod": SAMPLE_MOD,
        "benchmark_experiments": int(mask.sum()),
        "benchmark_fitness_rows_estimate": None,
        "variance_decomposition": vd,
        "embedding_spotcheck": emb,
        "excel_experiment_sheet_coverage": xl_cov,
        "outputs": {
            "figures_dir": _rel(FIGURES_PHASE0),
            "media_audit_md": _rel(Path(audit_path)),
            "connected_media_csv": _rel(OUTPUT_DIR / "connected_media_degree_benchmark.csv"),
            "phase0_summary_json": _rel(OUTPUT_DIR / "phase0_summary.json"),
            "parquet_workbook_derived": [_rel(Path(p)) for p in parquet_derived_outputs],
        },
    }
    (OUTPUT_DIR / "phase0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
