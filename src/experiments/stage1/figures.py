"""Figure generation for S1.

Each function generates one numbered figure (PNG + sibling CSV) at the
location specified by REFACTORPLAN §11. Figures live under
research_log/figures/stage1/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import reporting as R

log = logging.getLogger(__name__)

FIG_DIR = Path("research_log/figures/stage1")


def fig_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


# ---------------------------------------------------------------------------
# Overlap (figs 01-04)
# ---------------------------------------------------------------------------

def fig_01_org_media_overlap_heatmap(org_media: pd.DataFrame) -> None:
    from src.experiments.stage1.analyses import pairwise_org_overlap
    overlap = pairwise_org_overlap(org_media, mode="shared")
    R.save_heatmap(
        overlap.values, overlap.index, overlap.columns,
        title="Pairwise organism × organism shared media count",
        path=fig_dir() / "01_org_media_overlap_heatmap.png",
        cbar_label="shared media count",
    )


def fig_02_org_canonical_id_overlap_heatmap(org_chem: pd.DataFrame) -> None:
    from src.experiments.stage1.analyses import pairwise_org_overlap
    overlap = pairwise_org_overlap(org_chem, mode="shared")
    R.save_heatmap(
        overlap.values, overlap.index, overlap.columns,
        title="Pairwise organism × organism shared Canonical_ID count",
        path=fig_dir() / "02_org_canonical_id_overlap_heatmap.png",
        cbar_label="shared chemistry components",
    )


def fig_03_org_pair_jaccard_distribution(org_chem: pd.DataFrame) -> None:
    from src.experiments.stage1.analyses import pairwise_org_overlap
    jacc = pairwise_org_overlap(org_chem, mode="jaccard")
    # Take upper triangle excluding diag
    mask = np.triu(np.ones_like(jacc.values, dtype=bool), k=1)
    vals = jacc.values[mask]
    df = pd.DataFrame({"jaccard": vals, "_": "all_pairs"})
    R.save_distribution_per_group(
        df, "jaccard", "_", kind="violin",
        title="Distribution of pairwise Jaccard similarity across all 48×48 organism pairs",
        path=fig_dir() / "03_org_pair_jaccard_distribution.png",
        sort_groups_by="name",
    )


def fig_04_bipartite_org_media_top(org_media: pd.DataFrame) -> None:
    edges = []
    for org, row in org_media.iterrows():
        for media, n_exp in row.items():
            if n_exp > 0:
                edges.append((str(org), str(media), float(n_exp)))
    R.save_bipartite(
        edges,
        title="Top organisms ↔ top media (by experiment-count weight)",
        path=fig_dir() / "04_bipartite_org_media_top.png",
        left_label="organisms", right_label="media",
        max_nodes_per_side=25,
    )


# ---------------------------------------------------------------------------
# Support and sparsity (figs 05-09)
# ---------------------------------------------------------------------------

def fig_05_rows_per_organism_bar(rows_per_org: pd.DataFrame) -> None:
    df = rows_per_org.copy()
    df["__"] = "all"
    # Use distribution-per-group with each org as its own group → bar-equivalent
    # Simpler: make a one-shot bar via heatmap of shape (1, n_orgs)
    matrix = df["n_rows"].to_numpy().reshape(1, -1)
    R.save_heatmap(
        matrix, ["rows"], df["orgId"].tolist(),
        title="Fitness rows per organism (sorted descending, log-scale)",
        path=fig_dir() / "05_rows_per_organism_bar.png",
        cbar_label="rows",
        log_scale=True,
        figsize=(max(8, len(df) * 0.25), 1.8),
    )


def fig_06_conditions_per_gene_cdf(cond_per_gene: pd.DataFrame) -> None:
    R.save_ecdf(
        cond_per_gene, "n_conditions", group_col="orgId",
        title="ECDF of conditions per gene, faceted by organism",
        path=fig_dir() / "06_conditions_per_gene_cdf.png",
        log_x=True, xlabel="conditions per gene",
    )


def fig_07_conditions_per_gene_violin(cond_per_gene: pd.DataFrame) -> None:
    R.save_distribution_per_group(
        cond_per_gene, "n_conditions", "orgId", kind="violin",
        title="Conditions per gene, distribution per organism",
        path=fig_dir() / "07_conditions_per_gene_violin.png",
        log_y=True, sort_groups_by="median",
        ylabel="conditions per gene (log)",
    )


def fig_08_org_media_row_count_heatmap(org_media_rows: pd.DataFrame) -> None:
    # Trim to top N media for readability
    top_media = org_media_rows.sum(axis=0).sort_values(ascending=False).head(60).index
    sub = org_media_rows[top_media]
    R.save_heatmap(
        sub.values, sub.index, sub.columns,
        title="Fitness row counts per (organism, top-60 media)",
        path=fig_dir() / "08_org_media_row_count_heatmap.png",
        cbar_label="rows",
        log_scale=True,
    )


def fig_09_genes_per_organism_bar(genes_per_org: pd.DataFrame) -> None:
    matrix = genes_per_org["n_genes"].to_numpy().reshape(1, -1)
    R.save_heatmap(
        matrix, ["genes"], genes_per_org["orgId"].tolist(),
        title="Unique genes per organism (sorted descending, log-scale)",
        path=fig_dir() / "09_genes_per_organism_bar.png",
        cbar_label="unique gene_keys",
        log_scale=True,
        figsize=(max(8, len(genes_per_org) * 0.25), 1.8),
    )


# ---------------------------------------------------------------------------
# Quality and noise (figs 10-13)
# ---------------------------------------------------------------------------

def fig_10_fit_distribution_per_org_violin(fit_df: pd.DataFrame) -> None:
    R.save_distribution_per_group(
        fit_df.dropna(subset=["fit"]), "fit", "orgId", kind="violin",
        title="Fitness `fit` distribution per organism",
        path=fig_dir() / "10_fit_distribution_per_org_violin.png",
        sort_groups_by="median",
        ylabel="fit",
    )


def fig_11_t_stat_distribution_per_org(fit_df: pd.DataFrame) -> None:
    work = fit_df[["orgId", "abs_t"]].dropna()
    R.save_distribution_per_group(
        work, "abs_t", "orgId", kind="box",
        title="|t-statistic| distribution per organism",
        path=fig_dir() / "11_t_stat_distribution_per_org.png",
        log_y=True,
        sort_groups_by="median",
        ylabel="|t| (log)",
    )


def fig_12_cor12_distribution_per_experiment(per_exp_cor12: pd.DataFrame) -> None:
    R.save_ecdf(
        per_exp_cor12, "cor12", group_col="orgId",
        title="ECDF of replicate correlation `cor12` per experiment, faceted by organism",
        path=fig_dir() / "12_cor12_distribution_per_experiment.png",
        xlabel="cor12 (replicate correlation)",
    )


def fig_13_fit_qq_plot_global(fit_df: pd.DataFrame) -> None:
    vals = fit_df["fit"].dropna().to_numpy()
    if len(vals) > 1_000_000:
        rng = np.random.default_rng(0)
        vals = rng.choice(vals, size=1_000_000, replace=False)
    R.save_qq(
        vals, dist="norm",
        title="QQ plot of `fit` vs Normal (tail diagnostic)",
        path=fig_dir() / "13_fit_qq_plot_global.png",
    )


# ---------------------------------------------------------------------------
# Modality coverage (figs 14-16)
# ---------------------------------------------------------------------------

def fig_14_chemistry_mapped_unmapped_by_org(chem_cov: pd.DataFrame) -> None:
    df = pd.DataFrame({
        "orgId": np.repeat(chem_cov["orgId"].values, 2),
        "kind": np.tile(["mapped", "unmapped"], len(chem_cov)),
        "n_media": np.empty(2 * len(chem_cov), dtype=int),
    })
    df.loc[df["kind"] == "mapped", "n_media"] = chem_cov["mapped_media"].values
    df.loc[df["kind"] == "unmapped", "n_media"] = (
        chem_cov["total_media"].values - chem_cov["mapped_media"].values
    )
    R.save_stacked_bar(
        df, group_col="orgId", stack_col="kind", value_col="n_media",
        title="Fraction of media with v4 component data, per organism",
        path=fig_dir() / "14_chemistry_mapped_unmapped_by_org.png",
        normalize=True,
    )


def fig_15_embedding_coverage_by_org(emb_cov: pd.DataFrame) -> None:
    df = pd.DataFrame({
        "orgId": np.repeat(emb_cov["orgId"].values, 2),
        "kind": np.tile(["covered", "uncovered"], len(emb_cov)),
        "n_rows": np.empty(2 * len(emb_cov), dtype=int),
    })
    df.loc[df["kind"] == "covered", "n_rows"] = emb_cov["n_rows_with_embedding"].values
    df.loc[df["kind"] == "uncovered", "n_rows"] = (
        emb_cov["n_rows_total"].values - emb_cov["n_rows_with_embedding"].values
    )
    R.save_stacked_bar(
        df, group_col="orgId", stack_col="kind", value_col="n_rows",
        title="Fraction of fitness rows with embedding match, per organism",
        path=fig_dir() / "15_embedding_coverage_by_org.png",
        normalize=True,
    )


def fig_16_canonical_id_prevalence_distribution(ubiquity: pd.DataFrame) -> None:
    R.save_histogram(
        ubiquity["n_organisms"].values,
        # 48 organisms total → one bin per integer value 1..48
        bins=np.arange(0.5, 49.5, 1.0),
        title="Distribution of Canonical_ID prevalence: how many organisms each chemical appears in",
        path=fig_dir() / "16_canonical_id_prevalence_distribution.png",
        xlabel="number of organisms using this Canonical_ID",
        ylabel="number of chemicals",
    )


# ---------------------------------------------------------------------------
# OOD diagnostics (figs 17-19)
# ---------------------------------------------------------------------------

def fig_17_chemistry_seen_unseen_rate_per_protocol(candidates: list[dict]) -> None:
    rows = []
    for c in candidates:
        co = c["chemistry_overlap"]
        rows.append({"protocol": c["protocol_id"], "split": "val_seen",
                    "rate": co["val_canonical_id_seen_rate"]})
        rows.append({"protocol": c["protocol_id"], "split": "val_unseen",
                    "rate": co["val_canonical_id_unseen_rate"]})
        rows.append({"protocol": c["protocol_id"], "split": "test_seen",
                    "rate": co["test_canonical_id_seen_rate"]})
        rows.append({"protocol": c["protocol_id"], "split": "test_unseen",
                    "rate": co["test_canonical_id_unseen_rate"]})
    df = pd.DataFrame(rows)
    R.save_stacked_bar(
        df, group_col="protocol", stack_col="split", value_col="rate",
        title="Chemistry seen / unseen rate per candidate protocol (Canonical_ID level)",
        path=fig_dir() / "17_chemistry_seen_unseen_rate_per_protocol.png",
        normalize=False,
    )


def fig_18_embedding_cosine_to_nearest_train_per_protocol(homology_by_org: dict[str, pd.DataFrame],
                                                           candidates: list[dict]) -> None:
    rows = []
    for c in candidates:
        for o in c["val_org_ids"]:
            if o in homology_by_org:
                for v in homology_by_org[o]["max_cosine"]:
                    rows.append({"protocol": c["protocol_id"], "max_cosine": float(v)})
    if not rows:
        log.warning("fig 18: no homology data available; writing empty placeholder")
        df = pd.DataFrame({"protocol": ["_no_data"], "max_cosine": [np.nan]})
    else:
        df = pd.DataFrame(rows)
    R.save_distribution_per_group(
        df.dropna(), "max_cosine", "protocol", kind="violin",
        title="Val gene → nearest train gene cosine similarity, per candidate protocol",
        path=fig_dir() / "18_embedding_cosine_to_nearest_train_per_protocol.png",
        sort_groups_by="median",
        ylabel="max cosine to train",
    )


def fig_19_homology_similarity_by_org(homology_by_org: dict[str, pd.DataFrame]) -> None:
    rows = []
    for org, df in homology_by_org.items():
        for v in df["max_cosine"]:
            rows.append({"orgId": org, "max_cosine": float(v)})
    if not rows:
        log.warning("fig 19: no homology data; writing empty placeholder")
        df = pd.DataFrame({"orgId": ["_no_data"], "max_cosine": [np.nan]})
    else:
        df = pd.DataFrame(rows)
    R.save_distribution_per_group(
        df.dropna(), "max_cosine", "orgId", kind="violin",
        title="Per-organism distribution of nearest-train-gene cosine similarity",
        path=fig_dir() / "19_homology_similarity_by_org.png",
        sort_groups_by="median",
        ylabel="max cosine to nearest train gene",
    )


# ---------------------------------------------------------------------------
# Representation-mode audit (figs 20-21)
# ---------------------------------------------------------------------------

def fig_20_representation_mode_proportions_per_org(mode_per_org: pd.DataFrame) -> None:
    # mode_per_org is wide: orgId + mode columns. Reshape to long.
    mode_cols = [c for c in mode_per_org.columns if c != "orgId"]
    long = mode_per_org.melt(id_vars="orgId", value_vars=mode_cols,
                              var_name="representation_mode", value_name="weighted_rows")
    R.save_stacked_bar(
        long, group_col="orgId", stack_col="representation_mode", value_col="weighted_rows",
        title="Representation-mode proportions per organism (weighted by row count)",
        path=fig_dir() / "20_representation_mode_proportions_per_org.png",
        normalize=True,
    )


def fig_21_representation_mode_per_protocol(candidates: list[dict]) -> None:
    rows = []
    for c in candidates:
        for split, props in c["representation_mode_proportions"].items():
            for mode, frac in props.items():
                rows.append({"protocol_split": f"{c['protocol_id']}|{split}",
                            "mode": mode, "frac": frac})
    df = pd.DataFrame(rows)
    R.save_stacked_bar(
        df, group_col="protocol_split", stack_col="mode", value_col="frac",
        title="Representation-mode proportions per candidate protocol partition",
        path=fig_dir() / "21_representation_mode_per_protocol.png",
        normalize=True,
    )


# ---------------------------------------------------------------------------
# Cross-organism chemistry coverage (figs 22-24)
# ---------------------------------------------------------------------------

def fig_22_chemical_ubiquity_histogram(ubiquity: pd.DataFrame) -> None:
    R.save_histogram(
        ubiquity["n_organisms"].values,
        bins=np.arange(0.5, 49.5, 1.0),
        title="Chemical ubiquity: how many organisms use each Canonical_ID",
        path=fig_dir() / "22_chemical_ubiquity_histogram.png",
        xlabel="number of organisms using this chemical",
        ylabel="number of chemicals",
    )


def fig_23_organism_topN_chemical_heatmap(org_chem: pd.DataFrame, top_n: int = 100) -> None:
    # Top-N chemicals by total experiment count
    totals = org_chem.sum(axis=0).sort_values(ascending=False)
    top = totals.head(top_n).index
    sub = org_chem[top]
    R.save_heatmap(
        sub.values, sub.index, sub.columns,
        title=f"Organism × top-{top_n} chemicals (log experiment count)",
        path=fig_dir() / "23_organism_topN_chemical_heatmap.png",
        cbar_label="experiment count",
        log_scale=True,
    )


def fig_24_chemical_coverage_curve(ubiquity: pd.DataFrame) -> None:
    R.save_coverage_curve(
        ubiquity, primary_col="n_organisms", secondary_col="n_experiments",
        sort_descending=True, log_secondary=True,
        title="Chemical coverage curve: chemicals sorted by descending #organisms",
        path=fig_dir() / "24_chemical_coverage_curve.png",
        primary_label="# organisms using this chemical",
        secondary_label="# experiments (log)",
    )


# ---------------------------------------------------------------------------
# Optional fig 25 — media chemistry PCA (UMAP-fallback)
# ---------------------------------------------------------------------------

def fig_25_media_chemistry_umap(components: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    from sklearn.decomposition import PCA
    # Build (media × Canonical_ID) multihot matrix
    bin_mat = (components.assign(present=1)
               .pivot_table(index="Media", columns="Canonical_ID",
                            values="present", aggfunc="max", fill_value=0))
    media_to_norgs = (fit_df[["orgId", "media"]].dropna().drop_duplicates()
                     .groupby("media")["orgId"].nunique().to_dict())
    n_orgs = np.array([media_to_norgs.get(m, 0) for m in bin_mat.index])
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(bin_mat.values.astype(np.float32))
    R.save_umap_scatter(
        proj, color_values=n_orgs.astype(np.float32),
        color_label="# distinct organisms using medium",
        point_labels=list(bin_mat.index),
        title="Media chemistry PCA (2D), colored by # organisms (exploratory)",
        path=fig_dir() / "25_media_chemistry_pca.png",
    )
