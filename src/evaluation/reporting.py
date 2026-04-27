"""Plotting helpers for stage/tier reports (REFACTORPLAN §11).

Every helper writes BOTH a PNG and a sibling CSV with the underlying data,
so figures can be reproduced or re-styled without re-running the analysis.

Conventions:
    - matplotlib for plotting; networkx for graphs; sklearn.PCA for UMAP fallback
    - all helpers return None and write to `path: Path` directly
    - CSV sidecar shares the basename: `path.with_suffix('.csv')`
    - DPI = 150
    - tight_layout + savefig + close to release figure memory
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in headless runners
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 150


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _csv_sidecar(path: Path) -> Path:
    """Return the CSV path that pairs with the given PNG path."""
    return Path(path).with_suffix(".csv")


def _save_and_close(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    title: str,
    path: Path,
    cmap: str = "viridis",
    cbar_label: Optional[str] = None,
    annotate: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    log_scale: bool = False,
) -> None:
    """Save a heatmap as PNG + CSV.

    CSV format: matrix written with row_labels as index, col_labels as columns.
    """
    path = _ensure_parent(Path(path))
    matrix = np.asarray(matrix)
    if figsize is None:
        figsize = (max(6, len(col_labels) * 0.18), max(5, len(row_labels) * 0.18))
    fig, ax = plt.subplots(figsize=figsize)
    plot_data = np.log1p(matrix) if log_scale else matrix
    im = ax.imshow(plot_data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    if len(col_labels) <= 60:
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    ax.set_yticklabels(row_labels, fontsize=7)
    if annotate and matrix.size <= 2500:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:g}", ha="center", va="center",
                        fontsize=6, color="white")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if cbar_label:
        cbar.set_label(("log1p " if log_scale else "") + cbar_label)
    _save_and_close(fig, path)
    pd.DataFrame(matrix, index=list(row_labels), columns=list(col_labels)).to_csv(
        _csv_sidecar(path)
    )


def save_histogram(
    values: np.ndarray | pd.Series,
    *,
    bins: int | Sequence[float] | str = "auto",
    title: str,
    path: Path,
    xlabel: Optional[str] = None,
    ylabel: str = "count",
    log_y: bool = False,
    annotate_summary: bool = True,
) -> None:
    """Save a 1-D histogram as PNG + CSV.

    Use cases: distributions of a single integer- or real-valued quantity
    (e.g. number of organisms each chemical appears in).

    CSV format: per-bin table with bin_left, bin_right, count.
    """
    path = _ensure_parent(Path(path))
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges, _ = ax.hist(arr, bins=bins, edgecolor="white", linewidth=0.5,
                                color="C0", alpha=0.85)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or "value")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if annotate_summary and len(arr) > 0:
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        ax.axvline(median, color="C3", lw=1, ls="--", alpha=0.8,
                   label=f"median={median:g}")
        ax.axvline(mean, color="C2", lw=1, ls=":", alpha=0.8,
                   label=f"mean={mean:.2f}")
        ax.legend(fontsize=8)
    _save_and_close(fig, path)
    pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts.astype(int),
    }).to_csv(_csv_sidecar(path), index=False)


def save_distribution_per_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    kind: Literal["violin", "box"] = "violin",
    title: str,
    path: Path,
    log_y: bool = False,
    sort_groups_by: Optional[Literal["median", "count", "name"]] = "median",
    max_groups: Optional[int] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Save a per-group distribution plot (violin or box).

    CSV format: tidy long format with columns [group, value].
    """
    path = _ensure_parent(Path(path))
    work = df[[group_col, value_col]].dropna()
    if sort_groups_by == "median":
        order = work.groupby(group_col)[value_col].median().sort_values().index.tolist()
    elif sort_groups_by == "count":
        order = work.groupby(group_col)[value_col].size().sort_values(ascending=False).index.tolist()
    elif sort_groups_by == "name":
        order = sorted(work[group_col].unique())
    else:
        order = work[group_col].unique().tolist()
    if max_groups is not None and len(order) > max_groups:
        order = order[:max_groups]
        work = work[work[group_col].isin(order)]
    data = [work.loc[work[group_col] == g, value_col].to_numpy() for g in order]

    figsize = (max(6, len(order) * 0.25), 5)
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "violin":
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
    else:
        ax.boxplot(data, showfliers=False, widths=0.6)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    _save_and_close(fig, path)
    work.assign(**{group_col: pd.Categorical(work[group_col], categories=order)}).to_csv(
        _csv_sidecar(path), index=False
    )


def save_ecdf(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: Optional[str] = None,
    title: str,
    path: Path,
    log_x: bool = False,
    xlabel: Optional[str] = None,
) -> None:
    """Save an empirical CDF plot (one curve per group if group_col given).

    CSV format: tidy long format with columns [group, value, ecdf].
    """
    path = _ensure_parent(Path(path))
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    if group_col is None:
        vals = np.sort(df[value_col].dropna().to_numpy())
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, ecdf, where="post")
        rows.append(pd.DataFrame({"group": "_all_", "value": vals, "ecdf": ecdf}))
    else:
        groups = sorted(df[group_col].dropna().unique())
        cmap = plt.get_cmap("viridis", len(groups))
        for i, g in enumerate(groups):
            sub = df.loc[df[group_col] == g, value_col].dropna().to_numpy()
            if sub.size == 0:
                continue
            sub = np.sort(sub)
            ecdf = np.arange(1, len(sub) + 1) / len(sub)
            ax.step(sub, ecdf, where="post", color=cmap(i), alpha=0.6, lw=1.0)
            rows.append(pd.DataFrame({"group": g, "value": sub, "ecdf": ecdf}))
        # Put a small legend if reasonable
        if 1 < len(groups) <= 20:
            ax.legend(groups, fontsize=7, ncol=2, loc="lower right")
    ax.set_xlabel(xlabel or value_col)
    ax.set_ylabel("ECDF")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save_and_close(fig, path)
    pd.concat(rows, ignore_index=True).to_csv(_csv_sidecar(path), index=False)


def save_stacked_bar(
    df: pd.DataFrame,
    group_col: str,
    stack_col: str,
    *,
    value_col: Optional[str] = None,
    title: str,
    path: Path,
    normalize: bool = True,
    color_map: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    sort_groups_by_total: bool = True,
) -> None:
    """Save a stacked bar chart (one bar per group, stacked by stack_col).

    If `value_col` is None, count rows. If `normalize`, scale each bar to 1.0.
    CSV format: pivoted (group on index, stack values on columns).
    """
    path = _ensure_parent(Path(path))
    if value_col is None:
        pivot = df.groupby([group_col, stack_col]).size().unstack(fill_value=0)
    else:
        pivot = df.groupby([group_col, stack_col])[value_col].sum().unstack(fill_value=0)
    if sort_groups_by_total:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    if normalize:
        row_sums = pivot.sum(axis=1).replace(0, np.nan)
        pivot_n = pivot.div(row_sums, axis=0).fillna(0.0)
    else:
        pivot_n = pivot

    if figsize is None:
        figsize = (max(6, len(pivot_n) * 0.25), 5)
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(pivot_n))
    cols = list(pivot_n.columns)
    cmap = plt.get_cmap("tab10")
    for j, c in enumerate(cols):
        color = (color_map or {}).get(c, cmap(j % 10))
        ax.bar(np.arange(len(pivot_n)), pivot_n[c].to_numpy(), bottom=bottom,
               label=str(c), color=color, edgecolor="white", linewidth=0.3)
        bottom += pivot_n[c].to_numpy()
    ax.set_xticks(np.arange(len(pivot_n)))
    ax.set_xticklabels([str(g) for g in pivot_n.index], rotation=90, fontsize=7)
    ax.set_ylabel("fraction" if normalize else (value_col or "count"))
    ax.set_xlabel(group_col)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.02 if normalize else None)
    _save_and_close(fig, path)
    pivot_n.to_csv(_csv_sidecar(path))


def save_bipartite(
    edges: Iterable[tuple[str, str, float]],
    *,
    title: str,
    path: Path,
    left_label: str,
    right_label: str,
    max_nodes_per_side: int = 30,
    edge_weight_threshold: Optional[float] = None,
) -> None:
    """Save a bipartite graph PNG (e.g. organisms ↔ media).

    edges: iterable of (left_node, right_node, weight) tuples.
    Trimmed to the top max_nodes_per_side by total degree on each side.
    CSV format: edges as 3-column tidy table.
    """
    import networkx as nx
    path = _ensure_parent(Path(path))
    edge_list = list(edges)
    df_edges = pd.DataFrame(edge_list, columns=["left", "right", "weight"])
    if edge_weight_threshold is not None:
        df_edges = df_edges[df_edges["weight"] >= edge_weight_threshold]
    # Top-N nodes per side by total weight
    top_left = (df_edges.groupby("left")["weight"].sum()
                .sort_values(ascending=False).head(max_nodes_per_side).index)
    top_right = (df_edges.groupby("right")["weight"].sum()
                 .sort_values(ascending=False).head(max_nodes_per_side).index)
    df_top = df_edges[df_edges["left"].isin(top_left) & df_edges["right"].isin(top_right)]

    G = nx.Graph()
    for n in top_left:
        G.add_node(("L", n), bipartite=0)
    for n in top_right:
        G.add_node(("R", n), bipartite=1)
    for _, r in df_top.iterrows():
        G.add_edge(("L", r["left"]), ("R", r["right"]), weight=float(r["weight"]))

    pos = {}
    for i, n in enumerate(top_left):
        pos[("L", n)] = (0.0, -i)
    for i, n in enumerate(top_right):
        pos[("R", n)] = (1.0, -i * (len(top_left) / max(len(top_right), 1)))

    fig, ax = plt.subplots(figsize=(10, max(6, max(len(top_left), len(top_right)) * 0.25)))
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if weights:
        wmax = max(weights)
        edge_widths = [0.3 + 2.0 * (w / wmax) for w in weights]
    else:
        edge_widths = []
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=[("L", n) for n in top_left],
                           ax=ax, node_color="C0", node_size=140)
    nx.draw_networkx_nodes(G, pos, nodelist=[("R", n) for n in top_right],
                           ax=ax, node_color="C1", node_size=140)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                            labels={n: n[1] for n in G.nodes()})
    ax.set_axis_off()
    ax.set_title(f"{title}  ({left_label} ← → {right_label})")
    _save_and_close(fig, path)
    df_edges.to_csv(_csv_sidecar(path), index=False)


def save_qq(
    values: np.ndarray,
    *,
    dist: Literal["norm", "t"] = "norm",
    df_t: Optional[float] = None,
    title: str,
    path: Path,
) -> None:
    """Save a QQ plot to assess tail behavior.

    CSV format: two columns [theoretical_quantile, sample_quantile].
    """
    from scipy import stats
    path = _ensure_parent(Path(path))
    vals = np.asarray(values).astype(float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    sample_sorted = np.sort(vals)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    if dist == "norm":
        theoretical = stats.norm.ppf(quantiles)
    elif dist == "t":
        theoretical = stats.t.ppf(quantiles, df=df_t or 5)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, sample_sorted, s=2, alpha=0.5)
    # Reference line: through 1st and 3rd theoretical quartiles of sample
    q1, q3 = np.percentile(sample_sorted, [25, 75])
    qt1, qt3 = np.percentile(theoretical, [25, 75])
    if qt3 != qt1:
        slope = (q3 - q1) / (qt3 - qt1)
        intercept = q1 - slope * qt1
        xs = np.array([theoretical.min(), theoretical.max()])
        ax.plot(xs, slope * xs + intercept, "r-", lw=1, label="reference line (Q1-Q3)")
        ax.legend(fontsize=8)
    ax.set_xlabel(f"theoretical quantile ({dist})")
    ax.set_ylabel("sample quantile")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save_and_close(fig, path)
    pd.DataFrame({
        "theoretical_quantile": theoretical,
        "sample_quantile": sample_sorted,
    }).to_csv(_csv_sidecar(path), index=False)


def save_similarity_bin_scatter(
    df: pd.DataFrame,
    similarity_col: str,
    metric_col: str,
    *,
    bins: Sequence[float],
    title: str,
    path: Path,
    error_bars: bool = True,
) -> None:
    """Save a scatter / line plot of metric stratified by similarity bin.

    CSV format: per-bin aggregated table with mean, std, n.
    """
    path = _ensure_parent(Path(path))
    work = df[[similarity_col, metric_col]].dropna()
    bins_arr = np.asarray(bins)
    work["__bin"] = pd.cut(work[similarity_col], bins=bins_arr, include_lowest=True)
    g = (work.groupby("__bin", observed=True)[metric_col]
         .agg(["mean", "std", "count"]).reset_index())
    centers = [(b.left + b.right) / 2 for b in g["__bin"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    if error_bars:
        ax.errorbar(centers, g["mean"], yerr=g["std"], fmt="o-",
                    capsize=3, alpha=0.8)
    else:
        ax.plot(centers, g["mean"], "o-")
    for c, n in zip(centers, g["count"]):
        ax.text(c, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", fontsize=7)
    ax.set_xlabel(similarity_col)
    ax.set_ylabel(f"mean {metric_col}")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save_and_close(fig, path)
    g["bin_center"] = centers
    g.to_csv(_csv_sidecar(path), index=False)


def save_coverage_curve(
    df: pd.DataFrame,
    *,
    primary_col: str,
    secondary_col: Optional[str] = None,
    sort_descending: bool = True,
    title: str,
    path: Path,
    log_secondary: bool = True,
    primary_label: Optional[str] = None,
    secondary_label: Optional[str] = None,
) -> None:
    """Save a dual-axis coverage curve."""
    path = _ensure_parent(Path(path))
    sorted_df = df.sort_values(primary_col, ascending=not sort_descending).reset_index(drop=True)
    sorted_df["rank"] = np.arange(1, len(sorted_df) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sorted_df["rank"], sorted_df[primary_col], "C0-", lw=1.2,
             label=primary_label or primary_col)
    ax1.set_xlabel("rank")
    ax1.set_ylabel(primary_label or primary_col, color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(alpha=0.3)
    if secondary_col is not None:
        ax2 = ax1.twinx()
        ax2.plot(sorted_df["rank"], sorted_df[secondary_col], "C1-", lw=1.0, alpha=0.8,
                 label=secondary_label or secondary_col)
        ax2.set_ylabel(secondary_label or secondary_col, color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        if log_secondary:
            ax2.set_yscale("log")
    ax1.set_title(title)
    _save_and_close(fig, path)
    sorted_df.to_csv(_csv_sidecar(path), index=False)


def save_umap_scatter(
    embedding_2d: np.ndarray,
    *,
    color_values: np.ndarray,
    color_label: str,
    point_labels: Optional[Sequence[str]] = None,
    title: str,
    path: Path,
) -> None:
    """Save a 2D scatter (typically PCA / UMAP projection).

    embedding_2d shape: (n_points, 2). If 2D projection wasn't computed yet,
    pass through sklearn.decomposition.PCA(n_components=2).fit_transform(...).
    """
    path = _ensure_parent(Path(path))
    if embedding_2d.shape[1] != 2:
        raise ValueError("embedding_2d must be (n, 2)")
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                    c=color_values, cmap="viridis", s=12, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_label)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title)
    _save_and_close(fig, path)
    df_out = pd.DataFrame({
        "x": embedding_2d[:, 0],
        "y": embedding_2d[:, 1],
        "color_value": color_values,
    })
    if point_labels is not None:
        df_out["label"] = list(point_labels)
    df_out.to_csv(_csv_sidecar(path), index=False)
