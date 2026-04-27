"""Plotting helpers for stage/tier reports (REFACTORPLAN §11).

Every helper writes BOTH a PNG and a sibling CSV with the underlying data,
so figures can be reproduced or re-styled without re-running the analysis.

Conventions:
    - matplotlib only (no seaborn) for fewer transitive deps and tighter control
    - all helpers return None and write to `path: Path` directly
    - CSV sidecar shares the basename: `path.with_suffix('.csv')`
    - DPI = 150, figsize chosen per helper

Stubs are marked NotImplementedError; flesh out during S1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _csv_sidecar(path: Path) -> Path:
    """Return the CSV path that pairs with the given PNG path."""
    return Path(path).with_suffix(".csv")


# ---------------------------------------------------------------------------
# Helpers
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
) -> None:
    """Save a heatmap as PNG + CSV.

    Use cases: organism × organism overlap matrices, organism × chemical counts.
    CSV format: matrix written with row_labels as index, col_labels as columns.
    """
    raise NotImplementedError("Implement during S1")


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
) -> None:
    """Save a per-group distribution plot (violin or box).

    Use cases: fit / |t| / cor12 distributions per organism.
    CSV format: tidy long format with columns [group, value].
    """
    raise NotImplementedError("Implement during S1")


def save_ecdf(
    df: pd.DataFrame,
    value_col: str,
    *,
    group_col: Optional[str] = None,
    title: str,
    path: Path,
    log_x: bool = False,
) -> None:
    """Save an empirical CDF plot (one curve per group if group_col given).

    Use cases: conditions per gene, support distributions.
    CSV format: tidy long format with columns [group, value, ecdf].
    """
    raise NotImplementedError("Implement during S1")


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
) -> None:
    """Save a stacked bar chart (one bar per group, stacked by stack_col).

    Use cases: representation_mode proportions per organism / per protocol.
    If `value_col` is None, count rows. If `normalize`, scale each bar to 1.0.
    CSV format: pivoted (group on index, stack values on columns).
    """
    raise NotImplementedError("Implement during S1")


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
    raise NotImplementedError("Implement during S1")


def save_qq(
    values: np.ndarray,
    *,
    dist: Literal["norm", "t"] = "norm",
    df_t: Optional[float] = None,
    title: str,
    path: Path,
) -> None:
    """Save a QQ plot to assess tail behavior.

    Use cases: characterize fit residual distribution to inform Huber vs MSE.
    CSV format: two columns [theoretical, sample_quantile].
    """
    raise NotImplementedError("Implement during S1")


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

    Use cases: H-HOMO-01 evaluation — does error correlate with train similarity?
    CSV format: per-bin aggregated table with mean, std, n.
    """
    raise NotImplementedError("Implement during S1")


def save_coverage_curve(
    df: pd.DataFrame,
    *,
    primary_col: str,
    secondary_col: Optional[str] = None,
    sort_descending: bool = True,
    title: str,
    path: Path,
    log_secondary: bool = True,
) -> None:
    """Save a dual-axis coverage curve (e.g. chemicals sorted by #orgs using them).

    primary_col: y-left (e.g. number of organisms)
    secondary_col: y-right, log-scale by default (e.g. total experiments)
    CSV format: input df sorted as plotted, with rank column added.
    """
    raise NotImplementedError("Implement during S1")


def save_umap_scatter(
    embedding_2d: np.ndarray,
    *,
    color_values: np.ndarray,
    color_label: str,
    point_labels: Optional[Sequence[str]] = None,
    title: str,
    path: Path,
) -> None:
    """Save a 2D scatter (typically UMAP / PCA projection).

    Use cases: exploratory media-chemistry UMAP colored by #organisms.
    Flagged exploratory; not promotion-gating.
    CSV format: 4 columns [x, y, color_value, point_label].
    """
    raise NotImplementedError("Implement during S1")
