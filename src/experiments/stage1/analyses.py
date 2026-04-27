"""Pure analysis functions for S1 (no plotting, no I/O of figures).

Each function returns a DataFrame, dict, or numpy array that figures.py and
candidates.py consume. Keeps the math separate from the visualization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

CANONICAL_DIR = Path("data/derived/canonical/v0")
WORKBOOK = Path("data/media_composition_v4.xlsx")
EMBEDDING_DIR = Path("data/processed/ProtLM_embeddings_layer8")
SHEET = "Media_Components_ML"

_FITNESS_COLS = [
    "orgId", "locusId", "expName", "fit", "t", "cor12",
    "media", "gene_key", "abs_t", "has_media_composition",
    "expGroup", "temperature", "pH", "aerobic",
    "condition_1", "condition_2", "condition_3", "condition_4",
]


def load_fitness(cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load canonical fitness with the columns S1 needs."""
    if cols is None:
        cols = _FITNESS_COLS
    return pd.read_parquet(CANONICAL_DIR / "fitness_experiment_long.parquet",
                           columns=list(cols))


def load_v4_components() -> pd.DataFrame:
    """Load Media_Components_ML, filter to Include_in_ml=True, attach representation_mode."""
    df = pd.read_excel(WORKBOOK, sheet_name=SHEET)
    df = df[df["Include_in_ml"] == True].copy()  # noqa: E712
    decomp_to_mode = {
        "direct": "physical",
        "salt": "physical",
        "mix": "mix",
        "extract": "extract",
        "unrecoverable": "in_silico",
    }
    df["representation_mode"] = df["Decomposition_type"].map(decomp_to_mode).fillna("in_silico")
    return df


# ---------------------------------------------------------------------------
# Overlap analyses
# ---------------------------------------------------------------------------

def org_media_matrix(fit_df: pd.DataFrame) -> pd.DataFrame:
    """For each (org, media): how many experiments use this medium?

    Returns a wide DataFrame indexed by orgId, columns = media, values = exp count.
    """
    counts = (fit_df.dropna(subset=["media"])
              .groupby(["orgId", "media", "expName"]).size().reset_index()
              .groupby(["orgId", "media"]).size())  # exp count per (org, media)
    return counts.unstack(fill_value=0)


def org_chemical_matrix(fit_df: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """For each (org, Canonical_ID): how many experiments use this chemical (via media)?

    Joins fitness `media` to v4 `Media`, then aggregates by (orgId, Canonical_ID).
    """
    media_to_chem = components[["Media", "Canonical_ID"]].rename(
        columns={"Media": "media"}).drop_duplicates()
    exp_media = (fit_df[["orgId", "expName", "media"]]
                 .drop_duplicates(subset=["orgId", "expName"]))
    exp_chem = exp_media.merge(media_to_chem, on="media", how="inner")
    counts = (exp_chem.groupby(["orgId", "Canonical_ID"])
              .size().unstack(fill_value=0))
    return counts


def pairwise_org_overlap(matrix: pd.DataFrame, mode: str = "shared") -> pd.DataFrame:
    """Pairwise organism × organism matrix.

    matrix: rows=orgId, columns=items (media or Canonical_IDs).
    mode='shared': count of items both use (binary intersection).
    mode='jaccard': |intersection| / |union|.
    """
    binary = (matrix > 0).astype(np.int32)
    intersection = binary.values @ binary.values.T
    if mode == "shared":
        out = intersection
    elif mode == "jaccard":
        sums = binary.values.sum(axis=1)
        union = sums[:, None] + sums[None, :] - intersection
        out = np.where(union > 0, intersection / union, 0.0)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return pd.DataFrame(out, index=matrix.index, columns=matrix.index)


def chemical_ubiquity(org_chem: pd.DataFrame) -> pd.DataFrame:
    """For each Canonical_ID: how many distinct organisms use it, and total exp count."""
    binary = (org_chem > 0).astype(int)
    n_orgs = binary.sum(axis=0)              # over rows (orgs)
    n_exp = org_chem.sum(axis=0)
    out = pd.DataFrame({"Canonical_ID": org_chem.columns,
                        "n_organisms": n_orgs.values,
                        "n_experiments": n_exp.values})
    return out


# ---------------------------------------------------------------------------
# Support and sparsity
# ---------------------------------------------------------------------------

def rows_per_organism(fit_df: pd.DataFrame) -> pd.DataFrame:
    out = fit_df.groupby("orgId").size().reset_index(name="n_rows")
    return out.sort_values("n_rows", ascending=False)


def conditions_per_gene(fit_df: pd.DataFrame) -> pd.DataFrame:
    """For each (orgId, gene_key): number of experiments (=conditions) tested."""
    out = (fit_df.dropna(subset=["gene_key", "expName"])
           .groupby(["orgId", "gene_key"])["expName"]
           .nunique().reset_index(name="n_conditions"))
    return out


def org_media_row_counts(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Wide org × media matrix of fitness row counts (not experiment counts)."""
    out = fit_df.groupby(["orgId", "media"]).size().unstack(fill_value=0)
    return out


def genes_per_organism(fit_df: pd.DataFrame) -> pd.DataFrame:
    out = (fit_df.groupby("orgId")["gene_key"]
           .nunique().reset_index(name="n_genes"))
    return out.sort_values("n_genes", ascending=False)


# ---------------------------------------------------------------------------
# Quality and noise
# ---------------------------------------------------------------------------

def per_experiment_cor12(fit_df: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment with its cor12 (replicate correlation)."""
    return (fit_df[["orgId", "expName", "cor12"]]
            .drop_duplicates(subset=["orgId", "expName"])
            .dropna(subset=["cor12"]))


# ---------------------------------------------------------------------------
# Modality coverage
# ---------------------------------------------------------------------------

def chemistry_coverage_by_org(fit_df: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Per organism: fraction of (media used) that have v4 component data."""
    v4_media = set(components["Media"].unique())
    org_media = (fit_df[["orgId", "media"]].dropna().drop_duplicates())
    org_media["mapped"] = org_media["media"].isin(v4_media)
    out = (org_media.groupby("orgId")["mapped"]
           .agg(["sum", "count"]).reset_index()
           .rename(columns={"sum": "mapped_media", "count": "total_media"}))
    out["fraction_mapped"] = out["mapped_media"] / out["total_media"]
    return out.sort_values("fraction_mapped", ascending=False)


def embedding_coverage_by_org(fit_df: pd.DataFrame,
                              embedding_dir: Path = EMBEDDING_DIR) -> pd.DataFrame:
    """Per organism: fraction of fitness rows whose gene_key has an embedding."""
    rows = []
    for org in sorted(fit_df["orgId"].dropna().unique()):
        pt_path = embedding_dir / f"{org}_proteomelm.pt"
        if not pt_path.exists():
            rows.append({"orgId": org, "n_rows_total": int((fit_df["orgId"] == org).sum()),
                         "n_rows_with_embedding": 0, "fraction_covered": 0.0})
            continue
        bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
        emb_keys = set(bundle["group_labels"])
        org_rows = fit_df[fit_df["orgId"] == org]
        covered = org_rows["gene_key"].isin(emb_keys).sum()
        rows.append({
            "orgId": org,
            "n_rows_total": int(len(org_rows)),
            "n_rows_with_embedding": int(covered),
            "fraction_covered": float(covered / max(len(org_rows), 1)),
        })
    return pd.DataFrame(rows).sort_values("fraction_covered", ascending=False)


# ---------------------------------------------------------------------------
# Homology / OOD
# ---------------------------------------------------------------------------

def load_org_embeddings(org: str, embedding_dir: Path = EMBEDDING_DIR) -> tuple[np.ndarray, list[str]]:
    """Return (embeddings_fp32, gene_keys) for one organism, or empty if missing."""
    pt_path = embedding_dir / f"{org}_proteomelm.pt"
    if not pt_path.exists():
        return np.zeros((0, 0), dtype=np.float32), []
    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    emb = bundle["embeddings"].to(torch.float32).numpy()
    keys = list(bundle["group_labels"])
    return emb, keys


def cross_org_max_cosine(val_org: str, train_orgs: Iterable[str],
                         embedding_dir: Path = EMBEDDING_DIR,
                         max_train_genes: int | None = None) -> pd.DataFrame:
    """For each gene in val_org, max cosine similarity to any gene in any train_org.

    Returns DataFrame: gene_key, max_cosine, nearest_train_gene_key, nearest_train_org.
    To bound memory, optionally subsample train genes to max_train_genes.
    """
    val_emb, val_keys = load_org_embeddings(val_org, embedding_dir)
    if val_emb.size == 0:
        return pd.DataFrame(columns=["gene_key", "max_cosine",
                                     "nearest_train_gene_key", "nearest_train_org"])
    val_norm = val_emb / np.maximum(np.linalg.norm(val_emb, axis=1, keepdims=True), 1e-9)

    best = np.full(len(val_keys), -np.inf, dtype=np.float32)
    nearest_key = np.array(["" for _ in val_keys], dtype=object)
    nearest_org = np.array(["" for _ in val_keys], dtype=object)

    rng = np.random.default_rng(0)
    for org in train_orgs:
        if org == val_org:
            continue
        emb, keys = load_org_embeddings(org, embedding_dir)
        if emb.size == 0:
            continue
        if max_train_genes is not None and len(keys) > max_train_genes:
            idx = rng.choice(len(keys), size=max_train_genes, replace=False)
            emb, keys = emb[idx], [keys[i] for i in idx]
        emb_norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        # block matmul to avoid O(N*M) memory in one go
        block = 2000
        for start in range(0, len(val_keys), block):
            end = start + block
            sims = val_norm[start:end] @ emb_norm.T   # (b, M)
            block_max = sims.max(axis=1)
            block_argmax = sims.argmax(axis=1)
            improved = block_max > best[start:end]
            best[start:end] = np.where(improved, block_max, best[start:end])
            for i, imp in enumerate(improved):
                if imp:
                    nearest_key[start + i] = keys[block_argmax[i]]
                    nearest_org[start + i] = org
    return pd.DataFrame({
        "gene_key": val_keys,
        "max_cosine": best,
        "nearest_train_gene_key": nearest_key,
        "nearest_train_org": nearest_org,
    })


# ---------------------------------------------------------------------------
# Representation-mode audit
# ---------------------------------------------------------------------------

def representation_mode_per_org(fit_df: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """For each (orgId, representation_mode): number of fitness rows.

    A row's mode set is multi-valued (a medium can have many components with
    different modes). For S1 audit purposes, we count rows weighted by the
    fraction of components in each mode for the row's medium.
    """
    media_modes = (components.groupby(["Media", "representation_mode"]).size()
                   .reset_index(name="n_components"))
    media_total = media_modes.groupby("Media")["n_components"].sum().rename("total")
    media_modes = media_modes.merge(media_total, on="Media")
    media_modes["mode_fraction"] = media_modes["n_components"] / media_modes["total"]
    pivot = media_modes.pivot(index="Media", columns="representation_mode",
                              values="mode_fraction").fillna(0.0)

    org_media_rows = fit_df.groupby(["orgId", "media"]).size().reset_index(name="n_rows")
    merged = org_media_rows.merge(pivot.reset_index().rename(columns={"Media": "media"}),
                                  on="media", how="left")
    mode_cols = [c for c in pivot.columns]
    for mc in mode_cols:
        merged[mc] = merged[mc].fillna(0.0) * merged["n_rows"]

    out = merged.groupby("orgId")[mode_cols].sum().reset_index()
    return out
