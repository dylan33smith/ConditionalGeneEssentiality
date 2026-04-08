#!/usr/bin/env python3
"""Phase 0 elucidation: figures + SQL summaries + media workbook audit.

Run from repo root:
  python data_analysis/run_phase0.py

Outputs:
  figures/phase0/*.png
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
    MEDIA_XLSX,
    OUTPUT_DIR,
    REPO_ROOT,
)

SAMPLE_MOD = 313  # ~88k rows from ~27.4M


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




def excel_experiment_coverage(experiments: pd.DataFrame) -> dict:
    """How many DB experiments appear in the Excel `Experiments` sheet (orgId + name)."""
    ex = pd.read_excel(MEDIA_XLSX, sheet_name="Experiments", header=0)
    ex = ex.rename(columns={"name": "expName"})
    key = ["orgId", "expName"]
    db = experiments[key + ["media"]].drop_duplicates()
    merged = db.merge(ex[key + ["Media"]], on=key, how="left")
    matched = int(merged["Media"].notna().sum())
    mismatch = merged.dropna(subset=["Media"])
    mismatch = mismatch[mismatch["media"].astype(str) != mismatch["Media"].astype(str)]
    return {
        "n_experiments_db": int(len(db)),
        "n_rows_excel_experiments": int(len(ex)),
        "matched_on_orgId_expName": matched,
        "unmatched_db_experiments": int(len(db) - matched),
        "media_string_mismatches_on_matches": int(len(mismatch)),
    }


def write_media_audit_md() -> str:
    xl = pd.ExcelFile(MEDIA_XLSX)
    lines = [
        "# Media composition workbook audit (Phase 0)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} UTC",
        "File: `data/media_composition.xlsx` (repo root-relative)",
        "",
        "## Sheet inventory",
        "",
        "| Index | Sheet name | Rows | Cols |",
        "| ----- | ---------- | ---- | ---- |",
    ]
    for i, name in enumerate(xl.sheet_names):
        df = pd.read_excel(MEDIA_XLSX, sheet_name=name, header=None)
        lines.append(f"| {i} | {name} | {df.shape[0]} | {df.shape[1]} |")
    conn_x = connect()
    try:
        exp_db = pd.read_sql_query("SELECT orgId, expName, media FROM Experiment", conn_x)
    finally:
        conn_x.close()
    cov = excel_experiment_coverage(exp_db)
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
            "## Authoritative mapping (restart)",
            "",
            "- **Medium → components:** sheet **`Media_Components`** (index 1): columns *Media*, *Component*, *Concentration*, *Units*, …",
            "- **Experiment ↔ medium / metadata:** sheet **`Experiments`** (index 2): join to `Experiment` table on **`(orgId, name)`** ↔ **`(orgId, expName)`**; medium string in column *Media*.",
            "- v1 loader note referenced sheet index 2 for *composition*; this workbook places **composition** on **`Media_Components`**, not index 2. Index 2 is experiment-level metadata.",
            "",
            "## `Media_Components` header row (row 0)",
            "",
        ]
    )
    mc = pd.read_excel(MEDIA_XLSX, sheet_name="Media_Components", header=0)
    lines.append("| " + " | ".join(str(c) for c in mc.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(mc.columns)) + " |")
    for _, row in mc.head(15).iterrows():
        lines.append("| " + " | ".join(str(row[c])[:80] for c in mc.columns) + " |")
    lines.append("")
    lines.append(f"*({len(mc)} component rows total.)*")
    lines.append("")
    lines.append("## `Experiments` sheet columns (header row)")
    lines.append("")
    ex = pd.read_excel(MEDIA_XLSX, sheet_name="Experiments", header=0)
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

    vd = variance_decomposition_table(sample)
    audit_path = write_media_audit_md()
    emb = embedding_spotcheck()
    xl_cov = excel_experiment_coverage(experiments)

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
        },
    }
    (OUTPUT_DIR / "phase0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
