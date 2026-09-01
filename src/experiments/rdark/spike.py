"""R-DARK spike — dark-genome residual miner, Steps 1-3 (go/no-go).

Step 1: per-(gene,condition) observed fit + replicate sigma(condition); leave-one-
        condition-out chem-kNN expectation; studentized residual r* = (obs-exp)/sigma.
Step 2: orphan (dark-genome) filter from feba.db Gene (type=1 + unknown-function desc).
Step 3: column-permutation NULL (shuffle gene labels within each condition, destroying
        gene-specific cross-condition structure) -> is the top ORPHAN |r*| above chance?

No training, no GPU. The real chem-kNN pass reuses the canonical harness
(`chemistry_knn_predict(exclude_self=True)`); a fast inline kNN (same fixed neighbor
structure) drives the permutation null and is asserted to reproduce the harness.

Run:  PYTHONPATH=. python src/experiments/rdark/spike.py
"""
from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pandas as pd

from src.ranking.pipeline import _load_canonical
from src.data.datasets.condition_chemistry import load_condition_chemistry_features
from src.ranking.eval.harness import chemistry_knn_predict, _cosine_dist_matrix

DB = "data/raw/feba.db"
ORGS = ["Keio", "Caulo", "MR1"]
K = 5
N_PERM = 200
TOPN = 50                     # top-N orphan |r*| summarized as the discovery statistic
RSTAR_THRESH = 5.0            # "surprise" threshold for the count statistic
RNG = np.random.default_rng(0)

_ORPHAN_RE = r"hypothetical|duf\d|uncharacterized|unknown function|domain of unknown|putative uncharacterized"


def load_gene_table(org: str) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    g = pd.read_sql("SELECT locusId, type, gene, sysName, desc FROM Gene WHERE orgId=?",
                    con, params=[org])
    con.close()
    desc = g["desc"].fillna("").str.lower().str.strip()
    g["orphan"] = (g["type"] == 1) & (
        (desc == "") | desc.str.contains(_ORPHAN_RE, regex=True, na=False))
    g["locusId"] = g["locusId"].astype(str)
    return g.set_index("locusId")


def sigma_condition(raw: pd.DataFrame) -> tuple[dict, float, dict]:
    """Pooled within-(gene,condition) replicate std per condition (cells with n_rep>=2),
    with fallback: condition -> its expGroup median -> global median.
    Returns (sigma_by_condition, global_sigma, coverage_info)."""
    raw = raw.copy()
    raw["cell_mean"] = raw.groupby(["gene_key", "condition_key"])["fit"].transform("mean")
    raw["cell_n"] = raw.groupby(["gene_key", "condition_key"])["fit"].transform("size")
    raw["dev"] = raw["fit"] - raw["cell_mean"]
    rep = raw[raw["cell_n"] >= 2]
    direct = {}
    if len(rep):
        ss = rep.groupby("condition_key")["dev"].apply(lambda x: float((x ** 2).sum()))
        nrows = rep.groupby("condition_key")["dev"].size()
        ncells = rep.groupby("condition_key").apply(
            lambda d: d.groupby("gene_key").ngroups)
        dof = (nrows - ncells).clip(lower=1)
        sig = np.sqrt(ss / dof)
        direct = {c: float(v) for c, v in sig.items() if np.isfinite(v) and v > 0}
    # expGroup map per condition (first)
    cond_grp = raw.groupby("condition_key")["expGroup"].first().to_dict()
    grp_med = {}
    if direct:
        tmp = pd.DataFrame({"condition_key": list(direct), "sigma": list(direct.values())})
        tmp["expGroup"] = tmp["condition_key"].map(cond_grp)
        grp_med = tmp.groupby("expGroup")["sigma"].median().to_dict()
    global_sig = float(np.median(list(direct.values()))) if direct else 1.0
    all_conds = raw["condition_key"].unique()
    sigma_map = {}
    src = {"direct": 0, "expgroup": 0, "global": 0}
    for c in all_conds:
        if c in direct:
            sigma_map[c] = direct[c]; src["direct"] += 1
        elif cond_grp.get(c) in grp_med and np.isfinite(grp_med[cond_grp.get(c)]):
            sigma_map[c] = float(grp_med[cond_grp[c]]); src["expgroup"] += 1
        else:
            sigma_map[c] = global_sig; src["global"] += 1
    return sigma_map, global_sig, src


def inline_rstar(Marr: np.ndarray, knn_idx: np.ndarray, sigma_arr: np.ndarray) -> np.ndarray:
    """expected[g,c] = nanmean over the k nearest OTHER conditions; r* = (M-exp)/sigma(c)."""
    exp = np.full(Marr.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j in range(Marr.shape[1]):
            sub = Marr[:, knn_idx[j]]                       # [n_genes, k]
            allnan = np.all(np.isnan(sub), axis=1)
            col = np.nanmean(np.where(np.isnan(sub), np.nan, sub), axis=1)
            col[allnan] = np.nan
            exp[:, j] = col
    return (Marr - exp) / sigma_arr[None, :]


def top_orphan_stats(rstar: np.ndarray, orphan_rows: np.ndarray) -> dict:
    vals = np.abs(rstar[orphan_rows])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"max": np.nan, "topN_mean": np.nan, "n_above": 0}
    top = np.sort(vals)[::-1][:TOPN]
    return {"max": float(vals.max()),
            "topN_mean": float(top.mean()),
            "n_above": int((vals > RSTAR_THRESH).sum())}


def run_org(org: str, raw_all: pd.DataFrame, cf: dict) -> dict:
    raw = raw_all[raw_all["orgId"] == org].dropna(subset=["fit", "condition_key", "gene_key"]).copy()
    raw = raw[raw["condition_key"].isin(cf)]
    if raw.empty:
        return {"org": org, "error": "no rows with chemistry features"}

    sigma_map, global_sig, src = sigma_condition(raw)

    pooled = (raw.groupby(["gene_key", "condition_key"])
              .agg(fit=("fit", "mean"), n_rep=("fit", "size")).reset_index())

    # --- REAL chem-kNN LOO via the canonical harness (ground truth) ---
    knn = chemistry_knn_predict(pooled, pooled, cf, k=K, exclude_self=True)
    pooled["knn"] = knn.values
    pooled = pooled.dropna(subset=["knn"]).copy()
    pooled["sigma"] = pooled["condition_key"].map(sigma_map)
    pooled["resid"] = pooled["fit"] - pooled["knn"]
    pooled["rstar"] = pooled["resid"] / pooled["sigma"]

    # orphan flag
    gtab = load_gene_table(org)
    pooled["locusId"] = pooled["gene_key"].str.split(":", n=1).str[1]
    pooled["orphan"] = pooled["locusId"].map(gtab["orphan"]).fillna(False).astype(bool)
    pooled["desc"] = pooled["locusId"].map(gtab["desc"]).fillna("")

    # --- inline kNN (fixed neighbor structure) + correctness check vs harness ---
    M = pooled.pivot_table(index="gene_key", columns="condition_key", values="fit")
    conds = list(M.columns); genes = list(M.index)
    feat = np.vstack([cf[c] for c in conds])
    dist = _cosine_dist_matrix(feat, feat)
    np.fill_diagonal(dist, np.inf)                          # exclude self
    knn_idx = np.argsort(dist, axis=1)[:, :K]
    Marr = M.to_numpy(float)
    sigma_arr = np.array([sigma_map[c] for c in conds])
    rstar_inline = inline_rstar(Marr, knn_idx, sigma_arr)

    # check: inline expected matches harness knn on the same cells
    gi = {g: i for i, g in enumerate(genes)}; cj = {c: j for j, c in enumerate(conds)}
    exp_inline = Marr - rstar_inline * sigma_arr[None, :]
    diffs = []
    for _, r in pooled.sample(min(2000, len(pooled)), random_state=0).iterrows():
        e = exp_inline[gi[r["gene_key"]], cj[r["condition_key"]]]
        if np.isfinite(e):
            diffs.append(abs(e - r["knn"]))
    max_diff = float(np.nanmax(diffs)) if diffs else np.nan

    orphan_rows = np.array([bool(pooled.set_index("gene_key")["orphan"].groupby(level=0).first().get(g, False))
                            for g in genes])

    # --- Step 3: column-permutation NULL ---
    real_stat = top_orphan_stats(rstar_inline, orphan_rows)
    obs_idx_by_col = [np.where(np.isfinite(Marr[:, j]))[0] for j in range(len(conds))]
    null = {"max": [], "topN_mean": [], "n_above": []}
    for _ in range(N_PERM):
        Mp = Marr.copy()
        for j in range(len(conds)):
            obs = obs_idx_by_col[j]
            if obs.size > 1:
                Mp[obs, j] = Mp[RNG.permutation(obs), j]
        rp = inline_rstar(Mp, knn_idx, sigma_arr)
        s = top_orphan_stats(rp, orphan_rows)
        for kk in null:
            null[kk].append(s[kk])

    def emp_p(real, nullvals):
        nv = np.array(nullvals, float)
        return float((np.sum(nv >= real) + 1) / (len(nv) + 1))

    res = {
        "org": org,
        "n_genes": len(genes), "n_orphan_genes": int(orphan_rows.sum()),
        "n_conditions": len(conds), "n_cells_scored": int(len(pooled)),
        "n_orphan_cells": int(pooled["orphan"].sum()),
        "sigma_src": src, "harness_vs_inline_max_diff": max_diff,
        "real": real_stat,
        "null_mean": {kk: float(np.mean(v)) for kk, v in null.items()},
        "null_p95": {kk: float(np.percentile(v, 95)) for kk, v in null.items()},
        "emp_p": {kk: emp_p(real_stat[kk], null[kk]) for kk in null},
    }
    # preview top orphan hits (NOT validated — eyeball only)
    top_hits = (pooled[pooled["orphan"]].reindex(
        pooled[pooled["orphan"]]["rstar"].abs().sort_values(ascending=False).index)
        .head(15)[["gene_key", "condition_key", "fit", "knn", "rstar", "n_rep", "desc"]])
    res["top_hits"] = top_hits
    return res


def main():
    print(f"[R-DARK spike] orgs={ORGS} K={K} N_PERM={N_PERM} TOPN={TOPN} thresh={RSTAR_THRESH}")
    print("loading canonical fitness + chemistry features ...")
    raw_all = _load_canonical(ORGS)
    cf = load_condition_chemistry_features(orgs=ORGS)
    print(f"  loaded {len(raw_all):,} rows; {len(cf):,} conditions with chemistry features")
    results = []
    for org in ORGS:
        print(f"\n===== {org} =====")
        r = run_org(org, raw_all, cf)
        results.append(r)
        if "error" in r:
            print("  ERROR:", r["error"]); continue
        print(f"  genes={r['n_genes']} (orphan {r['n_orphan_genes']}), conds={r['n_conditions']}, "
              f"cells={r['n_cells_scored']} (orphan {r['n_orphan_cells']})")
        print(f"  sigma source: {r['sigma_src']}  | harness-vs-inline max |Δexp|={r['harness_vs_inline_max_diff']:.2e}")
        for stat in ("max", "topN_mean", "n_above"):
            print(f"  [{stat:9s}] real={r['real'][stat]:.3f}  null_mean={r['null_mean'][stat]:.3f}  "
                  f"null_p95={r['null_p95'][stat]:.3f}  emp_p={r['emp_p'][stat]:.4f}")
        print("  top orphan hits (UNVALIDATED preview):")
        with pd.option_context("display.width", 200, "display.max_colwidth", 45):
            print(r["top_hits"].to_string(index=False))

    print("\n===== GO/NO-GO SUMMARY =====")
    print(f"{'org':8s} {'topN_mean real/null':>22s} {'emp_p(topN)':>12s} {'emp_p(n_above)':>15s} {'verdict':>10s}")
    for r in results:
        if "error" in r:
            continue
        v = "SIGNAL" if (r["emp_p"]["topN_mean"] < 0.05 or r["emp_p"]["n_above"] < 0.05) else "null"
        print(f"{r['org']:8s} {r['real']['topN_mean']:8.3f}/{r['null_mean']['topN_mean']:<8.3f}"
              f"     {r['emp_p']['topN_mean']:>10.4f} {r['emp_p']['n_above']:>15.4f} {v:>10s}")
    print("\nInterpretation: SIGNAL = top orphan studentized residuals exceed the gene-label-"
          "permutation null (the gene-specific chemical surprise is real). null = stop cheaply.")


if __name__ == "__main__":
    main()
