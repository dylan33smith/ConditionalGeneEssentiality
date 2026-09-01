"""R-DARK spike v2 — replicate-CONFIRMED go/no-go (fixes v1's two flaws).

v1 found that the naive top "surprises" are dominated by n_rep=1 single measurements
(σ-fallback artifact) and that a gene-label-permutation null tests predictability, not
surprise-beyond-noise. v2 does the honest test the proposal called for:

  - restrict to cells with >=2 replicate measurements (the only cells where a surprise
    CAN be validated);
  - split each cell's replicates into halves A,B; a REAL surprise must appear in BOTH
    (concordant sign, both |z|>THR) where z = (half - knn_expected)/sigma(condition);
  - NULL = shuffle the B-half across genes WITHIN each condition (breaks a gene's own
    replicate pairing, preserves the per-condition noise scale). Repeat; compare the
    count of CONFIRMED orphan surprises real vs null.
  - secondary: confirmed-surprise RATE orphan vs annotated (label-permutation p).

knn expectation reuses the canonical harness (chemistry_knn_predict, exclude_self).
Run:  PYTHONPATH=. python src/experiments/rdark/spike_repro.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.pipeline import _load_canonical
from src.data.datasets.condition_chemistry import load_condition_chemistry_features
from src.ranking.eval.harness import chemistry_knn_predict
from src.experiments.rdark.spike import load_gene_table, sigma_condition, ORGS, K

THR = 3.0          # per-half studentized-residual threshold for a "surprise"
N_PERM = 500
TOPN = 50
RNG = np.random.default_rng(0)


def split_halves(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (gene,condition): mean of replicate-half A and half B (needs n_rep>=2)."""
    rs = raw.sort_values(["gene_key", "condition_key", "expName"]).copy()
    grp = rs.groupby(["gene_key", "condition_key"])
    rs["rk"] = grp.cumcount()
    rs["n"] = grp["fit"].transform("size")
    rs["half"] = np.where(rs["rk"] < np.ceil(rs["n"] / 2), "A", "B")
    cell = (rs.groupby(["gene_key", "condition_key", "half"])["fit"].mean()
            .unstack("half"))
    cell["n_rep"] = grp["fit"].size().reindex(cell.index).values
    cell = cell.dropna(subset=["A", "B"]).reset_index()      # both halves present => n_rep>=2
    return cell


def run_org(org, raw_all, cf):
    raw = raw_all[raw_all["orgId"] == org].dropna(subset=["fit", "condition_key", "gene_key"]).copy()
    raw = raw[raw["condition_key"].isin(cf)]
    sigma_map, global_sig, _ = sigma_condition(raw)

    pooled = (raw.groupby(["gene_key", "condition_key"])
              .agg(fit=("fit", "mean")).reset_index())
    knn = chemistry_knn_predict(pooled, pooled, cf, k=K, exclude_self=True)
    pooled["exp"] = knn.values
    pooled = pooled.dropna(subset=["exp"])

    cell = split_halves(raw).merge(
        pooled[["gene_key", "condition_key", "exp"]], on=["gene_key", "condition_key"])
    cell["sigma"] = cell["condition_key"].map(sigma_map)
    cell["zA"] = (cell["A"] - cell["exp"]) / cell["sigma"]
    cell["zB"] = (cell["B"] - cell["exp"]) / cell["sigma"]

    gtab = load_gene_table(org)
    cell["locusId"] = cell["gene_key"].str.split(":", n=1).str[1]
    cell["orphan"] = cell["locusId"].map(gtab["orphan"]).fillna(False).astype(bool)
    cell["desc"] = cell["locusId"].map(gtab["desc"]).fillna("")

    zA = cell["zA"].to_numpy(); zB = cell["zB"].to_numpy()
    orphan = cell["orphan"].to_numpy()
    confirmed = (np.sign(zA) == np.sign(zB)) & (np.abs(zA) > THR) & (np.abs(zB) > THR)

    # condition grouping for the within-condition B-shuffle null
    cond_codes = pd.factorize(cell["condition_key"])[0]
    idx_by_cond = [np.where(cond_codes == c)[0] for c in range(cond_codes.max() + 1)]

    def confirmed_count(zb):
        conf = (np.sign(zA) == np.sign(zb)) & (np.abs(zA) > THR) & (np.abs(zb) > THR)
        return int((conf & orphan).sum())

    real_conf_orphan = confirmed_count(zB)
    null_counts = []
    for _ in range(N_PERM):
        zb = zB.copy()
        for ix in idx_by_cond:
            if ix.size > 1:
                zb[ix] = zb[RNG.permutation(ix)]
        null_counts.append(confirmed_count(zb))
    null_counts = np.array(null_counts)
    emp_p = float((np.sum(null_counts >= real_conf_orphan) + 1) / (N_PERM + 1))

    # orphan vs annotated confirmed RATE + label-permutation p
    rate_orph = float(confirmed[orphan].mean()) if orphan.any() else np.nan
    rate_anno = float(confirmed[~orphan].mean()) if (~orphan).any() else np.nan
    real_diff = rate_orph - rate_anno
    lab_null = []
    for _ in range(N_PERM):
        perm = RNG.permutation(orphan)
        lab_null.append(confirmed[perm].mean() - confirmed[~perm].mean())
    lab_p = float((np.sum(np.array(lab_null) >= real_diff) + 1) / (N_PERM + 1))

    # confirmed orphan hits, ranked by reproducible magnitude min(|zA|,|zB|)
    cell["repro_mag"] = np.minimum(np.abs(zA), np.abs(zB))
    hits = cell[confirmed & orphan].sort_values("repro_mag", ascending=False)
    topN_mean = float(hits["repro_mag"].head(TOPN).mean()) if len(hits) else np.nan

    print(f"\n===== {org} =====")
    print(f"  replicated cells (n_rep>=2): {len(cell):,}  (orphan {int(orphan.sum()):,}); "
          f"conditions with replication: {len(idx_by_cond)}")
    print(f"  CONFIRMED orphan surprises (|z|>{THR} both halves, concordant): real={real_conf_orphan}  "
          f"null_mean={null_counts.mean():.1f}  null_p95={np.percentile(null_counts,95):.1f}  emp_p={emp_p:.4f}")
    print(f"  confirmed RATE  orphan={rate_orph:.5f}  annotated={rate_anno:.5f}  "
          f"diff={real_diff:+.5f}  label_perm_p={lab_p:.4f}")
    print(f"  top confirmed orphan repro_mag (top{TOPN} mean)={topN_mean:.3f}")
    if len(hits):
        with pd.option_context("display.width", 200, "display.max_colwidth", 42):
            print(hits.head(12)[["gene_key", "condition_key", "A", "B", "exp",
                                  "zA", "zB", "n_rep", "desc"]].to_string(index=False))
    return {"org": org, "real": real_conf_orphan, "null_mean": float(null_counts.mean()),
            "emp_p": emp_p, "rate_orph": rate_orph, "rate_anno": rate_anno, "label_p": lab_p,
            "n_confirmed_orphan": int(len(hits))}


def main():
    print(f"[R-DARK spike v2] replicate-confirmed | orgs={ORGS} K={K} THR={THR} N_PERM={N_PERM}")
    raw_all = _load_canonical(ORGS)
    cf = load_condition_chemistry_features(orgs=ORGS)
    res = [run_org(o, raw_all, cf) for o in ORGS]
    print("\n===== GO/NO-GO (replicate-confirmed) =====")
    print(f"{'org':8s} {'confirmed real/null':>22s} {'emp_p':>8s} {'orph vs anno rate':>22s} {'label_p':>8s}  verdict")
    for r in res:
        signal = (r["emp_p"] < 0.05) and (r["n_confirmed_orphan"] > 0)
        enriched = r["label_p"] < 0.05
        v = "SIGNAL" if signal else "null"
        if signal and enriched:
            v = "SIGNAL+enriched"
        print(f"{r['org']:8s} {r['real']:>9d}/{r['null_mean']:<10.1f}  {r['emp_p']:>7.4f}  "
              f"{r['rate_orph']:.5f} vs {r['rate_anno']:.5f}  {r['label_p']:>7.4f}  {v}")
    print("\nSIGNAL = orphan surprises reproduce across replicates beyond the within-condition "
          "replicate-shuffle null. enriched = orphans more surprising than annotated genes.")


if __name__ == "__main__":
    main()
