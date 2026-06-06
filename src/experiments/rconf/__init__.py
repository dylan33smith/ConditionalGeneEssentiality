"""R-CONF — measurement-confidence characterization (idea 1).

Stratifies the within-org ranking evaluation by per-cell measurement confidence
(`abs_t`, the Wetmore-et-al-2015 moderated t carried in the canonical data) to
answer: how much of the model-vs-kNN gap and the gap to the replicate ceiling is
imposed by label noise vs structure? Also tests a t-confidence-weighted training
arm. This is CHARACTERIZATION (no promotion gate).
"""
