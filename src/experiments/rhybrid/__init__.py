"""R-HYBRID — combine the global parametric model with the local chem-kNN.

R1+R-LOSS established that no global parametric model beats the local chem-kNN
gate (NDCG@5 0.485). R-HYBRID tests whether COMBINING them beats either alone —
i.e. whether the learned model carries signal COMPLEMENTARY to the local lookup
(fixing kNN's sparse-neighborhood / cross-gene blind spots). See
research_log/SCIENTIFIC_SYNTHESIS.md §7.

R-HYBRID-A (ensemble α-curve): per-gene z-scored convex combination of the
standalone model and chem-kNN; sweep α. If the curve peaks above kNN -> the model
adds complementary signal (build the residual hybrid next). If it peaks at pure
kNN -> the model adds nothing (characterization result).
"""
