"""src.ranking — the conditional-essentiality ranking objective.

Self-contained package for the within-organism top-k ranking task: data layer,
model, loss family, evaluation + baselines, and a shared train→eval→report
runner. This is the basis for ongoing top-k work; the legacy T-regime has been
pruned (see the pruned→learning index in the docs).
"""
