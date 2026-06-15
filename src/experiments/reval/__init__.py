"""R-EVAL — the locked regression check for the ranking-branch cleanup.

Reproduces, in ONE command, the headline ranking comparison: the locked-best
arm (pointwise_huber + multihot_425) vs the chem-kNN baseline, on a fixed split
seed, at k=5, with model + kNN scored on the identical eligible val gene set.

Two modes (set by config):
  * fast  — small fixed org subset, 1 seed: the per-migration-step regression gate.
  * full  — 23 replicate orgs, 3 seeds: the published headline (~0.435 / ~0.485).

If a baseline JSON exists it is compared within tolerance and PASS/FAIL printed,
so any refactor that silently moves a number is caught.
"""
