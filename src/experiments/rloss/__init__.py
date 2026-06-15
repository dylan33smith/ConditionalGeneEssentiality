"""R-LOSS — loss-family retest under ranking (ARCHITECTURE.md R3+, promoted ahead of R2).

Tests whether changing the OBJECTIVE (pointwise MSE -> pairwise/listwise ranking
losses, incl. ones that directly optimize NDCG) closes the gap to the chem-kNN
gate. Architecture (T5-A) and encoder (multihot) held constant. See R1-DEC-001.
"""
