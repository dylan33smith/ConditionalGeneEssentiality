# PLAN — current objective & next tasks

The working file. Update it whenever priorities change or a task lands.

## Current objective

Build the **top-k ranking objective** on the cleaned `src/ranking` core: a loss
that optimizes the *top* of each gene's condition ranking (where "find the top
stressors" actually lives), evaluated against the chem-kNN gate.

## Next tasks

1. **Wire `RankingBatch` into training (the top-k starting move).** The
   `src/data/datasets/ranking_batch.py` samplers (pointwise/pairwise/listwise)
   are built + tested but not yet used by the trainers (which roll their own
   batching). Adopt them in `src/ranking/train.py`. **This is expected to move
   the numbers** (different batch composition) — re-baseline `R-EVAL` and report
   before/after; it is a modeling change, not a regression.
2. **Top-k loss experiment.** Use the runner to compare top-focused losses
   (lambdarank / approxndcg, already in `src/ranking/losses`) with list-truncated
   variants; judge on NDCG@5 + precision@5 vs the gate. A new arm = one `ArmSpec`.
3. **Cold-gene diagnostic (optional characterization).** Quantify how far chem-kNN
   degrades on unseen genes — the one regime a global model could help — to bound
   the value of further modeling.

## Cleanup follow-ups (low priority, tracked)

- Migrate `r1/run.py` and `rconf/run.py` onto the runner (reval + rloss already are).
- Resolve the one layering note: `src/ranking/pipeline.py` imports fitness-loading
  + condition-key helpers from `src/experiments/r0/analyses.py`; extract those into
  `src/ranking/data/conditions.py` so `src/ranking` is fully self-contained.
- Optionally physically de-duplicate `eval/contract.py`'s within-gene-correlation
  helpers against `eval/harness.py` (kept separate during cleanup to avoid a silent
  behavior change to the contract test — verify equivalence first).

## Done (this branch)

The ranking-branch cleanup (see PROGRESS.md): regression gate, model/eval/loss/
pipeline extraction into `src/ranking`, shared runner + standardized reporting,
prune of the T-regime + dead code (−13k lines), docs scheme.
