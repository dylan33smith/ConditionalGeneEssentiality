# Project memory — Progress

Where the project actually is. Read this first when picking the work back up.
The dated log below is append-only (newest first) — never rewrite past entries.

---

## Where we left off (2026-06-17)

- **Branch:** `ranking` (trunk), pushed and in sync with `origin/ranking`.
- **Repo:** cleaned + modular — self-contained `src/ranking/` core, shared runner,
  bit-exact `R-EVAL` regression gate. Tests green (~125 ranking tests).
- **Data:** recovered byte-for-byte after the `data` symlink incident (canonical
  parquet rebuilt from `feba.db`, sha256 matches the manifest). `data` is now an
  untracked, gitignored machine-local symlink. See `bugs.md`.
- **Headline anchor (in progress / just completed):** the corrected 23-org / 3-seed
  `R-EVAL` is confirming the published numbers — seed-2 showed model NDCG@5 **0.4345**
  / chem-kNN **0.4852** (right on ~0.435/0.485). When it finishes it writes the
  `full` entry into `data_contract/ranking/reval_baseline.json`.
- **Open loose ends:** (1) commit the `.gitignore /data/`→`/data` fix (it was left
  uncommitted) — being handled now; (2) delete the merged throwaway branch
  `ranking-cleanup` once this session's worktree is freed; (3) write the `full`
  baseline entry once the headline run lands.

## Next tasks (the top-k objective)

1. **Wire `RankingBatch` into training** — the `src/data/datasets/ranking_batch.py`
   samplers (pointwise/pairwise/listwise) are built + tested but not yet used by the
   trainers. Adopt them in `src/ranking/train.py`. **This will move the numbers by
   design** (different batch composition) — re-baseline `R-EVAL` and report
   before/after; it's a modeling change, not a regression.
2. **Top-k loss experiment** — use the runner to compare top-focused losses
   (lambdarank / approxndcg, already in `src/ranking/losses`) and list-truncated
   variants; judge on NDCG@5 + precision@5 vs the chem-kNN gate. A new arm = one
   `ArmSpec`.
3. **Cold-gene diagnostic (optional)** — quantify how far chem-kNN degrades on
   unseen genes (the one regime a global model could help), to bound the value of
   further modeling.

### Lower-priority follow-ups
- Migrate `r1/run.py` and `rconf/run.py` onto the shared runner (reval + rloss
  already are).

---

## Log

### 2026-06-17 — `data` symlink incident + byte-exact recovery
While running the 23-org headline anchor, the `data` link resolved to a self-loop
and the canonical parquet was unreachable. Root cause + permanent fix in `bugs.md`
(gitignore `/data` + keep `data` untracked). **Recovery:** rebuilt all four
canonical parquets from `feba.db` via `archive/data_processing/build_canonical_v0.py`;
`fitness_experiment_long.parquet` verified byte-identical to the original
(27,410,721 rows, 805,220,433 bytes, sha256 `9b981201…` = manifest). No permanent
loss. Re-launched the 23-org headline anchor (confirming ~0.435/0.485).

### 2026-06-15 — Documentation consolidation (this scheme)
Consolidated the scattered top-level docs into `README.md` (single source of truth
for current state) + `docs/project_memory/{decisions,bugs,progress}.md` (modular AI
working memory). Added the Memory Protocol hook to `CLAUDE.md`. Deleted the
now-redundant `ARCHITECTURE.md` / `PLAN.md` / `PROGRESS.md` (folded in).

### 2026-06-15 — Layering fixes (post-cleanup polish)
- **Fix A** — condition-key helpers extracted from `src/experiments/r0/analyses.py`
  into `src/data/datasets/conditions.py` (data layer). Net: nothing in `src/data` or
  `src/ranking` imports upward into `src/experiments`.
- **Fix B** — deduped `eval/contract.py`'s per-gene-correlation against the
  canonical `harness.per_gene_correlations` (verified equivalent; test_ranking_metrics
  17/17). Kept contract's flat bootstrap distinct from harness's hierarchical one
  (different methods; merging would swap a flat CI for a hierarchical one).
Regression: R-EVAL fast gate bit-exact (Δ=0.0000); pytest 119 passed.

### 2026-06-15 — Ranking-branch cleanup (reorganize + aggressive prune + modular runner)
**Goal:** clean `ranking` branch as the basis for the top-k objective. Each step
gated by R-EVAL (bit-exact) + pytest:
1. R-EVAL regression harness (fast gate: model 0.4516 / chem-kNN 0.5091 on
   Keio+Caulo+MR1; baseline in `data_contract/ranking/reval_baseline.json`).
2. Extracted the model (`ResidualBlock` + `AdapterResidualMLP`) into `src/ranking/models.py`.
3. Decoupled ranking from the T-tier (fingerprint loader, eval `harness`+`contract`,
   loss family moved into `src/ranking`); no ranking module imports any `tier*`.
4. Relocated pipeline + trainers into `src/ranking/{pipeline,train}.py`; added
   `src/ranking/runner.py` (ArmSpec + run_experiment + standardized report). `reval`
   and `rloss` handlers became thin runner specs.
5. Pruned the T-regime + dead code (130 files / −13.4k lines): stage0-5, tier1-6,
   diagnostics, `src/{train,models,domain,training,evaluation}`, failed hybrids, dead
   data utils, 51 T-regime tests, T*/stage* configs, root cruft; deregistered
   S0-S5/T1-T6 handlers. Learnings preserved in the ledger + SCIENTIFIC_SYNTHESIS;
   mapping in `docs/PRUNED_INDEX.md`.
6. Docs scheme (since superseded by the consolidation above).

**Learned:** training is deterministic → the gate is bit-exact, which made every
refactor verifiable. Kept the reusable top-k substrate (loss family, `RankingBatch`,
eval/baseline harness); deleted only single-use glue and the T-regime.

### Earlier (T-regime + R-regime exploration)
Full history is in the decision ledger `research_log/decisions/**` and the narrative
`research_log/SCIENTIFIC_SYNTHESIS.md`. Highlights: T-regime cross-org regression
failed (within-gene ranking ≈ random); reframed to within-org ranking (R); R1
(encoder), R-LOSS (objective), capacity, and R-HYBRID-A/B (model+kNN hybrids) all
failed to beat the chem-kNN gate; R-CONF showed the negative is noise-robust.
