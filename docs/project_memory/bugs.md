# Project memory — Bugs, quirks & proven fixes

Recurring traps and their confirmed fixes. Add an entry whenever a non-obvious bug
costs real time. Newest-ish first; grouped loosely.

---

## Data / git / environment

### `data/` symlink got committed and clobbered → data link broke (HIGH SEVERITY)
**Symptom:** `OSError: [Errno 40] Too many levels of symbolic links` on
`data/derived/canonical/v0/fitness_experiment_long.parquet`; `data` became a
self-referential symlink.
**Root cause:** `.gitignore` had `/data/` (matches a *directory* named data) but
**not** a `data` *symlink* (a symlink is a file). A stray `git add -A` then
committed the `data` symlink, and a later merge checked it out in another worktree,
clobbering the real `data` link.
**Fixes (both required):**
1. `.gitignore` must be `/data` (no trailing slash) so a `data` dir **or** symlink
   is ignored.
2. `data` must stay **untracked** (`git rm --cached data`). It is a machine-local
   symlink into the shared data root, not a repo file.
**Recovery:** the canonical parquet is rebuildable from `feba.db` via
`archive/data_processing/build_canonical_v0.py`; verify byte-identical against
`docs/canonical_build_manifest_v0.json` (the manifest stores expected sha256 +
bytes + rows). The rebuild reproduced sha256 `9b981201…` exactly.
**Lesson:** never `git add -A` when a `data` symlink is present; check
`git status` for an unexpected `data` entry before committing.
**Related gotcha:** switching branches in a worktree can DELETE the gitignored
`data` symlink (if a branch in the checkout's history tracked `data`, git removes
it on checkout) → `FileNotFoundError` on the parquet. Fix: recreate it —
`ln -sfn /home/ds85/projects/GeneEssentiality/data data` (the real data root).

### `orgs=null` ≠ the 23-org headline
`experiment.orgs=null` means **all** organisms (~107k eligible genes) and gives
different, *lower* numbers (~0.38/0.43). The published ~0.435/0.485 are on the **23
replicate-org subset** only. Always pass the explicit 23-org list for the headline
(it's in README / CLAUDE quick-start).

### LibreOffice/`soffice` is non-functional in this environment
Cannot convert/render anything ("source file could not be loaded", missing JRE).
So PPTX/PDF visual QA via `soffice → pdftoppm` is unavailable here; `python-pptx`
authoring works, but verify visuals by opening the file elsewhere.

---

## Training / determinism

### The regression gate is bit-exact — rely on it
With `torch.backends.cudnn.deterministic=True`, `benchmark=False`, and fixed
`torch`/`numpy` seeds, training reproduces **bit-exactly** on a fixed device. So a
behavior-preserving refactor yields Δ=0.0000 on `R-EVAL`; any non-zero drift is a
real behavior change → stop and investigate (don't widen tolerance to hide it).

### Pointwise losses must be ROW-batched, ranking losses GENE-batched
Gene-batched pointwise (MSE/Huber) underperforms row-batched (~0.34 vs ~0.39
NDCG@5) because each step sees too few distinct genes (low gradient diversity).
**Fix:** dispatch by loss family — pointwise → row-batched; pairwise/listwise →
gene-batched (each padded to L conditions with a mask). See `src/ranking/train.py`.

### ApproxNDCG didn't train (vanishing smooth-sort gradient)
The differentiable-sort surrogate had near-zero gradient at the default
temperature. **Fix:** set `temp=0.5`.

### LambdaRank collapses full-list Spearman (expected, not a bug)
LambdaRank reaches good NDCG@5 but tanks Spearman (~0.08) — it's purely
top-focused. Early-stopping on NDCG@5 lets it overfit the top. Make tests
loss-aware (full-Spearman for full-list losses, NDCG@k for top-focused ones)
rather than asserting one threshold for all.

### OOM materializing per-row chemistry
Building a per-row chemistry matrix for ~11M train rows × dims (~90 GB) OOMs.
**Fix:** the "memory-safe per-experiment gather" — build a small per-*experiment*
chemistry matrix and index into it per batch (`chem_matrix_for_rows`).

---

## Evaluation / metrics

### `condition_chemistry` cache collision across org-sets
A namespaced-by-nothing cache returned a stale/empty (0-feature) chemistry map when
a different org-set was loaded (e.g. Keio reading an 8-org cache). **Fix:**
namespace the cache dir by an org-set hash.

### Noise floor: use MEAN, not MEDIAN aggregation
`retrieval_noise_floor` with median aggregation collapsed binary precision@1 to 0
(misleading). **Fix:** aggregate per-gene metrics with MEAN.

### chem-NULL / per-condition-mean is NaN under cold columns
Under condition-holdout the val conditions are 100% cold, so a per-condition
train-mean baseline returns NaN for all val rows. The split-specific baselines
(chem-NULL = nearest-condition profile, chem-kNN, inductive-MF) replace it; vanilla
MF only applies to the `cell_holdout` diagnostic, not the primary split.

---

## Imports / packaging (from the cleanup)

### Private helpers aren't re-exported by the package `__init__`
`from src.ranking.eval import _cosine_dist_matrix` fails — underscore-prefixed
helpers are intentionally not in the package API. **Fix:** import privates from the
submodule (`from src.ranking.eval.harness import _cosine_dist_matrix`).

### Package `__init__.py` shows as "dead" in import graphs (false positive)
An import-reachability sweep flags live packages' `__init__.py` as unreferenced
(they have no inbound *import* edges by nature). They are structurally required —
never delete a live package's `__init__.py`.

### YAML: don't mix list items and keys under one node
A `metric_contract.yaml` node mixing `- item` list entries with `key:` mappings
fails to parse. **Fix:** use a `names:` sub-mapping.
