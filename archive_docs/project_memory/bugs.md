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

### Entire data ROOT vanished mid-session → recovered by re-downloading the live Fitness Browser (CRITICAL)
**Symptom (2026-07-06):** the whole external data root `/home/ds85/projects/GeneEssentiality/`
disappeared during a session — `feba.db`, the canonical parquet, and the embeddings all
unreachable (the `data` symlink was intact but its target directory was gone). No copy existed
under `/home/ds85` or the `/data` xfs mount.
**Root cause:** unknown — the sibling project directory the `data` symlink points at was
deleted/moved off the machine (more severe than the earlier symlink-clobber incident; this was the
target, not the link).
**Recovery (raw fitness data):** re-downloaded the live DB with a browser UA (LBL blocks default
curl UA with 403):
```bash
mkdir -p /home/ds85/projects/GeneEssentiality/data/raw && cd $_
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
curl -L -A "$UA" -C - --retry 8 -o feba.db  https://fit.genomics.lbl.gov/cgi_data/feba.db   # 9.08 GB, range-resumable
curl -L -A "$UA" -C - --retry 8 -o aaseqs   https://fit.genomics.lbl.gov/cgi_data/aaseqs
```
Pinned Figshare releases (stable/citable) are alternatives: Nov 2020 (id 13172087), June/Nov 2021
(5134840 / 16913530) — each ships `db.tar.gz` (feba.db + aaseqs).
**IMPORTANT — the re-downloaded DB is a NEWER release, not a byte-match.** Live db = **62 organisms /
9,532 experiments / 33.8M GeneFitness rows**, sha256 `b6627137…`; the project's pinned db was 48 orgs /
7,552 exps / 27.4M rows, sha256 `627f2097…`. So the canonical parquet will NOT byte-reproduce, and all
headline numbers (0.432/0.485 etc.) must be **re-derived + the R-EVAL gate re-pinned** on the new release
before they're treated as current.
**Still missing after the raw restore (NOT on the Fitness Browser):**
1. `data/media_composition.xlsx` (project sidecar; sha `b694685a…`) — required by
   `build_canonical_v0.py`; not found anywhere. Rebuild of the media columns is blocked until it's
   located/recreated.
2. `data/processed/ProtLM_embeddings_layer8/*.pt` — regenerate from `aaseqs` via the ProteomeLM
   encoder pipeline.
**Lesson:** the data root is a single point of failure. Consider a checksummed backup of `feba.db` +
`media_composition.xlsx` + the embeddings on the `/data` xfs mount (988 GB free), and keep the
build-manifest sha256s (`docs/canonical_build_manifest_v0.json`) as the integrity oracle.

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

### feba.db `Cofit` is TARGET LEAKAGE on the cold-gene split — never use it as a gene–gene edge
**What it is:** `feba.db` ships a precomputed `Cofit` table = Pearson correlation
between gene fitness *profiles*, computed over **all** experiments (split-blind).
**Why it leaks (cold-gene / condition-holdout):** the `cold_gene` split holds out
*whole genes*, but a `Cofit` correlation spans the held-out val conditions, so a cold
gene's top co-fitness neighbor's train profile reconstructs that gene's **own held-out
answer key**. Smoking gun (round-2 evolutionary-search verification, Keio/Caulo/MR1):
cofit-only NDCG@5 ≈ **0.73–0.75**, which *exceeds even the warm chem-kNN gate (~0.485)*
— physically impossible without a leak; the rank-1 cofit-neighbor's train profile
correlates with the val gene's TRUE held-out profile at median r ≈ 0.56 (63% of genes
> 0.5). Inverted ablation confirms it (cofit-only > any fusion that includes it).
**Rule:** never read `feba.db Cofit` (or any all-experiment co-fitness matrix) into a
cold-gene or condition-holdout method. Co-fitness must be **recomputed train-only**
(gene fitness vectors over TRAIN conditions only). On the cold-gene split a train-only
co-fitness edge is *undefined* (a held-out gene has zero train rows to correlate), so
the **frozen embedding is the only leakage-free gene–gene edge source there** — which
is also why the honest gene-axis methods (GENE-NW) reduce to the R-COLD embedding
signal rather than adding new information.

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
