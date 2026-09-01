# Reframe candidates — different QUESTIONS in the conditional-essentiality space

**Date:** 2026-06-25 · **Method:** multi-agent reframe search (recon on `feba.db` → 5 category
generators → triage → 2 adversarial verifiers/reframe [data-feasibility + wall-escape] → synth).
**Why:** the warm "beat chem-kNN at within-gene condition ranking" objective is memorization-capped
(see [LITERATURE_local_vs_global_memorization.md](LITERATURE_local_vs_global_memorization.md)). The
only honest path is a *different question*, not a better ranker. **Steer:** stop predicting held-out
fitness of a *seen* gene — every version of that is the wall. Predict something the lookup
*structurally cannot read* (a residual it can't explain, or a relational/annotation label with no
near-condition analog).

## Ranking

| Rank | Reframe | Escapes wall? | Buildable now? | Effort | Verdict |
|---|---|---|---|---|---|
| **1** | **Dark-genome residual miner** | **Yes, by construction** | Yes, no new labels | days | **PURSUE — spike first** |
| **2** | **De-leaked co-essentiality → functional linkage** | Yes (fitness is feature) | Yes, no new labels | 1–2 wk | **PURSUE — if scoped tightly** |
| 3 | Essentiality-fingerprint → SEED function | Yes (static label) | Yes | med | MAYBE — narrow delta claim |
| 4 | Cross-org MoA classification | Partial (compound-kNN re-wall) | Yes + curated labels | med-high | MAYBE — strict joint split only |
| 5 | User GNN — inductive **cold-chemical** arm | Yes (held-out compound) | Yes (fingerprints exist) | high | MAYBE — only after a kill-switch |
| 6 | SpecOG phenotype conservation | Partial | No — label must be rebuilt | high | AVOID as specified (leak) |
| — | User GNN — transductive / cold-**gene** arm | **No — same wall** | Yes | — | AVOID (baseline only) |

## #1 — Dark-genome residual miner (strongest escape; weaponizes the negative result)
> Full implementation-ready plan: [PROPOSAL_dark_genome_residual_miner.md](PROPOSAL_dark_genome_residual_miner.md)
**Question:** among genes with *no* informative annotation (hypothetical/DUF), which show a
chemically-specific essentiality signal the chem-kNN/population lookup **cannot** explain, and which
condition does each point to? **Target:** per-orphan score = max over conditions of the studentized
lookup residual `r* = (fit_obs − fit_pred_knn)/σ_replicate(cond)`, on orphan genes only.
**Why it escapes:** chem-kNN becomes the *null model* and the residual is the product — "lookup
beats learning" is structurally inapplicable because it *defines* the target. **Validation
(non-circular):** cross-organism **ortholog concordance** of the residual signal (degree-preserving
permutation null) is the primary; SpecificPhenotype enrichment is a *secondary* validator only —
it's a thresholded contrast of the *same* fit/t, so it's **partially circular**. **Go/no-go gate:**
replicate-swap / sign-scramble null — if top-orphan residuals are indistinguishable from it, stop
cheaply. **Ceiling:** modest — a short, cross-org-corroborated dark-gene nomination list (tens to
low-hundreds), each with a stressing condition. Novel + publishable even at small effect; the
negative-result-as-method framing holds regardless. **Effort:** days; reuses the harness.

## #2 — De-leaked co-essentiality → functional linkage (rehabilitates the Cofit footgun)
**Question:** do genes with correlated conditional-essentiality tend to be functionally linked
(operon / SEED subsystem), predicted from **train-only** co-essentiality, and does it add signal
**beyond** ProteomeLM sequence cosine? **Why it escapes:** fitness is the *feature*, linkage the
*target* — not a held-out fitness cell. Train-only correlation is the de-leaked replacement for the
banned `Cofit` table. **Mandatory corrections:** (a) **drop KEGGMember as the label** — here it only
yields shared k-group = homology, conflating target with the embedding control; use SEED subsystem /
Metacyc. (b) **pair-disjoint holdout** or it's transductive and uninteresting. **Headline must be:**
incremental AUPRC of co-essentiality *over embedding-cosine* on **non-adjacent, low-homology,
same-subsystem** pairs (the base claim "co-essentiality predicts linkage" is known biology, Price
2018). **Threat:** fragile LOPO cross-org transfer (R-AUG negative transfer). **Effort:** 1–2 wk, no GPU.

## Verdict on the organism-level GNN / bipartite gene–chemical idea (3 arms, opposite verdicts)
- **Transductive bipartite GNN (gene_id × chem_id edge scorer) = the SAME wall** — matrix completion
  with identity params; memorizes per-gene/per-chemical offsets. Keep only as the adversarial
  baseline that *demonstrates* the wall.
- **Inductive cold-GENE subgraph GNN (IGMC-style) = same wall in disguise** — the SpecificPhenotype
  bipartite graph is too sparse (**66% of genes degree ≤2, 39% degree exactly 1**); a held-out gene's
  enclosing subgraph has zero internal structure, so it provably reduces to embedding-kNN +
  fingerprint-kNN side-edge lookup. Don't build it.
- **Inductive cold-CHEMICAL arm = the genuine escape, but gated** — a held-out compound has no
  same-compound row to copy, so structural-fingerprint generalization is the only path; fingerprints
  **exist** (`canonical_fingerprints.npz`, ~77% SMILES). **Kill-switch first:** does
  fingerprint-kNN-on-edges already beat a plain bilinear scorer on held-out compounds? Only if
  there's headroom does the GNN earn its machinery. Build the kill-switch, not the full IGMC stack.

## Avoid / demote
- **Transformer with gene + chemical tokens → predict fitness, read attention for gene-chemical
  importance** (evaluated 2026-06-25) — **same wall.** The wall is a property of the *inputs +
  task*, not the architecture: any model whose inputs are `(frozen gene emb, condition chemistry)`
  and whose target is held-out fitness of a *seen* gene is the dead global-parametric family (R1:
  deep ≈ linear; tabular transformers don't beat kNN/trees — Grinsztajn 2022, McElfresh 2023).
  Variants all map to known results: gene+chem tokens→fit = the attractor; whole-matrix
  masked-cell transformer *with gene identity* = transductive matrix completion (memorizes offsets,
  matches kNN at best) → on cold genes falls back to R-COLD; attending over the gene's *own
  (condition, fitness) history* = a learned kNN (TabR/EASE) → "match/slightly-beat" ceiling, not an
  escape. **Attention-as-importance** additionally fails on its own terms: attention ≠ faithful
  explanation (Jain & Wallace 2019); a model at the kNN floor isn't a trustworthy mechanism source;
  and "which chemicals matter for a gene" is already directly the gene's measured `fit` profile, so
  the attention mostly re-derives chemical similarity, not new biology. The useful version of
  "which chemicals interact with which genes" is the **residual miner** (surprising interactions
  the lookup can't explain) or the **cold-chemical** arm — the generalization axis, not the
  attention, is what carries value.
- **Transductive bipartite GNN & cold-gene subgraph GNN** — same wall (above).
- **SpecOG conservation (#6) as specified** — `SpecOG` is positives-only (no denominator; `nInOG`
  counts only hitters), `ogId` is condition-specific (one gene → up to 17 ogIds), and splitting on
  `ogId` **leaks the target**; usable N collapses ~24,789 → ~1,810. Needs a from-scratch rebuild from
  the 2.84M-row `Ortholog` table + tested-negative denominators, and a phylogenetic-breadth-only
  baseline likely wins. Avoid unless someone commits to that.
- **#4 SEED-function & #2 MoA** — not same-wall but interest-capped (#4: sequence already predicts
  function, only the fingerprint-vs-embedding delta is novel and finding-#3 predicts a likely LOPO
  negative; #2: prior art (Price 2018) + re-walls as compound-kNN unless a strict leave-compound-AND-
  leave-org split, which guts N to ~33 compounds).

## Grounded data facts (verified against feba.db / repo — don't re-derive)
- `SpecificPhenotype` = 38,525 rows; ~54,850 orphan protein-coding genes; ~14,577 orphans with a
  cross-org ortholog; replicate σ available for ~72–75% of conditions.
- `chemistry_knn_predict(exclude_self=...)` + replicate floor: `src/ranking/eval/harness.py:337`.
- `materialize_cold_gene` / `materialize_condition_holdout`: `src/data/datasets/build_ranking_split.py:127`.
  **No LOPO (leave-one-compound) materializer exists — must be built.**
- Fingerprints DO exist: `data_contract/chemistry/canonical_fingerprints.npz` (Morgan/RDKit/MACCS) +
  `canonical_id_smiles.json` (~77% SMILES). (Corrects a stale "no SMILES" assumption.)
- Functional labels: SEED via `SEEDAnnotation → SEEDAnnotationToRoles → SEEDRoles`; operons via
  `Gene.scaffoldId/begin/end/strand`; orthology via `Ortholog` (2.84M rows).
- **Leakage (do not use as split-blind features):** `Cofit`, `ConservedCofit`, `SpecOG` (positives-only).

## Recommendation
1. **Spike #1 (dark-genome residual miner) first** — days; the one go/no-go is the replicate-swap
   null. It's the only reframe the wall cannot recapture and it turns the central negative into the
   contribution (a ranked dark-gene nomination list w/ stressing condition + cross-org corroboration).
2. **Then #2 (de-leaked co-essentiality)** if a methods/benchmark paper is wanted, scoped strictly to
   the incremental-AUPRC-over-embeddings headline with a pair-disjoint hold-out.
3. **User GNN:** build only the **cold-chemical kill-switch baseline**; skip the full stack unless it
   shows headroom over fingerprint-kNN.
