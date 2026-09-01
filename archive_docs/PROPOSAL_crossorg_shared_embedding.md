# PROPOSAL — Cross-organism shared gene–condition embedding (conserved-vs-rewired conditional essentiality)

**Date:** 2026-07-06 · **Status:** proposal (implementation-ready pending parquet rebuild on the
restored 62-org DB) · **Lineage:** the buildable form of Direction 1 in
`DIRECTIONS_adversarial_slate_2026-06-29.md`, upgraded with a shared-embedding architecture. Sits
squarely in **H2 / the non-rigged regime** (see `SCIENTIFIC_SYNTHESIS §9`).

---

## 1. The question (and why it escapes the rigged benchmark)

Writing `fit_o(g,c) = a_g^o + b_c^o + I_o(g,c) + noise`, we ask of the **interaction** `I`:
**is a gene's conditional-essentiality response conserved or rewired across organisms, and can we
predict a held-out organism's response to a compound it never tested?** The target is a
*cross-organism* object; a within-organism lookup (chem-kNN) structurally cannot compute it (measured
cross-org transfer of within-gene ranking ≈ 0.045). So the lookup's rigged advantage (its own-history
crutch) does not apply — this tests **H2** (do our features carry generalizable `I` signal), not H1.

## 2. Data grounding (verified against the restored 62-org `feba.db`, 2026-07-06)

- **Shared conditions (the anchor for a common condition space):** ~117 canonical conditions in ≥10
  organisms — 48 stress compounds (26 in ≥20), 39 carbon sources, 30 nitrogen sources.
- **Orthologs (`Ortholog` table, ratio = bitscore/self; use ratio>0.5):** 3.6M directed links / 3,782
  org-pairs; **179,217 genes with ≥1 cross-org ortholog**, **~52k conserved across ≥10 orgs**, ~26k
  across ≥20. Pilot pairs: **DvH↔Miya 1,784** (close), **Caulo↔MR1 164** (distant).
- **Caveat:** the new distant organisms are weakly orthologous to the panel (M. tuberculosis 8% /
  67 orthologs to Keio; Bifido 10%) — they strengthen **Mode A** (chemistry/MoA/transfer), not
  **Mode B** (rewiring). Do the rewiring pilot on the Proteobacteria + same-species pairs.

## 3. The architecture

For organism *o*, gene *g*, condition *c*, latent dim d (start 32–64):

```
fit_o(g,c)  ≈   a_g^o  +  b_c^o  +  ( s_g + δ_g^o ) · v_c
```

| Term | What it is | Source |
|---|---|---|
| `a_g^o` | per-(org,gene) mean — the gene's average importance in that organism | free scalar |
| `b_c^o` | per-(org,condition) mean — that condition's average harshness in that organism | free scalar |
| `s_g = φ(emb_g)` | **conserved sequence prior** — orthologs get ≈ the same `s_g` by construction | shared MLP φ over a **context-free** protein embedding (ESM-C) |
| `δ_g^o` | **organism-specific, fitness-learned deviation** — how the gene departs from its sequence prior | free latent per (org,gene), L2-regularized toward 0 (strength **λ**) |
| `v_c` | **shared condition vector** — a compound's biological "meaning," learned from *all* organisms | free latent for warm/shared conditions, **tied to `ψ(nutrient_profile(c))`** for inductive placement of novel/cold conditions |

- `(s_g + δ_g^o) · v_c` is the predicted **interaction** `I_o(g,c)` — the geometric "closeness = fitness effect."
- **De-meaning is built in** (explicit `a`,`b` biases → the dot product models only `I`).
  **Denoising is built in** (low-rank factorization reconstructs reproducible signal, discards per-cell noise).
- **The λ knob is the conserved↔rewired control.** High λ forces the model to explain fitness with the
  shared `s_g · v_c` structure; the `δ` that *survive* regularization mark genes the data insists have
  departed from their sequence prior = rewiring candidates.

## 4. Design decisions (with rationale)

### 4.1 Protein embedding → **ESM-C (context-free)**; verify in Phase 0
The prior `s_g` must be **conserved for orthologs** so `δ` isolates rewiring. ProteomeLM is
proteome-*contextualized*, so the same protein can embed differently across organisms — muddying the
conserved baseline. ESM-C (context-free) gives near-identical embeddings for near-identical sequences →
orthologs cluster by construction. **Evo2 is worse for this** (DNA-level, orthologs diverge faster).
**Phase-0 empirical gate:** compute ESM-C over `aaseqs`; measure cosine similarity of high-confidence
ortholog pairs (ratio>0.8) for ESM-C vs (if regenerated) ProteomeLM; pick the embedding with high
ortholog similarity that still separates functions. *(Alternative design if no embedding gives clean
ortholog similarity: Variant-2 below — free gene latents with an ortholog-coupling regularizer, so the
embedding choice matters less.)*

**Phase-0 RESULT (2026-07-06): PASS.** `esmc_300m` (960-d, mean-pooled) generated for all 279,140
proteins / 62 orgs. Orthologs cluster decisively: ortholog cosine median **0.92–0.99** vs random **~0.70**,
**AUROC 0.97–0.998** across close (DvH↔Miya) *and* distant (Caulo↔MR1, Keio↔*M. tuberculosis*) pairs, and
cosine is **monotone in sequence identity**. The conserved-sequence-prior assumption holds → proceed with
ESM-C.

> **Modeling-phase note — center/whiten before φ.** Mean-pooled ESM-C is somewhat **anisotropic**: random
> cross-organism pairs sit at cosine ~0.70 (embeddings occupy a narrow cone), so *absolute* cosine is
> compressed even though ortholog-vs-random *separation* is excellent. Before feeding embeddings to `φ`,
> **center (subtract the global mean) and whiten** (e.g. PCA-whiten or per-dim standardize) so the space is
> spread out and distances are meaningful. The learned projection `φ` can absorb an affine transform, so
> this mainly helps optimization/conditioning and any distance-based diagnostics — do it, it's cheap.

### 4.2 Condition features → learned `v_c` + a **nutrient-profile** inductive bridge
Multihot is identity-like and cannot place a **novel** condition (unseen indicator → zero vector),
which is exactly what Mode A / cold-condition need. Build a **nutrient profile**:
```
nutrient_profile(c) = [ Σ_i w(conc_i)·fingerprint(compound_i)  over media components + stressor(s) ]
                        ⊕ [pH, temperature, aerobic, liquid/solid, mediaStrength]
```
(fingerprints: Morgan/RDKit/MACCS, ~77% SMILES coverage; `w` = log-concentration or learned dose; keep
separate "base-media" and "stressor" channels). **Use both:** learn `v_c` freely for warm shared
conditions (captures *biological* mechanism, not just chemistry) and regress `v_c` on
`ψ(nutrient_profile)` so novel/cold conditions are placeable. Keep **multihot as the baseline** ψ to
measure the nutrient-profile's lift. (Fingerprints didn't help the *warm* task because it needs no
generalization; here generalization is the point.)

### 4.3 Gene scope → **train on ALL genes, compare only orthologs**
Train `φ`, `v_c`, `δ`, biases on **every gene in every organism** (fills the space, sharpens `v_c`/`φ`).
Restrict the **Mode-B rewiring comparison** to ortholog pairs (the only place a cross-org comparison is
defined). Accessory (non-orthologous) genes are a separate category — organism-specific essentiality
that *cannot* be conserved-vs-rewired, but still informs the shared space.

## 5. The two outputs

**Mode A — a *fair* cross-organism prediction the lookup can't make.** Organism B never tested compound
X, but A/C/D did → `v_X` is defined. Place B's genes in the shared space → **predict B's response to a
never-measured-in-B compound.** Honest test (B has zero own-history for X): does it beat B's **chem-NULL**
on held-out (org, compound) cells? A publishable prediction result if yes.

**Mode B — the rewiring map (discovery).** For each ortholog pair, compare `δ_g^A·v_c` vs `δ_{g'}^B·v_c`
across shared conditions. Agreement → conserved; disagreement above the replicate-noise floor → **rewired**
(same protein, different conditional role), with the *direction* naming which conditions it moved
toward/away from.

## 6. Phased plan (go/no-go at each gate)

- **Phase 0 — data + embedding (days).** Rebuild the parquet on the 62-org DB; canonicalize conditions
  across organisms; compute ESM-C; **gate:** orthologs must (a) align as canonical conditions and
  (b) show high ESM-C cosine similarity. If conditions won't canonicalize or orthologs aren't similar,
  stop / switch to Variant 2.
- **Phase 1 — joint model.** Fit the shared model on all organisms. **Gate:** does shared `v_c`+`s_g`
  reconstruct held-out cells **better than per-organism** linear-MF? If sharing doesn't help
  reconstruction, the geometry isn't transferring.
- **Phase 2a — Mode A.** Leave-(org,compound) transfer vs chem-NULL. **Gate:** beats chem-NULL?
- **Phase 2b — Mode B.** Rewiring readout on DvH↔Miya (close) + Caulo↔MR1 (distant), gated on
  replicate-half reproducibility of `δ` + the noise floor. **Gate:** is the sequence-conserved /
  phenotype-rewired population non-empty above noise?
- **Phase 3 — validation.** Enrich rewired hits on an **independent** axis (regulatory genes,
  mobile-element proximity, operon/genomic-context differences) — non-circular.

## 7. Risks & mitigations

1. **Cross-org transfer ≈0 (measured).** Mode A may not beat chem-NULL. Bet: orthology-aware `φ` +
   shared conditions share the *right* thing (unlike naive pooling → R-AUG negative transfer). Phase-1
   reconstruction gate tests it cheaply before investing.
2. **`δ` is transductive** → Mode B is **warm-gene only** (fine for characterization; can't rewire-analyze
   unseen genes).
3. **Condition canonicalization is messy** (free-text `condition_1`, dose differences). Conservative
   canonicalization + dose-aware features; this is the Phase-0 make-or-break.
4. **"Rewired" could be a batch/technical artifact** between organisms → the replicate-half +
   ortholog-concordance + independent-axis gates kill those.
5. **Factorization identifiability** (rotation/scale gauge) → freeze `φ` or add orthogonality
   constraints so `δ` is comparable across organisms.
6. **Leakage discipline:** orthology from the **raw `Ortholog` table only** (never `ConservedCofit`);
   any co-fitness recomputed train-only; all cross-org neighbors from train.

## 8. Engineering checklist

- [ ] Parquet rebuild on 62-org DB (blocked on `media_composition.xlsx` or a media-less workaround).
- [ ] Canonical condition IDs across organisms (compound ± dose bucket) from `condition_1..4` + `Compounds`.
- [ ] ESM-C embeddings from `data/raw/aaseqs` (GPU; ~288k proteins; hours).
- [ ] Ortholog groups from raw `Ortholog` (ratio>0.5).
- [ ] Nutrient-profile featurizer (fingerprint mixture + scalars); multihot baseline retained.
- [ ] Model module: shared φ, shared `v_c` (free + `ψ` bridge), per-org `δ`/`a`/`b` — extend
  `inductive_mf_predict` / reuse `RankingBatch` + eval harness.
- [ ] Eval: reconstruction vs per-org MF; Mode-A leave-(org,compound) vs chem-NULL; Mode-B rewiring
  readout + replicate-half reproducibility + noise gate + ortholog concordance.

## 9. What "good" looks like

- **Mode A good:** shared model predicts held-out (org, compound) fitness above chem-NULL — a fair,
  novel cross-organism prediction the lookup cannot make.
- **Mode B good:** a replicate-gated, ortholog-concordant population of sequence-conserved /
  phenotype-rewired genes, enriched (independently) for regulators / mobile-element-adjacent / operon-
  reorganized genes — with the shared embedding providing denoising + ragged-overlap robustness +
  an interpretable "which conditions moved" readout that raw profile-correlation cannot.
- **Either bad → still informative:** a clean null on rewiring (genotype→conditional-phenotype is rigid)
  or a failed transfer (features don't carry cross-org `I`) is a publishable characterization result.
