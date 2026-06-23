## Decision: R-AUG-DEC-001

### Header
- decision_id: R-AUG-DEC-001
- stage_or_tier: R-AUG
- regime: R
- date: 2026-06-18
- owner: project lead
- status: approved
- approved_date: 2026-06-18
- related_experiments: [R-AUG_train_org_augmentation]
- related_hypotheses: [H-R-AUG-01]

### Assumption Under Test
- assumption_statement: H-R-AUG-01 — the headline trains the global model on only
  the 23 replicate organisms, leaving ~25 feba organisms (incl. some of the
  largest: Putida, psRCH2, the pseudo* set, WCS417, Phaeo, Pedo557 …) out of
  training entirely. Training the model on ALL 48 embedded organisms — while
  evaluating on the same locked 23 against the same chem-kNN gate — narrows the
  model→gate gap, because more organisms give the shared encoder more
  (protein-embedding, chemistry)→fitness examples to generalize from.
- assumption_type: data
- why_it_matters: This is the one lever that helps ONLY the global model —
  chem-kNN is per-organism-local and gains nothing from extra training
  organisms, so its gate number is fixed. It is the cleanest, cheapest test of
  whether the model→gate gap is a training-DATA-VOLUME problem (fixable with more
  data, incl. the external Tn-seq datasets surveyed 2026-06-18) or a STRUCTURAL
  one (no transferable cross-organism signal). A null/negative result closes the
  "just add more organisms" line of attack for good.

### Pre-Registered Test Plan
- comparison: locked arm (pointwise_huber + multihot_425 + T5-A) trained twice on
  the IDENTICAL locked evaluation —
    - base_23org : model trained on the 23 eval organisms only
    - aug_48org  : model trained on all 48 embedded organisms (23 eval ∪ 25 extra)
  Both scored on the identical eligible val genes against the identical chem-kNN
  gate. 3 model seeds, split seed 0, 8 epochs.
- fixed_controls: val set, R-LOCK-1 eligibility, and ALL baselines (chem-kNN gate,
  chem-NULL, linear-MF) are produced exactly as `prepare_r1_data(23 orgs)` — the
  extra organisms contribute ENTIRELY to training and never touch the evaluation.
  Implemented via `R1Data.baseline_train` (baselines score on the locked eval-org
  train) + `prepare_r_aug_data` (grafts extra-org rows onto `.train` only,
  extends emb/multihot to the union). The chem-kNN gate is asserted bit-identical
  between the two arms (`drift = 0.000000`), which proves the eval is held fixed.
- metrics_primary: [ndcg_at_5, within_gene_spearman]
- promotion_rule: aug must IMPROVE the model toward the gate (ΔNDCG@5 > 0 vs base,
  consistent across seeds) to motivate ingesting more (incl. external) organisms.

### Evidence Summary

**Model arm (seed-mean, 3 seeds, 23 eval orgs, identical eligible val gene set):**

| arm | Spearman | NDCG@5 | prec@5 | vs gate (NDCG@5) |
|---|---|---|---|---|
| base_23org (train 23) | 0.1522 | 0.4319 | 0.4014 | −0.0533 |
| **aug_48org (train 48)** | **0.1270** | **0.4166** | **0.3870** | **−0.0686** |
| Δ (aug − base) | **−0.0252** | **−0.0152** | −0.0144 | gap widened |
| **chem-kNN (GATE)** | 0.2402 | **0.4852** | 0.446 | — (drift 0.000000) |

**Per-seed (disjoint groups — the effect is robust, not seed noise):**
- base NDCG@5: [0.4333, 0.4302, 0.4321] · aug NDCG@5: [0.4173, 0.4133, 0.4193].
  Every aug seed sits below every base seed.

**Findings:**
1. **Training on more organisms did NOT help — it actively HURT.** aug_48org is
   −0.0152 NDCG@5 and −0.0252 Spearman below base_23org, consistently across all
   3 seeds. H-R-AUG-01 **rejected** (and then some — the sign is negative).
2. **This is negative transfer.** The shared model weights, forced to fit 25
   additional heterogeneous organisms whose conditional structure does not
   transfer (the T-regime already measured cross-org transfer ≈ random), are
   pulled away from the eval-organisms' specific structure. The extra data is the
   wrong kind: distribution-shift / noise relative to the target organisms.
3. **The model→gate gap is NOT a training-data-volume problem.** Doubling the
   training organisms (≈ doubling the rows) widened the gap rather than closing
   it. The bottleneck is structural: the absence of transferable cross-organism
   signal, exactly as R1 / R-LOSS / R-HYBRID predicted.
4. **Gate parity verified.** chem-kNN NDCG@5 = 0.4852 in BOTH arms (drift
   0.000000) — the comparison is a clean controlled A/B; only the model's
   training data changed.

### Decision
- decision_outcome: **reject H-R-AUG-01.** Do NOT augment training with additional
  organisms for the within-organism headline objective. Keep the 23-org training
  set as the locked configuration.
- rationale:
  1. More organisms produce negative transfer here, so the move is strictly
     counterproductive for the warm-gene within-org metric.
  2. It empirically backs the 2026-06-18 external-Tn-seq survey's "skip" verdict:
     adding even MORE DISTANT organisms (MtbTnDB, A. baumannii — different clades,
     different assay methods) would almost certainly worsen negative transfer, not
     help. The external-data integration cost is not justified for this objective.
  3. Combined with R-LOSS / R-TOPK (objective doesn't matter) and R1 (encoder/
     capacity don't matter), the training-data axis is now also closed: the
     headline is memorization-dominated and local, and no global-model lever
     tried has moved it.
- scope / what this does NOT close: R-AUG measures the warm-gene, within-org
  headline — chem-kNN's home turf. The **cold-gene** regime (genes with no
  within-org history for kNN to retrieve) is untouched and remains the one place a
  global model could win; more-diverse training could even help THERE while
  hurting the warm headline. The cold-gene diagnostic (`materialize_cold_gene`
  already exists) is the designated next experiment.
- reproduce: `python -m src.cli.run_experiment +experiment=R-AUG_train_org_augmentation`
  (artifacts in `artifacts/runs/raug/`).
