# terms.md — glossary

Search this before naming anything. Every metric/term is defined once, with
provenance and status. If a name is not here, it is not an approved name.

**Status values:** `PRIMARY` | `SECONDARY` | `DIAGNOSTIC` | `DEMOTED` | `RETIRED`

---

## Metrics

### ndcg_at_5  [metric] [retrieval]
```
Is:                   Normalized discounted cumulative gain at k=5 for one gene's ranked
                      conditions, averaged over eligible val genes. Relevance of condition c
                      for gene g is max(0, -fit_gc); the model ranks conditions by predicted
                      fit ascending. Measures top-of-list agreement = "find the top stressors".
Computed by:          src/ranking/eval/harness.py:ndcg_at_k, aggregated by within_gene_retrieval
CHANGES MEANING WITH: the eligible val gene set (denominator parity); the split
                      (condition_holdout vs cold_gene); k; MEAN vs median aggregation
                      (this project always uses MEAN)
Valid vs:             any method scored on the SAME eligible val gene set, same split, same k
Status:               PRIMARY
Aliases:              "NDCG@5", "the headline metric". Do not write "ndcg5".
```

### within_gene_spearman_mean  [metric] [full-list]
```
Is:                   Per-gene Spearman correlation between predicted and observed fit across
                      that gene's val conditions, averaged over eligible val genes. The
                      completeness / anti-gaming check on the full ranked list.
Computed by:          src/ranking/eval/harness.py:per_gene_correlations (metric="spearman")
CHANGES MEANING WITH: eligible val gene set; split; min_n conditions per gene (default 5);
                      MEAN vs median aggregation
Valid vs:             any method on the same eligible val gene set. NEVER quoted as the
                      headline — it never outranks ndcg_at_5.
Status:               SECONDARY
Aliases:              "within-gene Spearman", "Spearman". Never just "correlation".
```

### within_gene_kendall_mean  [metric] [full-list]
```
Is:                   As within_gene_spearman_mean but Kendall tau. Carried as the
                      no-regression guard: a promoted model must not degrade it.
Computed by:          src/ranking/eval/harness.py:per_gene_correlations (metric="kendall")
CHANGES MEANING WITH: same axes as within_gene_spearman_mean
Valid vs:             same eligible val gene set
Status:               SECONDARY
Aliases:              none
```

### precision_at_5  [metric] [retrieval] [diagnostic]
```
Is:                   Fraction of a gene's predicted top-5 conditions that are truly stressors.
                      Reported alongside ndcg_at_5; not gated on.
Computed by:          src/ranking/eval/harness.py:precision_at_k
CHANGES MEANING WITH: eligible val gene set; k; the stressor threshold implicit in
                      relevance = max(0, -fit). MEDIAN aggregation collapses binary
                      precision@1 to zero — this project uses MEAN.
Valid vs:             same eligible val gene set
Status:               DIAGNOSTIC
Aliases:              "prec@5"
```

### n_genes  [diagnostic]
```
Is:                   The number of eligible val genes a metric was computed over — the
                      denominator. Identical across methods in a report is the parity
                      guarantee; a differing value means the numbers are not comparable.
Computed by:          src/ranking/eval/harness.py:_metrics_for_pred (per method)
CHANGES MEANING WITH: the split; the eligibility filter; whether parity was enforced
                      (the common gene set) or each method scored on what it could cover
Valid vs:             the other methods in the same report — it must match
Status:               DIAGNOSTIC
Aliases:              "n", "the denominator"
```

---

## Reference points (ceiling and floor)

### replicate_ceiling_ndcg_at_5  [ceiling]
```
Is:                   The best ndcg_at_5 any predictor could achieve on this target: replicate
                      A's fit used as the PREDICTION, replicate B's fit as the TRUTH, for genes
                      with >=2 replicate expNames at >=5 distinct conditions. MEAN over genes.
Computed by:          src/ranking/eval/harness.py:retrieval_noise_floor
CHANGES MEANING WITH: the eligible val gene set (parity-eligible vs all val genes);
                      min_conditions (default 5); org subset (23 replicate orgs vs subsets);
                      MEAN vs median; AND the target definition — this is the ceiling of the
                      SINGLE-NOISY-MEASUREMENT task, and denoising the target RAISES it
Valid vs:             methods scored on the same eligible val gene set. Quote as "fraction of
                      ceiling" only when numerator and denominator share that gene set.
Status:               PRIMARY
Aliases:              "noise floor", "the ceiling", "replicate oracle". "Noise floor" is a
                      misnomer for a ceiling — prefer "replicate ceiling".
```

### replicate_ceiling_spearman  [ceiling]
```
Is:                   As above, for within_gene_spearman_mean: per-gene cross-condition Spearman
                      of replicate A vs replicate B, MEAN over PARITY-ELIGIBLE val genes.
Computed by:          src/ranking/eval/contract.py:task_relevant_noise_floor
                      -> use the "per_gene_mean" return field, with eligible_genes supplied
CHANGES MEANING WITH: **aggregator (mean vs median) and eligibility filter — BOTH.** The
                      unfiltered median variant is a materially different number and is
                      registered separately below. Also org subset and min_conditions.
                      CAVEAT for external claims: eligibility selects high-spread genes, which
                      replicate better, so this ceiling is conditional on the eligibility
                      filter. It is NOT a statement about the assay in general.
Valid vs:             methods scored on the same parity-eligible val gene set with MEAN
                      aggregation — i.e. every number in the standard report
Status:               PRIMARY
Aliases:              "Spearman ceiling", "task-relevant noise floor"
```

### replicate_ceiling_spearman_unfiltered_median  [ceiling] [diagnostic]
```
Is:                   The same per-gene replicate Spearman distribution, but MEDIAN-aggregated
                      over ALL val genes with no eligibility filter.
Computed by:          src/ranking/eval/contract.py:task_relevant_noise_floor
                      -> the "primary_value" return field, eligible_genes=None
CHANGES MEANING WITH: it IS the change — median instead of mean, unfiltered instead of
                      parity-eligible. Both axes move it in the same direction.
Valid vs:             NOTHING in the standard report. It shares no gene set and no aggregator
                      with the model or the baselines. Quoting it against a model score is an
                      apples-to-oranges comparison in two ways at once.
Status:               DIAGNOSTIC
Aliases:              "0.3214", "the median ceiling". Historically mislabeled "USE THIS" in
                      data_contract/ranking/metric_contract.yaml — see the correction in
                      docs/memory.md dated 2026-08-25.
```

### chem_null  [baseline] [floor]
```
Is:                   The population condition profile: predict gene g's fit at held-out
                      condition c from the TRAIN-GENE MEAN fit at c's chemically nearest train
                      condition. Gene-identity-free — it captures the condition main effect
                      b_c and nothing gene-specific. The floor.
Computed by:          src/ranking/eval/harness.py (chemistry nearest-condition profile)
CHANGES MEANING WITH: chemistry encoder (multihot vs fingerprints); the nearest-condition
                      metric (Jaccard vs Tanimoto); eligible val gene set; split
Valid vs:             any method on the same eligible val gene set. It is THE GATE on the
                      cold_gene split, where chem_knn is inapplicable.
Status:               PRIMARY (as the cold_gene gate) / SECONDARY (as the warm floor)
Aliases:              "chem-NULL", "population profile", "the null"
```

### chem_knn  [baseline] [reference]
```
Is:                   Predict gene g's fit at held-out condition c from g's OWN train fit at
                      its k chemically-nearest train conditions. Uses the target gene's own
                      measured history; a lossless non-parametric reader of that history.
Computed by:          src/ranking/eval/harness.py:chemistry_knn_predict (exclude_self param)
CHANGES MEANING WITH: k; chemistry encoder; exclude_self; **the split — on cold_gene it has
                      no own-gene history and coverage goes to zero, making it inapplicable
                      rather than merely worse**; eligible val gene set
Valid vs:             any method on the same eligible val gene set, on the condition_holdout
                      split only. On cold_gene it is undefined, not zero.
Status:               PRIMARY (the warm gate) — but see the DEMOTED note below
Aliases:              "chem-kNN", "the gate", "the lookup"
Note:                 As of 2026-07-06 this is understood as a MEMORIZATION REFERENCE (what a
                      lookup handed the gene's own answer key achieves), not the scientific
                      bar. It remains the operational promotion gate for the warm split.
```

### linear_mf  [baseline]
```
Is:                   Linear inductive matrix factorization: a free per-gene latent U[g] times
                      a chemistry-feature condition map. A LEARNED, FITNESS-AWARE gene
                      representation — the control that shows the bottleneck is not the
                      embedding's fitness-blindness.
Computed by:          src/ranking/eval/harness.py (cached per split)
CHANGES MEANING WITH: rank; regularization; **the split — needs a learned per-gene latent, so
                      it is unlearnable on cold_gene**; eligible val gene set
Valid vs:             same eligible val gene set, condition_holdout split only
Status:               SECONDARY
Aliases:              "linear-MF", "inductive MF". NOT "matrix factorization" — vanilla MF is
                      RETIRED for the primary split (see below).
```

---

## The model and the task

### model  [system]
```
Is:                   AdapterResidualMLP — an adapter over the frozen gene embedding
                      concatenated with condition chemistry features, trained pointwise.
Computed by:          src/ranking/models.py:AdapterResidualMLP; trained via
                      src/ranking/train.py; locked loss pointwise_huber
CHANGES MEANING WITH: the embedding source (ProteomeLM-L8 RETIRED -> ESM-C current — every
                      historical model number was produced with ProteomeLM-L8); chemistry
                      encoder; loss; model_seeds; training org set
Valid vs:             baselines on the same eligible val gene set and split
Status:               PRIMARY
Aliases:              "the deep model", "R1 model", "the global model"
```

### interaction_target  [concept]
```
Is:                   The decomposition fit(g,c) = mu + a_g + b_c + I(g,c) + noise.
                      a_g = gene main effect, b_c = condition main effect, I(g,c) = the
                      gene-by-condition interaction. Within-gene ranking fixes g (removing
                      a_g) and eligibility selects genes where b_c is small, so the task IS
                      predicting I(g,c). This target was chosen by the metric, not discovered.
Computed by:          not computed — the framing that governs every metric here.
                      See docs/memory.md 2026-07-06.
CHANGES MEANING WITH: whether the target is de-meaned (a_g, b_c explicitly subtracted) and
                      whether it is denoised (replicate-averaged / t-shrunk). Neither is
                      implemented yet; both change what "the target" means.
Valid vs:             n/a (a framing, not a number)
Status:               PRIMARY
Aliases:              "the interaction", "I(g,c)"
```

### promotion_delta  [gate]
```
Is:                   The margin a learned model must clear over chem_knn to be promoted.
                      Defined as 0.15 x (replicate_ceiling - chem_knn) on the PRIMARY metric,
                      evaluated on the parity-eligible val gene set, plus disjoint
                      hierarchical-bootstrap CIs. On ndcg_at_5 this evaluates to ~0.026.
Computed by:          data_contract/ranking/metric_contract.yaml (promotion block)
CHANGES MEANING WITH: **which ceiling is used.** Against replicate_ceiling_ndcg_at_5 the
                      formula gives the documented ndcg_at_5 gate; against the DIAGNOSTIC
                      unfiltered-median Spearman ceiling it gives a different, superseded
                      Spearman value. Also: the metric it is applied to.
Valid vs:             a model-minus-chem_knn difference on the same eligible val gene set
Status:               PRIMARY
Aliases:              "the gate", "the promotion bar"
```

### eligibility  [protocol]
```
Is:                   The rule selecting which val genes are scorable. A gene is eligible if
                      its spread tail_g exceeds a per-organism threshold. Ranking a flat gene
                      is meaningless, so flat genes are excluded from val and down-weighted in
                      train via w_g.
Computed by:          src/data/datasets/ranking_eligibility.py;
                      policy in data_contract/ranking/eligibility_policy.yaml
CHANGES MEANING WITH: the per-org threshold; the spread metric (tail_g vs IQR); whether it is
                      applied to val only or to train weighting too
Valid vs:             n/a (a protocol). Every reported number inherits it.
Status:               PRIMARY
Aliases:              "the eligible val gene set", "parity-eligible", "R-LOCK-1"
```

### tail_g  [statistic]
```
Is:                   Per-gene spread of fit across conditions, p95 minus p5. The eligibility
                      spread metric. Chosen over IQR because a gene essential in a handful of
                      conditions has IQR near zero but a large tail — IQR would exclude exactly
                      the most rankable genes.
Computed by:          src/data/datasets/ranking_eligibility.py
CHANGES MEANING WITH: the percentile pair (p95/p5); whether computed on train rows only
Valid vs:             other genes within the same organism (thresholds are per-org)
Status:               PRIMARY
Aliases:              "spread", "the tail". NOT "IQR" — see Retired.
```

### w_g  [statistic]
```
Is:                   Per-gene train weight proportional to discriminability, so all train
                      genes contribute but flat genes contribute little. Anchors the variance
                      scale and teaches a "predict near-flat for non-conditional genes" prior.
Computed by:          src/data/datasets/ranking_eligibility.py
CHANGES MEANING WITH: the weighting policy (weighted_all vs hard-filtered train)
Valid vs:             n/a (a training weight)
Status:               PRIMARY
Aliases:              "train weight"
```

### pointwise_huber  [loss]
```
Is:                   The locked training objective: Huber loss on predicted vs observed fit,
                      applied per (gene, condition) row and weighted by w_g. Robust to fit
                      outliers. Marginally the best of every loss tested; no ranking loss beat it.
Computed by:          src/ranking/losses/ (pointwise family); dispatched by
                      src/ranking/train.py with a ROW-batched sampler
CHANGES MEANING WITH: the batching mode — pointwise losses must be row-batched; a
                      gene-batched sampler changes what the loss sees. Also the Huber delta
                      and whether w_g weighting is applied.
Valid vs:             other losses trained under the same split, seeds and epoch budget
Status:               PRIMARY
Aliases:              "huber", "the locked loss"
```

### split_seed  [protocol]
```
Is:                   The RNG seed that decides WHICH conditions (or genes, or compounds)
                      are held out. It selects the partition of the data.
Computed by:          passed to the materializer in
                      src/data/datasets/build_ranking_split.py; recorded in split_hash
CHANGES MEANING WITH: nothing -- but note what it does NOT do. Varying model_seeds does not
                      vary this. Every headline number in this project uses split_seed=0
                      only, so all reported spread is initialisation noise on a SINGLE
                      partition, not split variance.
Valid vs:             results at the same split_seed. Comparing across split_seeds measures
                      a different thing (partition sensitivity) and must be labelled as such.
Status:               PRIMARY
Aliases:              "the split seed". Never conflate with model_seeds.
```

### model_seeds  [protocol]
```
Is:                   The RNG seeds for network initialisation and batch shuffling. Each
                      produces one trained model on the SAME partition; results are averaged
                      and the spread across them is reported.
Computed by:          src/ranking/runner.py:run_arm, one training run per seed
CHANGES MEANING WITH: the number of seeds; whether the spread is reported as a range across
                      seeds or as a bootstrap CI. **A "disjoint across all 3 seeds" statement
                      is a claim about initialisation stability, NOT about sampling
                      uncertainty over the data.**
Valid vs:             other arms trained on the same partition with the same seed set
Status:               PRIMARY
Aliases:              "seeds", "model seed". Never conflate with split_seed.
```

### random  [baseline] [floor]
```
Is:                   Per-gene random scoring of conditions. THE chance floor for every
                      retrieval metric. Measured `ndcg_at_5` ~0.161 (Keio, indicative).
Computed by:          permute the prediction within each gene, average over permutations
CHANGES MEANING WITH: the number of conditions per gene (NDCG's floor rises as the list
                      shortens); the eligible gene set; the relevance definition
Valid vs:             every other method on the same gene set. Read every reported value as
                      a fraction of the distance from this floor to 1.0, not of 1.0.
Status:               PRIMARY (as the floor)
Aliases:              "chance", "the random baseline". NOT chem_null -- that is a learned
                      population profile and sits well above chance.
```

### constant  [baseline] [diagnostic]
```
Is:                   A predictor returning the same value for every condition, so every
                      within-gene ranking is a tie. Measures what the metric awards for no
                      information at all under this tie-handling. ~0.157 (Keio, indicative).
Computed by:          a zeros vector as the prediction
CHANGES MEANING WITH: tie_handling (average_ranks here); the metric
Valid vs:             `random` -- the two should agree closely; a gap between them indicates
                      the metric is sensitive to tie structure rather than to information
Status:               DIAGNOSTIC
Aliases:              "the constant predictor"
```

### a_g  [statistic]
```
Is:                   The gene main effect: how detrimental knocking out gene g is ON AVERAGE
                      across conditions. This is the NON-CONDITIONAL essentiality quantity,
                      measured within the same data. 40.6% of var(fit) on Keio+Caulo+MR1.
Computed by:          src/ranking/targets.py:fit_additive_effects (sum-to-zero constrained)
CHANGES MEANING WITH: the condition set it is averaged over; whether it is fit on train rows
                      only; the sum-to-zero constraint (without which it is not identified)
Valid vs:             other genes in the same organism and the same fit
Status:               PRIMARY
Aliases:              "gene main effect", "average essentiality". NOT "essentiality" without
                      qualification -- RB-TnSeq cannot assay genes that are unconditionally
                      required, so this is average fitness cost among ASSAYABLE genes.
```

### b_c  [statistic]
```
Is:                   The condition main effect: how harsh condition c is on average across
                      genes. Only 1.7% of var(fit) on Keio+Caulo+MR1 -- nearly negligible.
Computed by:          src/ranking/targets.py:fit_additive_effects
CHANGES MEANING WITH: the gene set averaged over; train-only fitting; **and it is NOT
                      ESTIMABLE for a held-out condition on a cold-column split**
Valid vs:             other conditions in the same organism and the same fit
Status:               PRIMARY
Aliases:              "condition main effect", "condition harshness"
```

---

## Splits

### condition_holdout  [split]
```
Is:                   The primary split. Within each organism, hold out a fraction of
                      CONDITIONS. Held-out conditions are 100% disjoint from train (cold
                      columns), so this is inductive/cold-start matrix completion with
                      chemistry as condition side-features — NOT warm collaborative filtering.
Computed by:          src/data/datasets/build_ranking_split.py:materialize_condition_holdout
CHANGES MEANING WITH: holdout fraction; split_seed; min_holdout for small orgs
Valid vs:             n/a. Numbers from this split are never comparable to cold_gene numbers.
Status:               PRIMARY
Aliases:              "the warm split", "the primary split". "Warm" refers to genes being
                      warm, not conditions — conditions are cold here. Ambiguous; prefer
                      "condition_holdout".
```

### cold_gene  [split] [diagnostic]
```
Is:                   Hold out WHOLE genes per organism — val genes have zero train rows. This
                      makes chem_knn structurally inapplicable (coverage zero) and linear_mf
                      unlearnable, isolating the learned/global component against chem_null.
Computed by:          src/data/datasets/build_ranking_split.py:materialize_cold_gene;
                      src/ranking/pipeline.py:prepare_cold_gene_data
CHANGES MEANING WITH: holdout fraction; split_seed; which baselines are declared applicable
                      (parity_pred_cols restricts the denominator to model + chem_null)
Valid vs:             chem_null only. Never against the condition_holdout gate — chem_null is
                      a weaker bar, so a win here is not a promotion.
Status:               PRIMARY (the live objective)
Aliases:              "cold-gene", "inductive over genes", "R-COLD"
```

### leave_compound_out  [split] [planned]
```
Is:                   PLANNED, NOT BUILT. Hold out whole compounds / chemical scaffolds so no
                      held-out condition has a chemically near measured neighbor. Intended to
                      neutralize the near-neighbor gift that makes chem_knn win.
Computed by:          nothing yet — no materializer exists in
                      src/data/datasets/build_ranking_split.py
CHANGES MEANING WITH: scaffold definition (exact compound vs chemical class); whether it is
                      combined with cold_gene or organism holdout
Valid vs:             chem_null, once built
Status:               DIAGNOSTIC (planned; see docs/plan.md P-EVL-leave-compound-out)
Aliases:              "LOPO", "leave-compound-out", "scaffold split"
```

### coverage  [diagnostic]
```
Is:                   Fraction of eligible val genes for which a given method produces a
                      non-NaN prediction. Exists because a baseline can be INAPPLICABLE on a
                      split rather than merely bad; coverage zero means undefined, not zero
                      skill.
Computed by:          src/ranking/eval/harness.py:_metrics_for_pred
CHANGES MEANING WITH: the split; the eligible val gene set
Valid vs:             other methods on the same split
Status:               DIAGNOSTIC
Aliases:              none
```

---

## Statistics

### hierarchical_bootstrap_ci  [statistic]
```
Is:                   95% CI from resampling organisms with replacement, then genes within each
                      chosen organism. Genes within an organism share conditions, batch effects
                      and noise floor, so a flat gene-level bootstrap is overconfident and
                      inflates "disjoint CI" promotions.
Computed by:          src/ranking/eval/harness.py:hierarchical_bootstrap_ci
CHANGES MEANING WITH: n_bootstrap; ci_level; **the metric — ndcg_at_5 has higher per-gene
                      variance than Spearman, so it yields a WIDER CI; a Spearman-only
                      confidence read overstates significance**; multi-seed handling (a mean
                      of per-seed CIs is a summary band, NOT a pooled bootstrap)
Valid vs:             another method's CI on the same split and gene set
Status:               PRIMARY
Aliases:              "hierarchical CI", "org->gene bootstrap". Not "the bootstrap" — the flat
                      variant in eval/contract.py is a different method.
```

### abs_t  [statistic]
```
Is:                   Absolute value of the Wetmore moderated t statistic for a fitness cell;
                      the per-measurement confidence used to stratify genes.
Computed by:          source column in the canonical fitness parquet; used by
                      src/experiments/rconf/
CHANGES MEANING WITH: whether stratification is by per-gene quartile or per-cell threshold
Valid vs:             other cells in the same organism
Status:               DIAGNOSTIC
Aliases:              "|t|", "confidence"
```

---

## Representations

### esmc_300m  [embedding] [diagnostic]
```
Is:                   Mean-pooled ESM-C 300M protein embeddings, 960-d, fp16, for all proteins
                      in the 62-organism release. Generated 2026-07-06 for the cross-organism
                      shared-embedding proposal. NOT usable as ProteomeLM input (wrong width).
Computed by:          generated 2026-07-06 -> data/processed/ESMC_embeddings/<org>_esmc.pt
CHANGES MEANING WITH: pooling (mean vs CLS); **whitening — mean-pooled ESM-C is anisotropic
                      (random pairs sit at high cosine), so center/whiten before any
                      similarity computation**; fp16 vs fp32
Valid vs:             other embeddings only under a re-run of the full pipeline
Status:               DIAGNOSTIC (the shared-embedding proposal's input)
Aliases:              "ESM-C 300M". Never just "ESM-C" -- the variant is load-bearing.
```

### esmc_600m  [embedding]
```
Is:                   Mean-pooled ESM-C 600M protein embeddings, 1152-d, for all 279,140
                      proteins across the 62-organism release. The input ProteomeLM-L requires.
Computed by:          src/data/encode_esmc.py --model esmc_600m
                      --output-dir data/processed/ESMC_embeddings_600m
CHANGES MEANING WITH: pooling (mean vs CLS); dtype; **and it must not be confused with
                      esmc_300m -- the directory name `ESMC_embeddings` does NOT record which
                      variant it holds, which is exactly how the two got mixed up.**
Valid vs:             as the input to proteomelm_l8; not directly comparable to esmc_300m
Status:               PRIMARY
Aliases:              "ESM-C 600M"
```

### proteomelm_l8  [embedding]
```
Is:                   Frozen ProteomeLM-L layer-8 gene embeddings, 1152-d. The embedding
                      behind EVERY historical result in docs/memory.md. ProteomeLM-L is a
                      proteome-context model applied ON TOP of per-protein ESM-C 600M
                      embeddings -- it is a deterministic function of (ESM-C 600M
                      embeddings, frozen checkpoint), not an independently trained artifact.
Computed by:          src/data/encode_proteomelm_layers.py --keep-layers 8
                      --esmc-dir data/processed/ESMC_embeddings_600m
                      -> data/processed/PLM_embeddings_layer8/{org}_proteomelm.pt
                      Checkpoint: Bitbol-Lab/ProteomeLM-L, snapshot 0f834036...
CHANGES MEANING WITH: **the ESM-C variant feeding it** -- ProteomeLM-L requires 1152-d input,
                      i.e. ESM-C 600M. ESM-C 300M (960-d) does NOT fit and silently is not
                      an option (it raises a shape error). Also: the organism's PROTEOME
                      COMPOSITION, because the model is proteome-contextual -- the same gene
                      in a differently-composed proteome does not get the same vector.
Valid vs:             other embeddings only under a full pipeline re-run. Regenerated values
                      are on the 62-organism release, so they are NOT guaranteed identical to
                      the lost 48-organism ones even for shared genes.
Status:               PRIMARY
Aliases:              "ProteomeLM", "ProtLM-L8", "the frozen embedding"
```

---

## Retired

Kept with reasons so they are not reintroduced.

- **`cross_gene_within_condition_replicate_spearman`** — cross-GENE agreement within one
  condition (one Spearman per replicate pair), from `src/experiments/r0/analyses.py:
  replicate_noise_floor`. Was the original ceiling. RETIRED as a ceiling: it measures a
  different quantity than the within-gene task and reads much higher, overstating headroom.
  Survives as an R0 data-characterization diagnostic only.
- **`matrix_factorization`** (vanilla CF) — invalid on `condition_holdout`: cold columns have
  zero observed entries, so it cannot place them. Applies to a cell-holdout diagnostic only.
  Use `linear_mf` (inductive, chemistry-featured) instead.
- **`per_condition_train_mean`** — undefined on `condition_holdout` (no train mean exists for
  an unseen condition; returned NaN for all val genes on real data). Valid only where
  conditions are warm.
- **`IQR`** as the eligibility spread metric — excludes sparsely-conditional genes, exactly the
  most rankable ones. Superseded by `tail_g`.
- **`Cofit`, `ConservedCofit`, `SpecOG`** — leakage. `Cofit` is derived from the same fitness
  matrix being predicted (cofit-only scores above the replicate ceiling); `SpecOG` is
  positives-only with no denominator and its `ogId` split leaks the target. Never use as
  split-blind features.
- **`ndcg5`, `NDCG-5`, "the correlation"** — non-approved spellings. Use the identifiers above.

---

## ID bridge table

Historical IDs are frozen and referenced by `docs/memory.md`. New work uses
`<phase>-<KIND>-<slug>` with `KIND` in `DAT TRN EVL ANL FIX LCK`, phase letter `P`.
`GEN` is deliberately absent — there is no generative work in this project.

| historical ID | what it was | new-scheme equivalent (for reference only) |
|---|---|---|
| `S0`-`S5` | stage regime: smoke, data characterization, baselines, split lock, feature contract, quality policy | `S-DAT-*`, `S-LCK-*` |
| `T1`-`T6` | tier regime: cross-organism regression | `T-TRN-*`, `T-EVL-*` |
| `R0` | ranking data characterization | `R-DAT-characterization` |
| `R1` | encoder sweep | `R-TRN-chemistry-encoder` |
| `R-LOCK-1`..`4` | eligibility, split, quality, metric contract | `R-LCK-*` |
| `R-LOSS`, `R-TOPK` | loss family, top-k truncated losses | `R-TRN-loss-family` |
| `R-CONF` | confidence stratification | `R-ANL-confidence-strat` |
| `R-COLD` | cold-gene diagnostic | `R-EVL-cold-gene` |
| `R-AUG` | training-organism augmentation | `R-TRN-org-augmentation` |
| `R-HYBRID` | model + lookup fusion | `R-TRN-hybrid` |
| `R-EVAL` | regression gate | `R-EVL-regression` |
| `R-DARK` | dark-genome residual spike | `R-ANL-dark-residual` |
| `*-DEC-NNN` | decision records | unchanged — decision IDs stay as-is |
| `H-*-NN` | hypothesis IDs | unchanged |
| `OPEN-NNN` | open issues | unchanged |

Historical entries are NEVER rewritten to the new scheme. This table exists so a reader
of the ledger can map between them.
