# Literature: local-memorization-beats-global-learning (the wall this project hit)

**Date:** 2026-06-25 · **Method:** citation-verified multi-agent literature sweep (8 fields,
search→adversarial-verify→synthesize; hallucinated/mis-attributed cites caught and corrected).
**One line:** our "chem-kNN beats every global model" wall is a *named, characterized,
cross-field regime* — not a modeling failure and not a leak. The honest move is to **reframe
the project around the phenomenon**, not to keep trying to beat the local baseline head-on.

---

## 1. The question we put to the literature

We have a gene × condition fitness matrix; the task is within-gene ranking of conditions
(NDCG@5). A non-parametric **chem-kNN** — predict a gene's fitness at a held-out condition from
*that same gene's* fitness at chemically-near conditions — beats every global parametric model
(MLP over frozen ProteomeLM embedding + chemistry; linear MF; ranking losses; capacity scaling).
Adding training organisms **hurts** (negative transfer). Naive model+kNN **score-fusion fails**.
The only parametric win is on **cold genes** (zero training rows) vs a weak population baseline,
and it is small. Q1: is "local memorization beats global learning" a recognized phenomenon, or
are we cheating? Q2: what techniques address it that are *categorically different* from the naive
fusion we already failed at?

## 2. Verdict: a named regime, in ≥5 fields. chem-kNN winning is EXPECTED, not cheating.

The single most on-point precedent is titled for our situation: **Khandelwal et al.,
"Generalization *through* Memorization: Nearest Neighbor Language Models" (kNN-LM), ICLR 2020**
([arXiv:1911.00172](https://arxiv.org/abs/1911.00172)). A frozen global model is beaten *on the
long tail* by a kNN lookup over its own representation space; the field's response was **not**
"the model failed" — it was "keep the nonparametric memory as a permanent first-class component."

| Field | Result | Maps to our wall |
|---|---|---|
| Learning theory | **Feldman**, long-tail memorization is *necessary* for near-optimal generalization, [STOC 2020](https://arxiv.org/abs/1906.05271); empirical companion Feldman & Zhang, NeurIPS 2020 | If each gene is a rare subpopulation, a smooth global map *provably can't* match a memorizer |
| NLP | **Xu, Alon, Neubig**, *why kNN-LM works*, ICML 2023 | The naive blend fails because the gain is **representation + distance-temperature, not ensembling** — explains R-HYBRID's failure directly |
| Tabular ML | **Grinsztajn** ([NeurIPS 2022](https://arxiv.org/abs/2207.08815)); **McElfresh** ([NeurIPS 2023](https://arxiv.org/abs/2305.02997)) | MLPs are too smooth / rotation-invariant for irregular per-instance targets → predicts our capacity-scaling failure |
| Recommenders | **Wide & Deep** ([Cheng 2016](https://arxiv.org/abs/1606.07792)); *Are We Making Progress?* ([Dacrema 2019](https://arxiv.org/abs/1907.06902)); NCF-vs-MF Revisited ([Rendle 2020](https://arxiv.org/abs/2005.09683)) | Names our exact memorization-vs-generalization split; deep models "over-generalize on sparse, high-rank" matrices — our gene×condition matrix |
| Cheminformatics / QSAR | applicability domain ([Sahigara 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3679843/)); time-split ([Sheridan 2013](https://pubs.acs.org/doi/10.1021/ci400084k)) | A similarity lookup is the strong *in-domain* baseline by design; the model earns its keep only *out-of-domain* |

In-domain precedent (frame as **analog, not same-task**; the genomics framing had the weakest
citation reliability — only this one is solidly verified): **Ahlmann-Eltze, Huber & Anders,
"Deep-learning perturbation prediction does not outperform simple linear baselines," Nature
Methods 2025** ([PMC12328236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/)).

**Are we cheating? No.** chem-kNN exploits each gene's *own observed* fitness at a chemically-near
condition = an in-domain near-neighbor. A strong similarity baseline in-domain and a model that
only earns its keep out-of-domain is the textbook expectation. Our three failure modes are
*predicted*: capacity fails (Rendle: an MLP struggles to even learn a dot product), more organisms
hurt (Feldman: averaging dilutes per-gene memorized structure). Our gate discipline (beat a tuned
baseline with disjoint CIs) is exactly what Dacrema/Rendle prescribe.

## 3. Honest evaluation in this regime

Random / within-gene condition-holdout splits **manufacture near-neighbors** and reward lookup
(Sheridan 2013; Walters, *Splitting Chemical Datasets*, 2024; Elangovan et al., *Memorization vs.
Generalization: Quantifying Data Leakage*, EACL 2021). The genuine generalization test removes
local support — **cold-gene** (our R-COLD), cold-chemical, or leave-group-out. R-COLD is the
field-standard scaffold/leave-group-out move, and it already shows the model wins where the lookup
has no neighbor. The diagnostic that formalizes "are we cheating?": **similarity-stratified
NDCG@5** — bucket held-out cells by the gene's distance to its nearest in-train condition; expect
kNN to dominate the near bucket and collapse in the far bucket where the model should win.

## 4. Techniques that are categorically different from the failed naive fusion

The lit is emphatic: **one scalar blend weight (R-HYBRID) is the documented *weak* version.** The
principled successors reuse the locked model + the existing chem-kNN datastore and preserve the
cold-gene win by construction.

**Tier 1 (low effort, do first):**
1. **ResMem** (Yang et al., [NeurIPS 2023](https://arxiv.org/abs/2302.01576)) — kNN memorizes the
   **residual** the MLP got wrong, not a competing fitness prediction. No blend weight to mis-set.
   *Recommended first hybrid.*
2. **Correct-and-Smooth** (Huang et al., [ICLR 2021](https://arxiv.org/abs/2010.13993)) — propagate
   a gene's own residuals over a chemistry-similarity condition graph; graph-native sibling of ResMem.
3. **Adaptive Meta-k gate** (Adaptive kNN-MT, [Zheng 2021](https://arxiv.org/abs/2105.13022);
   SPALM, [Yogatama 2021](https://arxiv.org/abs/2102.02557)) — a tiny net emits a **per-query** k +
   weight from neighborhood geometry + a calibrated distance-temperature. The literal documented
   successor to the fusion we failed at.

**Tier 2 (higher ceiling):** **EASE** (Steck, [WWW 2019](https://arxiv.org/abs/1905.03375)) — a
*learned* condition-condition similarity matrix; strongest single shot at beating chem-kNN on its
own terms. **TabR** (Gorishniy, [ICLR 2024](https://arxiv.org/abs/2307.14338)) trained-retrieval;
**metric learning** (NCA/LMNN/deep-kernel) — learn the *metric the kNN indexes*, not the prediction
(guard against deep-kernel collapse, Ober et al. 2021).

## 5. Dead-ends + the realistic ceiling (stated by the literature)

- **More MLP capacity/depth** — ruled out (Zhang 2017; Rendle 2020; Grinsztajn 2022). Confirmed by us.
- **More training organisms / global pooling** — Feldman + homophily-consistency explain the
  negative transfer we observed. Do not retry augmentation.
- **Any naive scalar-λ blend** — the documented weak version (SPALM, Adaptive kNN-MT).
- **A smooth global map as the PRIMARY warm predictor** — loses on irregular per-instance targets
  by construction. **Realistic ceiling:** on warm genes the best is to *match* a learned-local
  method (EASE/TabR) or *correct its residual* (ResMem). The genuine generalization win lives on
  **cold genes / out-of-applicability-domain** (R-COLD) — the program should *characterize and
  widen that regime*, not beat kNN everywhere.

## 6. The publishable reframe this implies

Not "we built a better predictor" (we won't). The defensible paper: **a rigorous characterization
of a memorization-dominated conditional-essentiality task** — strong tuned baseline, denominator
parity, hierarchical-bootstrap CIs, **similarity-stratified / applicability-domain-aware
evaluation**, the **cold-gene generalization result** (R-COLD / GENE-NW), and a **principled
local+global hybrid** (ResMem / Correct-and-Smooth) that matches the local baseline and owns the
cold regime. The "this feels like cheating" worry becomes the thesis. Modest impact, but real and
well-positioned. *(A separate active line — see the reframe search — asks whether a different
QUESTION in this space is more interesting than ranking at all.)*

## 7. Citation-reliability notes (corrections the verifier applied)

- Dacrema et al. 2019: use **"6 of 7"** reproducible neural methods beaten by simple baselines —
  **not** the "11 of 12" that circulates.
- The "Sasse/Mostafavi" personal-transcriptome citation conflates two *separate* companion Nature
  Genetics 2023 papers (Huang et al., s41588-023-01574-w; Sasse et al., s41588-023-01524-6).
- "one PCA still rules them all" — real authors are **Bendidi et al.** (NeurIPS 2024 wksp), not the
  fabricated names in the first pass.
- The activity-cliff "memorization" framing should be cited to **Elangovan 2021 / Walters 2024**,
  not mis-attributed to van Tilborg et al. 2022 (which attributes the gap to representation limits).
- Genomics-specific perturbation benchmarks (PertEval-scFM, Systema, Rastogi/Mostafavi) surfaced as
  plausible but were **not** text-verified — fetch before using as load-bearing cites.

## Key references (verified)

- Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. *Generalization through Memorization: Nearest
  Neighbor Language Models.* ICLR 2020. arXiv:1911.00172.
- Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC 2020. arXiv:1906.05271.
- Feldman, Zhang. *What Neural Networks Memorize and Why.* NeurIPS 2020.
- Xu, Alon, Neubig. *Why do Nearest Neighbor Language Models Work?* ICML 2023.
- Grinsztajn, Oyallon, Varoquaux. *Why do tree-based models still outperform deep learning on
  tabular data?* NeurIPS 2022 D&B. arXiv:2207.08815.
- McElfresh et al. *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* NeurIPS 2023 D&B. arXiv:2305.02997.
- Cheng et al. *Wide & Deep Learning for Recommender Systems.* DLRS@RecSys 2016. arXiv:1606.07792.
- Ferrari Dacrema, Cremonesi, Jannach. *Are We Really Making Much Progress?* RecSys 2019. arXiv:1907.06902.
- Rendle, Krichene, Zhang, Anderson. *Neural Collaborative Filtering vs. Matrix Factorization
  Revisited.* RecSys 2020. arXiv:2005.09683.
- Steck. *Embarrassingly Shallow Autoencoders for Sparse Data (EASE).* WWW 2019. arXiv:1905.03375.
- Huang, He, Singh, Lim, Benson. *Combining Label Propagation and Simple Models Out-performs Graph
  Neural Networks.* ICLR 2021. arXiv:2010.13993.
- Yang, Wang, Chen, Kumar, et al. *ResMem: Learn what you can and memorize the rest.* NeurIPS 2023. arXiv:2302.01576.
- Zheng et al. *Adaptive Nearest Neighbor Machine Translation.* ACL 2021. arXiv:2105.13022.
- Yogatama, de Masson d'Autume, Kong. *Adaptive Semiparametric Language Models (SPALM).* TACL 2021. arXiv:2102.02557.
- Gorishniy, Rubachev, Kartashev, et al. *TabR: Tabular Deep Learning Meets Nearest Neighbors.* ICLR 2024. arXiv:2307.14338.
- Bottou, Vapnik. *Local Learning Algorithms.* Neural Computation 1992.
- Elangovan, He, Verspoor. *Memorization vs. Generalization: Quantifying Data Leakage in NLP
  Performance Evaluation.* EACL 2021.
- Sheridan. *Time-Split Cross-Validation.* JCIM 2013.
- Sahigara et al. *Defining a novel k-NN approach to assess the applicability domain of a QSAR
  model.* J. Cheminformatics 2013.
- Ahlmann-Eltze, Huber, Anders. *Deep-learning perturbation-effect prediction does not outperform
  simple linear baselines.* Nature Methods 2025.
