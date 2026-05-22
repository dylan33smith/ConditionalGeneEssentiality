# Project Textbook: ConditionalGeneEssentiality

**Target audience:** An incoming PhD student at the intersection of computational biology and machine learning — comfortable with gradient descent and attention mechanisms, but new to transposon mutagenesis; comfortable with bacterial growth assays, but new to protein language models.

---

## 1. Executive Summary & Glossary

### 1.1 What This Project Does

Bacteria are not equally vulnerable to the removal of any given gene in every environment. A gene encoding a riboflavin biosynthesis enzyme may be dispensable when riboflavin is supplied in the growth medium, but essential when it is not. This context-dependence is called **conditional gene essentiality**. Mapping it systematically — across thousands of genes and hundreds of growth conditions, spanning 48 bacterial species — is the central objective.

The input to our model is a `(gene, condition)` pair. The output is a continuous scalar, the **fitness score** (`fit`), quantifying how much the loss of that gene hurts bacterial growth under that condition. The task is regression, not classification. The current best model achieves RMSE ≈ 0.499 on a held-out organism set (compared to a global-mean null baseline of ≈ 0.550), a meaningful but still modest improvement. The T7-prep diagnostics revealed that cross-organism transfer fails at the ranking level, motivating a pivot to within-organism cross-condition evaluation as the scientifically meaningful next frontier.

### 1.2 Term Dictionary

**Tn-seq (Transposon Insertion Sequencing)**
- *Analogy:* Imagine randomly disabling switches on a circuit board, one at a time, and measuring how bright the lights stay. More dimming → more important switch.
- *Technical definition:* A genome-wide loss-of-function screen that saturates a bacterial genome with transposon insertions. Each insertion disrupts a gene. Illumina sequencing quantifies each mutant's relative abundance before and after competitive growth. Genes where mutants deplete → fitness cost.
- *Why we use it:* It is the highest-throughput technology available for measuring essentiality continuously and genome-wide. The raw data is in `data/raw/feba.db`, a SQLite database with 27.4 million rows across 48 organisms.

**`fit` (Fitness Score)**
- *Analogy:* The fraction of its original value that a population of gene-X-knockouts retains after growing in condition Y, expressed on a log₂ scale per doubling.
- *Technical definition:* `fit = log₂(post_count / pre_count) / nGenerations`, computed by the FEBA pipeline per gene per experiment. Negative values indicate growth disadvantage. Distribution: median ≈ −0.014, p1 ≈ −3.044, p95 ≈ 0.464 — heavy left tail. Already normalized per-generation by FEBA; no further time-normalization is needed.
- *Why we use it:* It is the direct readout from the FEBA pipeline and the most widely used quantitative essentiality metric in the Tn-seq literature.

**ProteomeLM / PLM-L (Protein Language Model, Proteome-contextualized)**
- *Analogy:* GPT for protein sequences, but additionally trained to understand each protein in the context of the full proteome of its organism.
- *Technical definition:* A transformer protein language model built on top of ESM-C 600M. Its representations at layer 8 of 18 are 1152-dimensional vectors extracted per protein via mean-pooling over residues. The ProteomeLM-L pass adds a proteome-context attention step over all proteins in an organism, enabling each gene's embedding to reflect its role within the organism's broader proteome.
- *Why we use it:* Encodes evolutionary and functional information about each gene in a dense, learnable-adjacent format. Frozen at layer 8 throughout (locked L3). T5-C confirmed it adds 0.006 RMSE over raw ESM-C; T5-B confirmed layer 8 is optimal (later layers over-specialize to the PLM pre-training objective).

**Canonical_ID (Canonical Chemical Identifier)**
- *Analogy:* A universal SKU for chemicals, cutting through the fact that "glucose," "D-glucose," and "dextrose" all refer to the same molecule.
- *Technical definition:* A project-internal string key assigned by the v4 media workbook (`data/media_composition_v4.xlsx`, sheet `Media_Components_ML`) that standardizes ingredient names across all media. 425 distinct IDs appear in the training split, plus two special tokens: `<UNK>` (unseen medium component) and `<UNK_STRESSOR>` (unseen chemical stressor). Coverage: ≥97% mapped on average across organisms; worst-case organism has 50% mapping.
- *Why we use it:* Enables chemistry-aware generalization — even if two organisms use different medium names, they share Canonical_IDs. S1 confirmed ≥95% of Canonical_IDs appear in training across all candidate split protocols.

**Multihot Encoding**
- *Analogy:* A 425-bit checklist: "does this growth condition contain ingredient X?"
- *Technical definition:* A binary vector of shape `[425]` where entry `i` = 1 if Canonical_ID `i` is present in the experiment's medium or stressor, 0 otherwise. Concentrations are excluded (see T1-B.3). Fed into `nn.EmbeddingBag(vocab_size=427, embed_dim, mode="sum")`, which sums learned per-Canonical_ID embeddings. Result: a fixed-dim chemistry vector.
- *Why chosen:* Beats media-name encoding by 0.04–3.26 RMSE depending on the test protocol (T1-A, T1-A.2). Concentrations were dropped because units are not harmonized across the workbook; the empirical gain from log1p amounts was sub-threshold (T1-B.3).

**RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error)**
- *Technical definition:* `RMSE = √(1/n Σ(ŷᵢ − yᵢ)²)`. `MAE = 1/n Σ|ŷᵢ − yᵢ|`. Both computed on the same denominator row set (denominator parity rule).
- *Why co-primary:* `fit` is heavy-tailed (p1 ≈ −3.044). RMSE overweights extreme errors; MAE is more median-focused. A Huber loss experiment (T4-A) demonstrated their independence: Huber improved MAE but regressed RMSE. Both must improve above the S2-locked thresholds (0.0212 RMSE, 0.005 MAE) for any promotion.

**Within-gene Spearman Correlation**
- *Analogy:* For each gene, rank all growth conditions by true essentiality and by predicted essentiality, then measure how well the ranks agree.
- *Technical definition:* For each gene with ≥ 5 measured conditions and IQR(y_true) ≥ v_min, compute Spearman rank correlation between true and predicted `fit`. Report the mean across eligible genes. Implemented in `src/evaluation/metrics.py:within_gene_spearman()`.
- *Why it matters:* RMSE/MAE measure absolute accuracy; Spearman measures conditional ranking — which is what matters biologically (identifying which conditions most compromise a gene). Under cross-organism holdout, model Spearman ≈ 0.045 vs a noise floor of 0.43–0.49 (T7-prep), indicating cross-organism transfer fails entirely at the ranking task.

**H-BASE-01 (Additive Baseline Gate, Revised)**
- *Technical definition:* Any model must beat the best non-global baseline. Under organism holdout, the additive model `fit ~ α[gene] + β[condition]` collapses to global mean (the val organism's genes and conditions are never seen in training, so `α` and `β` are undefined → collapse to intercept). The one surviving non-trivial baseline is `embedding_NN` (k-nearest-neighbor in PLM embedding space). This is the actual gate.
- *Why revised from original:* S2 showed that 4/5 baselines collapse to global mean under cold-start; only `embedding_NN` survives. The gate correctly tests "does the model do better than the best available cold-start strategy."

**FiLM (Feature-wise Linear Modulation)**
- *Analogy:* A dynamic brightness/contrast knob per neuron, set by the condition features.
- *Technical definition:* `h_out = γ(condition) ⊙ h_gene + β(condition)`, where `γ` and `β` are linear projections of the condition vector. Applied element-wise to the hidden representation.
- *Why we tested it:* FiLM is theoretically the natural way to let condition features modulate gene representations — the condition "tunes" each gene feature independently rather than just concatenating them.
- *Why it failed:* Sub-threshold in T2-C (RMSE gap 15% of threshold); reversed at depth in T3-D (FiLM 0.5074 vs concat 0.5057). Residual blocks subsume whatever benefit FiLM provides, making explicit modulation redundant.

**EmbeddingBag**
- *Technical definition:* `torch.nn.EmbeddingBag(vocab_size, embed_dim, mode="sum")` — a lookup table that takes a variable-length set of integer indices and returns their summed embeddings in a single forward pass.
- *Why used for multihot chemistry:* A medium is an order-invariant set of Canonical_IDs. EmbeddingBag with `mode="sum"` models "the chemistry vector is the sum of its ingredient embeddings" — a biologically reasonable inductive bias (each ingredient contributes independently in the first approximation) implemented efficiently in one fused CUDA op.

**ResidualMLP / ResBlock**
- *Technical definition:* A building block `h_{k+1} = h_k + FF(h_k)` where `FF` is Linear → ReLU → Linear. The skip connection allows gradient flow without vanishing, enabling deeper architectures.
- *Why 2 layers:* T3-A showed RMSE gap 0.0093 when going 1→2 layers (substantial, validated), and only 0.0016 for 2→4 (marginal). Parsimony: take the gain where it is large, stop where it is marginal.

**Organism-Holdout Split (`multi_org_balanced`)**
- *Technical definition:* Training: 43 organisms; validation: 4 organisms {`pseudo3_N2E3`, `BFirm`, `ANA3`, `SyringaeB728a_mexBdelta`}; test: 1 organism {`PV4`}. No gene or experiment appears in both train and val/test partitions.
- *Why this design:* Tests generalization to entirely novel organisms — the hardest and most scientifically meaningful generalization. Dense chemistry overlap (val chemistry ≥95% seen in training) ensures that Spearman failures reflect organism-level adaptation failures, not chemistry unseen-ness.

**Morgan / RDKit / MACCS Fingerprints**
- *Analogy:* A checklist of molecular "features" (circular substructures, or functional-group presence/absence) rather than identity tags.
- *Technical definition:* Morgan fingerprints (2048-bit) encode circular chemical substructures of radius 2. RDKit fingerprints (2048-bit) encode topological paths of length 1–7. MACCS fingerprints (167-bit) are a fixed set of predefined structural keys. All computed via RDKit after PubChem SMILES lookup. Stored in `data_contract/chemistry/canonical_fingerprints.npz`.
- *Why tested (T6-A):* Fingerprints enable generalization to structurally similar but unseen chemicals. *Why they failed:* All three regress vs multihot because val chemistry is 98%+ seen in training — the model needs identity, not structural proximity, at this evaluation.

**Hydra (Configuration Framework)**
- *Technical definition:* A Python framework that composes configuration from YAML files via a `defaults` list. Experiments are run as `python -m src.cli.run_experiment +experiment=T1-A_granularity train.seed=0,1,2 -m`. The `-m` flag triggers Hydra multirun (sweeping all combinations). Every run writes a `.hydra/config.yaml` snapshot.
- *Why:* Separates configuration from code, enforces reproducibility, and handles parameter sweeps without hand-rolled loops.

---

## 2. Project Architecture (The Tier System)

### 2.1 The Two-Phase Pipeline

The project separates into two phases with hard stage gates:

**Stages S0–S5** (foundation): Answer a specific structural question and lock an artifact that all downstream work inherits. Sequential and non-revisable without a formal decision-ledger entry.

| Stage | Question Answered | Locked Artifact |
|---|---|---|
| S0 | Can we reproduce bit-identical results? | Smoke pipeline + manifest schema |
| S1 | What does the data actually look like? | 4 candidate split protocols, 25 figures |
| S2 | Which metrics and baselines are trustworthy? | `eval_policy.yaml`, baseline registry |
| S3 | Which split protocol is primary? | `locked_protocol.yaml` = `multi_org_balanced` |
| S4 | What features go into the model? | Feature contract `de21504134c84a6c` (Option D) |
| S5 | Which rows are used for training? | `quality_policy.yaml` = `weighted_full` |

**Tiers T1–T6+** (optimization): Each tier owns exactly one architectural dimension and tests a pre-registered hypothesis. The ordering encodes a deliberate logic:

```
T1 (Chemistry Encoding) → T2 (Fusion Topology) → T3 (Capacity/Depth) →
T4 (Loss + Target) → T5 (Gene Embedding) → T6 (Chemistry Fingerprints) →
T7 (Within-org Cross-Condition Ranking — not yet started)
```

### 2.2 Why This Order?

Each tier's decision changes the input or output shape of the component that the next tier tests. If T2 (fusion) ran before T1 (chemistry encoding), fusion experiments would be confounded by an unoptimized chemistry representation. The strict gate also prevents premature optimization: a "no winner" result produces no promotion and no artifact change — this happened in T1-C, T1-D, T1-E, T4-A, T4-B, T4-C, T5-D, and T6-A (eight of fourteen completed experiments). A 57% no-winner rate is expected and correct behavior in this design.

### 2.3 No-Winner Semantics

Eight experiments produced no architecture change. This is not failure — it is information. Each no-winner result rules out a hypothesis, constrains the search space, and provides context for interpreting the broader pattern (e.g., T4-C's no-winner diagnosis that "the bottleneck is representation quality, not training budget" directly motivated T5's gene adapter approach).

---

## 3. Tier-by-Tier Breakdown

### Stage S0: Reproducibility & Governance

**Hypothesis:** The full pipeline runs end-to-end with bit-identical outputs at fixed seed.

**Method:** A smoke run trains on 80K randomly sampled rows (80/20 split), emitting `run_manifest.json` validated against `data_contract/schemas/run_manifest_v1.schema.json`. Two reruns at identical seed produced identical `smoke_digest` values. The manifest schema requires git SHA, data checksums (feba.db, workbook_v4, embeddings, canonical), split ID, preprocessing artifact ID, seed, RMSE, MAE, Spearman, `scored_rowset_hash`, and unknown-category rate.

**Code:** `src/cli/run_experiment.py` (Hydra entrypoint), config at `configs/stage/s0_reproducibility.yaml`.

**Result (S0-DEC-001):** Platform gate established. 15,936 val rows scored — sufficient for harness integrity, insufficient for biological conclusions.

---

### Stage S1: Data Characterization

**Key Findings (25 required figures, all committed to `research_log/figures/stage1/`):**

- **Organism row imbalance is extreme:** `Btheta` (2.1M rows) vs `RalstoniaUW163` (22,770 rows) — 92× ratio. Macro-averaged per-organism metrics are mandatory.
- **Media-string overlap is near-zero:** 67.7% of organism pairs share 0 media names. But canonical chemistry overlap is dense: 0% of pairs share zero Canonical_IDs (min 4, median 69). This structurally validates chemistry-based features over media-name features as the primary encoding.
- **`fit` is heavy-tailed:** p1 ≈ −3.044, p5 ≈ −0.857, p95 ≈ 0.464. RMSE + MAE co-primary is empirically necessary.
- **Embedding coverage is 98.7% on average, but `azobra` has only 13.6% covered rows.** Per-organism denominator parity tracking is mandatory.
- **4 candidate split protocols emitted:** `largest_by_rows`, `multi_org_balanced`, `low_overlap_stress`, `homology_0.85`.
- **H-HOMO-01 triggered at 1.6σ** (homology-stratified Spearman varies by cosine-similarity bin), below the 0.5σ threshold for scheduling homology-masked protocol. Diagnostic only.

**Decision (S1-DEC-001):** All 4 protocols advance to S2. `representation_mode` composition (extract/mix/physical) ratified as a required diagnostic axis.

---

### Stage S2: Evaluation Trustworthiness

**Hypothesis (H-BASE-01):** Any model must beat an additive baseline to be scientifically meaningful.

**Method:** Five baselines evaluated on all 4 candidate protocols under denominator parity:
1. Global mean
2. Gene mean (per-gene average `fit` in training)
3. Condition mean (per-condition average `fit` in training)
4. Additive model (`fit ~ α[gene] + β[condition]`, OLS)
5. Embedding NN (k-nearest-neighbor in PLM embedding space)

**Critical insight:** Under organism holdout (cold-start), baselines 2–4 collapse to global mean because neither the val organism's genes nor its conditions were in training. Only `embedding_NN` survives, using PLM embedding proximity to transfer fitness expectations from structurally similar train genes.

**Revised gate (S2-DEC-001):** Beat `embedding_NN` on `multi_org_balanced`. RMSE threshold locked at 0.0212; MAE threshold at 0.005. Within-gene Spearman near noise level (0.006–0.012) for all baselines under organism holdout — this is an important early signal of the cross-org problem that T7-prep later quantifies rigorously.

---

### Stage S3: Split Protocol Lock

**Decision (S3-DEC-001):**
- **Primary:** `multi_org_balanced` — val={pseudo3_N2E3, BFirm, ANA3, SyringaeB728a_mexBdelta}, test={PV4}.
- **Diagnostics:** `low_overlap_stress` (harder chemistry distribution) and `homology_0.85` (sequence similarity controlled, 0.85 cosine cutoff).
- **Rationale:** `multi_org_balanced` has the highest chemistry overlap (≥95% seen Canonical_IDs), meaning performance differences reflect organism-level generalization, not chemistry coverage.

---

### Stage S4: Feature Contract

**Original design (S4-DEC-001):** condition_1..condition_4 encoded as categorical metadata.

**Problem (OPEN-001):** condition_1..4 are *chemical stressors* (e.g., "5 mM H₂O₂"), not categorical descriptors. Encoding them as categories loses all chemical structure.

**Revised design (S4-DEC-002, Option D):**
- `experiment_chemistry.parquet` (long format): one row per (experiment_id, canonical_id) with columns `role ∈ {medium, stressor}`, `amount`, `log1p_amount`. Stressors parsed, canonicalized, and added as chemistry rows alongside medium components.
- `experiment_metadata.parquet` (wide format): one row per experiment with columns `oxygen`, `experiment_group`, `liquid_state`, `temperature_c`, `pH`, `shaking_rpm`.
- Artifact ID: `de21504134c84a6c`.

---

### Stage S5: Training-Recipe Lock

**Decision (S5-DEC-001):** `weighted_full` — train on all rows with loss weight = f(|t|) (downweight low-confidence fitness rows). Full organism pool (all 48 organisms). The `strict_slice` alternative discards too many rows from low-coverage organisms, worsening macro-averaged metrics. Weighting achieves noise-robustness without discarding signal.

---

### Tier 1 (T1): Chemistry Representation

**What is locked:** How the 425-dim chemistry vocabulary is encoded into a fixed-dimension vector.

#### T1-A: Granularity (H-ENC-01)

**Hypothesis:** Component-level multihot encoding beats media-name-level encoding because it generalizes across media that share chemical components but differ in name.

**Code:**
```python
# src/models/representations/condition_encoders.py
class MultihotConditionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout=0.0):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="sum", padding_idx=0)
        self.dropout = nn.Dropout(dropout)
    def forward(self, component_ids):
        return self.dropout(self.embed(component_ids))

class MediaIDEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    def forward(self, media_ids):
        return self.embed(media_ids)
```

**Results (multi_org_balanced):**
- multihot: RMSE 0.5153 ± 0.0016, MAE 0.2958 ± 0.0013
- media_id: RMSE 0.5575 ± 0.0042, MAE 0.3208 ± 0.0022
- Gap: RMSE 0.0422 >> 0.0212 threshold. H-ENC-01 supported. Decision: **multihot promoted by tie-breaker** (no arm cleared the embedding_NN gate; multihot wins on all metrics and mechanism).

**T1-A.2 stress test on `largest_by_rows`:**
- multihot: RMSE 0.5897 vs media_id: RMSE 3.8451 — gap 3.26 RMSE (650× threshold).
- On a protocol with *unseen media names* in val, media_id catastrophically fails (embedding undefined for unseen IDs). Multihot generalizes because it encodes chemistry, not identity. **H-ENC-01 strongly confirmed.** `largest_by_rows` is now mandatory as a diagnostic for all future T1+ promotions.

#### T1-B: Numeric Transform (H-ENC-02)

**Hypothesis:** Including concentration amounts (log1p-transformed) adds signal beyond presence/absence.

**Arms:** raw amounts, log1p amounts, bounded [0,1] amounts, binary presence only.

**Results:** log1p and bounded statistically tied; both beat raw (raw is worse on MAE, sub-threshold RMSE gap). **Decision (T1-DEC-003):** log1p promoted by tie-breaker.

#### T1-B.3: Controlled Binary vs log1p (T1-DEC-004 supersedes T1-DEC-003)

**Method:** Controlled comparison isolating encoding from confounders. **Result:** Binary matches or beats log1p.

**Root cause:** Concentration units are not harmonized in the v4 workbook (mg/L, mM, %, µL/L coexist). log1p of heterogeneous units adds noise rather than signal.

**Decision: Binary multihot locked.** Concentrations dropped. T1-DEC-003's secondary finding ("raw < log1p at controlled concentrations") remains valid if concentrations are reintroduced with proper unit harmonization.

#### T1-C: Unknown Chemistry Handling (H-ENC-03)

**Hypothesis:** Learnable UNK embedding outperforms zero-vector for unseen Canonical_IDs.

**Result: Skipped.** The locked `multi_org_balanced` split has fraction_canonical_id_not_in_train_vocab = 0.0. Both UNK arms are functionally identical — zero statistical power. T1-DEC-005: no experiment run.

#### T1-D: Metadata Bundle (H-META-01)

**Hypothesis:** Adding S4 wide-format metadata (oxygen, pH, temperature, shaking, liquid_state, experiment_group) improves over chemistry alone.

**Results:**
- chemistry_only: RMSE 0.5153 ± 0.0016, MAE 0.2958 ± 0.0013
- chemistry_plus_metadata: RMSE 0.5126 ± 0.0002, MAE 0.2940 ± 0.0005
- Gap: RMSE 0.0027 (12.5% of threshold), MAE 0.0018 (36%). Metadata *dramatically stabilizes seed variance* (RMSE std 0.0002 vs 0.0016) — a regularizing effect, not new discriminative content.

**Decision (T1-DEC-006): no_winner.** Gap sub-threshold.

#### T1-E: Decomposition Mode Flags (H-META-02)

**Hypothesis:** Binary role flags (`medium` vs `stressor`) per Canonical_ID improve over Canonical_IDs alone.

**Results:**
- chemistry_only: RMSE 0.5151 ± 0.0018, MAE 0.2959 ± 0.0013
- chemistry_plus_mode_flags: RMSE 0.5149 ± 0.0025, MAE 0.2956 ± 0.0023
- Gap: 1.2% RMSE threshold, 5.5% MAE threshold. Mode flags add seed variance.

**Decision (T1-DEC-007): no_winner.** Mode flags add neither signal nor stability.

**T1 Final Lock:** 425-dim binary multihot over Canonical_IDs (medium + stressor), no concentrations, no metadata, no mode flags.

---

### Tier 2 (T2): Fusion Topology

**What is locked:** How gene embeddings and chemistry encodings are combined. This tier owns all fusion decisions; T3 owns capacity.

#### T2-A: Nonlinearity (H-FUSE-01)

**Architecture (ShallowMLP):**
```python
# src/models/architectures/shallow_mlp.py
class ShallowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
```

**Results:**
- shallow_mlp: RMSE 0.5146 ± 0.0010
- linear_head: RMSE 0.6100 ± 0.0012
- Gap: 0.0954 >> 0.0212 threshold. **Nonlinearity is essential.** Gene × condition interactions cannot be captured by linear combination.

#### T2-B: Early vs Late Fusion (H-FUSE-02)

**Hypothesis:** Early concatenation outperforms two-tower (separate towers combined only at the final layer).

**Results:**
- early_concat: RMSE 0.5140 ± 0.0008
- two_tower: RMSE 0.5627 ± 0.0009
- Gap: 0.0487 >> 0.0212 threshold. **Early fusion is essential.**

**Why early fusion wins:** The most predictive signal is the interaction between specific gene features and specific chemistry features (e.g., "gene X encodes riboflavin synthase, condition Y lacks riboflavin"). Two-tower architectures force towers to produce useful representations independently before they interact, which fails when the signal is the interaction itself.

#### T2-C: FiLM Modulation (H-FUSE-03)

**Hypothesis:** FiLM `γ(condition) ⊙ h_gene + β(condition)` is more expressive than concatenation.

**Results:**
- early_concat: RMSE 0.5153, MAE 0.2958
- film_gated: RMSE 0.5121, MAE 0.2918
- FiLM gated gap: RMSE 0.0032 (15%), MAE 0.0040 (80% of threshold). Near-threshold MAE signal, sub-threshold RMSE.

**Decision (T2-DEC-001): no_winner.** Both metrics must improve above threshold.

**T2 Final Architecture:**
`cat(gene[1152], chem[425]) → Linear(1577, 256) → LayerNorm → ReLU → Dropout(0.1) → Linear(256, 1)`

---

### Tier 3 (T3): Capacity — Depth and Width

**What is locked:** Network depth, width, and block structure (including whether to use residuals).

#### T3-A: Depth (H-CAP-01)

| Depth | RMSE (mean ± std) |
|---|---|
| 1-layer | 0.5152 ± 0.0014 |
| 2-layer residual | 0.5059 ± 0.0008 |
| 4-layer residual | 0.5043 ± 0.0011 |

1→2 gap: 0.0093 (meaningful and reproducible). 2→4 gap: 0.0016 (sub-threshold). **Parsimony: 2-layer residual promoted.**

#### T3-B: Width (H-CAP-02)

| Width | RMSE (mean ± std) |
|---|---|
| 128 | 0.5080 ± 0.0005 |
| 256 | 0.5059 ± 0.0003 |
| 512 | 0.5028 ± 0.0007 |
| 1024 | 0.5013 ± 0.0017 |

1024 has highest seed variance (std 0.0017 vs 0.0003 for 512). 512→1024 gap sub-threshold. **Width 512 promoted** on Pareto stability.

#### T3-C: Embedding Fine-Tune (Conditional)

**Not triggered.** T3-A and T3-B showed clear capacity gains; the trigger (A/B near-tie) was not met.

#### T3-D: FiLM at Depth (Follow-up to T2-C)

**Hypothesis:** FiLM's near-threshold T2-C MAE signal amplifies at depth where it modulates richer representations.

**Results:**
- early_concat (2-layer residual): RMSE 0.5057
- film_gated (2-layer residual): RMSE 0.5074

**FiLM is worse at depth.** The T2-C signal reverses: residual blocks subsume whatever benefit FiLM provides. Explicit modulation is redundant when skip connections are present.

**T3 Final Architecture:**
```
Input: cat(gene[1152], chem[425]) = 1577-dim
→ Linear(1577, 512) → ReLU → Dropout(0.1)
→ ResBlock(512): h + [Linear(512,512) → ReLU → Linear(512,512)]
→ Linear(512, 1)
```

---

### Tier 4 (T4): Optimization Locks

**What is locked:** Loss function, target representation, and training schedule. T4 does not change architecture.

#### T4-A: Loss Family (H-LOSS-01)

| Loss | RMSE | MAE |
|---|---|---|
| MSE | **0.5022** ± 0.0013 | 0.2799 ± 0.0008 |
| Huber δ=0.5 | 0.5027 ± 0.0016 | **0.2769** ± 0.0008 |
| Huber δ=1.0 | 0.5034 ± 0.0016 | 0.2781 ± 0.0012 |
| Huber δ=1.5 | 0.5026 ± 0.0017 | 0.2790 ± 0.0010 |

**Metric conflict:** Huber δ=0.5 improves MAE (−0.003) but regresses RMSE (+0.0005). Co-primary promotion rule requires both to improve. **Decision: no_winner. MSE retained.**

**Interpretation:** Huber's outlier clipping improves the median-focused MAE by reducing the gradient contribution of extreme `fit` values. But those extremes contain real biological signal (very negative fit = strongly essential gene), so RMSE correctly penalizes Huber for discarding them. A winsorized loss (clip targets before MSE) was identified as a potential future direction.

#### T4-B: Target Normalization (H-TARGET-01)

- raw: RMSE 0.5024 ± 0.0011, MAE 0.2798 ± 0.0008
- zscore: RMSE 0.5206 ± 0.0050, MAE 0.2870 ± 0.0034

Z-scoring degrades both metrics (+0.018 RMSE, +0.007 MAE) and increases seed variance 5×. The inverse transform from z-score to raw scale adds noise. **Decision: no_winner. Raw targets retained.**

#### T4-C: Learning Rate Schedule (H-OPT-01)

| Arm | RMSE | MAE |
|---|---|---|
| baseline (8ep, const lr=1e-3) | 0.5029 ± 0.0001 | 0.2800 ± 0.0007 |
| cosine_16ep | 0.5022 ± 0.0002 | 0.2802 ± 0.0009 |
| cosine_32ep | 0.5014 ± 0.0018 | 0.2795 ± 0.0012 |
| cosine_32ep_wd (wd=1e-3) | 0.5014 ± 0.0012 | 0.2792 ± 0.0001 |

All gaps sub-threshold (cosine_32ep_wd: RMSE −0.0015, 7%; MAE −0.0008, 16%). Val metrics plateau around epoch 5–8 while train continues to improve — classic mild overfitting. **Decision: no_winner.** Diagnosis: **the bottleneck is representation quality, not training budget.** This finding directly motivates T5.

**T4 Final Lock:** MSE loss, raw fitness targets, 8 epochs, lr=1e-3 constant, weight_decay=1e-4.

---

### Tier 5 (T5): Gene Embedding Enhancement

**Motivation:** T2–T4 produced only sub-threshold improvements. Val loss curves plateau early while train loss continues to improve — the bottleneck is the gene representation (frozen 1152-dim PLM-L8 vector), not the head or optimizer.

#### T5-A: Learnable Gene Adapter (H-EMB-02)

**Hypothesis:** A learnable MLP adapter between the frozen embedding and the concat point projects the general PLM representation into a task-relevant subspace.

| Arm | RMSE | MAE |
|---|---|---|
| no_adapter | 0.5034 ± 0.0007 | 0.2805 ± 0.0008 |
| adapter_256 | 0.5031 ± 0.0006 | 0.2799 ± 0.0003 |
| adapter_512 | 0.5015 ± 0.0011 | 0.2803 ± 0.0004 |
| **adapter_1024_proj** | **0.4991** ± 0.0011 | **0.2780** ± 0.0008 |

`adapter_1024_proj` architecture: `Linear(1152, 1024) → ReLU → Dropout(0.1) → Linear(1024, 512)`. This reduces the gene dimension from 1152 → 512.

**Why dimension reduction helps:** Same-dim adapters (256, 512) show minimal improvement. The 1152→512 bottleneck forces the adapter to identify which PLM dimensions are task-relevant — a learned dimensionality reduction that discards PLM-specific but fitness-prediction-irrelevant variance. The frozen PLM was trained for masked protein language modeling; the 1152-dim space contains much structure that is irrelevant for predicting bacterial fitness from Tn-seq.

**Decision (T5-DEC-001): adapter_1024_proj promoted.** RMSE gap 0.0043 (20% of threshold), MAE gap 0.0025 (50%). Disjoint bootstrap CIs. First statistically robust, consistent improvement since T3-A depth. Adds ~1.4M parameters (1.07M → 2.45M total).

#### T5-B: PLM Layer Ablation (H-EMB-03)

| Layer | RMSE | MAE |
|---|---|---|
| 0 (input embed) | 0.5027 ± 0.0016 | 0.2831 ± 0.0014 |
| 4 | 0.5033 ± 0.0033 | 0.2811 ± 0.0018 |
| **8 (current)** | **0.5028** ± 0.0003 | **0.2799** ± 0.0007 |
| 12 | 0.5134 ± 0.0021 | 0.2847 ± 0.0004 |
| 18 (final) | 0.5599 ± 0.0011 | 0.3007 ± 0.0013 |

Layers 12 and 18 are dramatically worse. The final PLM layers are over-specialized to ProteomeLM's masked-LM pre-training objective. **Layer 8 confirmed as optimal.** No change.

#### T5-C: ProteomeLM Bypass (H-EMB-04)

- PLM-L8: RMSE 0.5030 ± 0.0001, MAE 0.2801 ± 0.0006
- ESM-C only: RMSE 0.5090 ± 0.0008, MAE 0.2844 ± 0.0004

Gap: RMSE 0.0060 (28%), MAE 0.0043 (86%), disjoint bootstrap CIs. **ProteomeLM's proteome-context layer provides real, measurable value.** It encodes each gene's role within its organism's proteome — information that raw per-protein embeddings cannot capture. Keep ProteomeLM-L8.

#### T5-D: Adapter Variants (H-EMB-05)

**Hypothesis:** The T5-A winner can be improved via different output dimension, adapter width, adapter depth, or LayerNorm.

| Arm | RMSE | MAE |
|---|---|---|
| **baseline_1024_512** | **0.5000** ± 0.0013 | **0.2784** ± 0.0012 |
| out_dim_256 | 0.5003 ± 0.0006 | 0.2798 ± 0.0007 |
| hidden_2048 | 0.5015 ± 0.0010 | 0.2805 ± 0.0008 |
| two_hidden_layers | 0.5006 ± 0.0003 | 0.2783 ± 0.0003 |
| layernorm | 0.5020 ± 0.0003 | 0.2796 ± 0.0011 |

No arm improves on baseline_1024_512 on both metrics. Wider adapter (hidden_2048) degrades. LayerNorm hurts. **Decision: no_winner. Baseline adapter retained.** The T5-A design was already at the optimum for this adapter family.

**T5 Final Architecture:**
```
Gene: frozen PLM-L8[1152]
  → Adapter: Linear(1152,1024) → ReLU → Dropout(0.1) → Linear(1024,512) → 512-dim
Chem: multihot → EmbeddingBag(427, embed_dim) → [425-dim]
Fusion: cat([512], [425]) = 937-dim
  → Linear(937, 512) → ReLU → Dropout(0.1)
  → ResBlock(512): h + [Linear(512,512) → ReLU → Linear(512,512)]
  → Linear(512, 1)
```
Total parameters: ~2.45M. Best val RMSE: 0.4991 ± 0.0011 (T5-A seeds).

---

### Tier 6 (T6): Chemical Fingerprint Encoders

**Motivation:** The 425-dim binary multihot vector is limited to chemicals seen in training. Structural fingerprints encode chemical *substructure*, potentially enabling generalization to novel chemicals via structural similarity. This would extend the scope claim (L7) beyond the current "known chemistry vocabulary" boundary.

**Chemistry pipeline built for T6:**
1. PubChem SMILES lookup for each Canonical_ID → `data_contract/chemistry/canonical_id_smiles.json` (2552 entries)
2. RDKit fingerprint computation → `data_contract/chemistry/canonical_fingerprints.npz` (Morgan 2048-bit, RDKit 2048-bit, MACCS 167-bit)
3. Per-experiment mean-pooled fingerprints → `data_contract/chemistry/experiment_fingerprints.npz`

#### T6-A: Fingerprints vs Multihot (H-CHEM-01)

**Arms:** multihot (control), morgan_only (2048-bit), rdkit_only (2048-bit), maccs_only (167-bit), morgan_plus_multihot, maccs_plus_multihot.

| Arm | RMSE (mean ± std) | MAE (mean ± std) |
|---|---|---|
| **multihot** | **0.4990** ± 0.0010 | **0.2780** ± 0.0007 |
| morgan_only | 0.5016 ± 0.0003 | 0.2808 ± 0.0008 |
| rdkit_only | 0.5038 ± 0.0004 | 0.2814 ± 0.0015 |
| maccs_only | 0.5053 ± 0.0010 | 0.2811 ± 0.0009 |
| morgan_plus_multihot | 0.5011 ± 0.0004 | 0.2788 ± 0.0006 |
| maccs_plus_multihot | 0.4998 ± 0.0017 | 0.2798 ± 0.0010 |

All fingerprint-only arms regress vs multihot. Hybrid arms come closer (maccs_plus_multihot: 0.4998) but do not beat multihot alone (0.4990).

**Why fingerprints fail here:** Val chemistry is ≥98% seen in training. In this regime, structural similarity introduces noise (conflating different chemicals that share substructure) rather than signal (structural similarity only helps for truly unseen chemicals). The fingerprint's theoretical advantage does not materialize because the primary split was specifically designed to have high chemistry overlap.

**Decision: no_winner. Multihot retained.** Chemical fingerprints remain a deferred experiment for when novel-chemistry generalization is required. The fingerprint infrastructure (SMILES lookup, RDKit pipeline) is now built and available for that future work.

---

### T7-Prep: Within-Gene Ranking Diagnostics and the Cross-Org Pivot

**Context:** After six tiers of architecture optimization, the model achieves RMSE ≈ 0.499 — meaningfully better than the null baseline (0.550). But the within-gene Spearman (the biologically meaningful ranking metric) tells a starkly different story.

**T7-Prep Diagnostics (D1–D4), figures in `research_log/figures/t7_prep/`:**

**D1 (Cross-organism Spearman):** Model within-gene Spearman ≈ **0.045** on the `multi_org_balanced` val set. All baselines (global mean, gene mean, embedding_NN) also near-zero: 0.006–0.012. The model is not learning condition-specific gene responses for held-out organisms.

**D2 (Cross-condition consistency):** Within the val organisms, testing whether the model correctly ranks conditions for a given gene across novel conditions. Also near-zero. Confirms the ranking failure is not a calibration issue — it is a transfer failure.

**D3 (Within-org noise floor):** Permutation test on training organisms. Randomly shuffle condition labels within each gene and compute Spearman. Noise floor: **0.43–0.49**. This is the expected Spearman for a model that has effectively memorized gene-level means and cannot rank conditions for truly novel organisms. The gap: model 0.045 vs noise floor 0.43–0.49.

**D4 (IQR per val organism):** The val organisms have median within-gene IQR ≈ 0.08. For ranking to be meaningful, genes must exhibit condition-dependent fitness variation. Low IQR means ranking is near-random even for a perfect predictor — this is an additional factor in the near-zero Spearman.

**Interpretation:** The 9.3% RMSE improvement (0.550 → 0.499) reflects better mean fitness prediction. But the model performs no better than chance at the ranking task for held-out organisms. The root cause is structural: the model learns to map `(gene, condition_chemistry)` → `fit` using both gene identity (frozen PLM embedding) and condition identity (multihot over 425 Canonical_IDs). For a novel organism, the gene embeddings are in a different part of the PLM space, and the model has no mechanism to adapt its fitness predictions based on what it knows about *that organism's* gene network.

**Pivot to within-organism cross-condition framing:** Hold out conditions (not organisms), and ask whether the model correctly ranks the essentiality of known genes under novel chemical conditions within the same organism. This reframes from "generalize to novel organisms" to "generalize to novel chemical conditions within known organisms." It is a different scientific question — but one that the current architecture is mechanistically capable of addressing (gene embeddings are already trained for these organisms; only the condition vector changes). T7 will design this new evaluation paradigm.

---

## 4. Cumulative RMSE Trajectory

| Phase | Change | Val RMSE |
|---|---|---|
| S2 null baseline (global mean) | — | ~0.550 |
| T1 (multihot binary encoding) | T1-DEC-004 | ~0.513 |
| T2 (early concat + ShallowMLP) | T2-DEC-001 | ~0.514 |
| T3 (2-layer ResidualMLP, h=512) | T3-DEC-001 | ~0.506 |
| T4 (MSE, raw targets, 8ep) | T4-DEC-001/002 — no change | ~0.502 |
| T5-A (adapter_1024_proj) | T5-DEC-001 | ~0.499 |
| T5-D (adapter variants) | no_winner — no change | ~0.500 |
| T6-A (fingerprints) | no_winner — no change | ~0.499 |

Total improvement from null: −0.051 RMSE (9.3% relative). All improvement after T3 is sub-threshold under the S2-locked bars, confirming T4-C's diagnosis that the bottleneck is representation quality and (per T7-prep) the evaluation paradigm.

---

## 5. Languages, Tools, and Infrastructure

**Python (primary):** All model code (`src/models/`, `src/data/`, `src/evaluation/`, `src/train/`), Hydra configs, data preprocessing. Core dependencies: PyTorch, NumPy, SciPy, Pandas, Hydra-core, RDKit.

**YAML (Hydra configs):** Every experiment is a `configs/experiment/T{N}-{X}_*.yaml` file. These are data, not code — they compose defaults from `configs/model/`, `configs/train/`, `configs/data/` groups and override specific parameters per arm.

**SQLite (`feba.db`):** The raw fitness database. Queried via `pandas.read_sql_query`. 27.4M rows × multiple columns including `orgId`, `locusId`, `expName`, `fit`, `t` (t-statistic), `nGenerations`, `cor12` (replicate correlation). `nGenerations` is missing for 16/48 organisms including the two largest (Btheta, DvH), precluding its use as a feature.

**Excel (v4 workbook):** `data/media_composition_v4.xlsx`, sheet `Media_Components_ML`. The canonical source of truth for media chemistry. Versions v1–v3 explicitly out of scope.

**Parquet:** All derived canonical artifacts. Key files: `data/derived/canonical/v0/fitness_experiment_long.parquet`, `experiments.parquet` (7,552 × 51 columns), `media_master.parquet`, `media_components_long.parquet`.

**JSON (manifests):** Every run emits `run_manifest.json` validated against `data_contract/schemas/run_manifest_v1.schema.json`. Required fields: git SHA, data checksums, split ID, preprocessing artifact ID, seed, RMSE, MAE, Spearman, `scored_rowset_hash`, `unknown_category_rate`.

**PyTorch:** Model training. Key modules: `nn.EmbeddingBag` (chemistry multihot), `nn.Sequential` + `nn.Linear` + `nn.LayerNorm` + `nn.ReLU` + `nn.Dropout` (MLP), custom `ResBlock`. Loss: `nn.MSELoss`. Optimizer: Adam (lr=1e-3, weight_decay=1e-4). Batch size: 8192.

**RDKit:** Chemical fingerprint computation (T6-A, T7+ chemistry pipeline). Morgan/RDKit/MACCS fingerprints from SMILES strings fetched from PubChem.

**Shell / Bash:** Data pipeline orchestration (SQLite queries, parquet exports, checksum generation). No non-trivial logic in shell; all computation is Python.

**No R:** All statistical analysis (bootstrap CIs, Spearman computation, IQR diagnostics) is in Python (`scipy.stats`, `numpy`). R is not present in this repository.

---

## 6. Governance Rules

**Denominator parity:** Model and baseline scored on the exact same row set. `scored_rowset_hash` in the manifest enforces this.

**Train-only preprocessing:** Vocabulary and scalers fit exclusively on training rows. Val/test unseen categories map to `<UNK>`. Unknown-category rate logged in every manifest.

**H-BASE-01 gate:** Model must beat `embedding_NN` on `multi_org_balanced` on both RMSE and MAE. The additive baseline collapses to global mean under cold-start organism holdout.

**Co-primary metrics:** RMSE and MAE must both improve above threshold for any promotion. Metric conflicts (as in T4-A Huber) correctly result in no_winner.

**No tier retroaction:** Once a tier is closed, no subsequent tier may change that tier's locked choice without a formal superseding decision record (e.g., T1-DEC-004 supersedes T1-DEC-003).

**Multi-seed reproducibility:** All experiments run with seeds {0, 1, 2}. Bootstrap CIs must be disjoint for promotion. High seed variance (e.g., T3-B 1024-width arm: std 0.0017) is correctly penalized under the Pareto-stability criterion.

**Scope claim (L7, locked):** "Given a gene and a growth medium and applied stressor chemistry drawn from a known chemistry vocabulary, our model predicts conditional gene essentiality — including for organisms not seen during training, and conditions structured differently from those the gene appeared in during training." We do NOT claim generalization to novel chemistry. T6-A's fingerprint experiment confirms this boundary is meaningful: the model improves with identity-based (multihot) features, not substructure-based (fingerprint) features, in the current evaluation regime.

---

*Textbook generated 2026-05-22. Covers S0–S5 (all stages complete), T1 through T6 (all with decision records), and T7-prep diagnostics. T7 (within-organism cross-condition ranking) not yet started.*
