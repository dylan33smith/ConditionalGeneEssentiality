# Pruned → learning index

The ranking-branch cleanup deleted code that was either dead or that "didn't work"
as a final approach. **No knowledge was lost** — every result is recorded in the
decision ledger (`research_log/decisions/**`) and the narrative
`research_log/SCIENTIFIC_SYNTHESIS.md`, and the code itself remains in git history
(and on the `refactor` branch). This table maps each deleted component to where
its learning lives, so a deleted module is one hop from its result.

| Pruned component | What it was | Where its learning lives |
|---|---|---|
| `src/experiments/stage0-5` | S0-S5 governance/setup (reproducibility, data characterization, evaluation trust, split lock, feature contract, training recipe) | `research_log/decisions/stage*/S*-DEC-*`; SYNTHESIS §1-2 |
| `src/experiments/tier1` | encoder/representation sweep (T-regime) | `decisions/tier1/T1-DEC-001..007`; locked: 425-d multihot |
| `src/experiments/tier2` | fusion topology (concat/two-tower/FiLM) | `decisions/tier2/T2-DEC-001`; locked: early-concat MLP |
| `src/experiments/tier3` | capacity (depth/width/residual) | `decisions/tier3/T3-DEC-001`; locked: 2-layer width-512 (ResidualBlock kept in `src/ranking/models.py`) |
| `src/experiments/tier4` | optimization + loss + target norm (T-regime) | `decisions/tier4/T4-DEC-001..002`; locked: MSE, raw targets |
| `src/experiments/tier5` | embedding adapter (frozen-PLM question) | `decisions/tier5/T5-DEC-001`; AdapterResidualMLP kept in `src/ranking/models.py` |
| `src/experiments/tier6` | chemistry fingerprints vs multihot | folded into R1; multihot retained; loader kept in `src/ranking/data/fingerprints.py` |
| `src/experiments/diagnostics/t7_prep` | T7-prep diagnostic suite (motivated the R-regime fork) | SYNTHESIS §1-2; the within-gene Spearman ≈0.045 finding |
| `src/experiments/rhybrid` | R-HYBRID-A/B model+kNN hybrids (failed to beat the gate) | `decisions/rhybrid/R-HYBRID-DEC-001..002`; SYNTHESIS §7. Reusable kNN/retrieval helpers kept in `src/ranking/eval/harness.py` |
| `src/evaluation/{additive,nn,null}_baseline, metrics, reporting` | T-regime baselines + figure helpers | `decisions/stage2/S2-DEC-001`; ranking baselines live in `src/ranking/eval/harness.py` |
| `src/{train,training,models,domain}` | early scaffolding / T-regime trainer + architectures + contracts | superseded by `src/ranking/{models,train,pipeline,runner}.py` |
| `src/data/datasets/{build_model_dataset,dataset_audits}`, `ingestion/load_embeddings` | T-regime data utilities | data provenance kept in `src/data/{ingestion,preprocessing}`; ranking split in `src/data/datasets/build_ranking_split.py` |
| `configs/experiment/T*.yaml`, `configs/stage/s*.yaml` | T-regime experiment/stage configs | the corresponding decision-ledger entries |
| `docs/{REFACTORPLAN,RPLAN,PROJECT_TEXTBOOK_T6,TEACHING_BLUEPRINT}.md`, `research_log/STAGE_LEARNINGS.md` | T-regime planning/teaching docs | ranking design folded into `ARCHITECTURE.md`; learnings in the ledger + SYNTHESIS |
| root `s5_*.parquet`, `testing.ipynb` | stray run outputs / scratch notebook | n/a (transient) |

**Recovery.** `git log --all -- <path>` to see history; the full pre-cleanup tree
is the `refactor` branch (and the tip commit before Step 5 on this branch).
