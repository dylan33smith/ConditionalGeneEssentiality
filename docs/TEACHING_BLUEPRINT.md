# MISSION
Generate a comprehensive teaching document named `PROJECT_TEXTBOOK.md`. This document must serve as a rigorous cross-domain bridge. It must explain the intuition of advanced machine learning architectures clearly enough for a biologist to grasp, while simultaneously explaining the biological constraints and realities rigorously enough for a computer scientist. Write for an incoming PhD student sitting exactly at this intersection.

# CONTEXT & SCOPE
Scan the entire codebase, commit history, and any available logs or scripts across all programming languages used in this repository. 
Pay specific attention to:
1. The structural "Tiers" of the project.
2. Formulated hypotheses regarding the biological data or the models.
3. Specific bioinformatics methodologies implemented (e.g., metabolic modeling, flux balance analysis, Tn-seq data processing, or genomic language model configurations).
4. The strategic pivots. You must reflect the most current reality of the research, not just the outdated data contracts. Specifically, document the original assumption regarding cross-org as the primary task, explain the empirical discovery that cross-org Spearman is at noise level, and detail our pivot to the within-org cross-condition framing (the T7-prep findings).

# REQUIRED DOCUMENT STRUCTURE
The final `PROJECT_TEXTBOOK.md` must contain the following sections:

## 1. Executive Summary & Glossary
- A high-level overview of the project's ultimate objective and the current state of the research.
- A "Term Dictionary": List every major domain-specific term, algorithm, or metric used in the code. For each, provide a conceptual analogy followed by a rigorous technical definition, and explicitly state *why* we chose it over alternatives.

## 2. Project Architecture (The Tier System)
- Explain the logic behind how the project is divided into Tiers. What is the overarching progression from Tier 1 to the final Tier?

## 3. Tier-by-Tier Breakdown
For *every single tier* discovered in the project, create a dedicated subsection containing:
- **The Hypotheses:** What exact assumptions about the data or model were we testing?
- **The Methods & Languages:** What scripts, tools, and languages were used to test this? How does the code actually work under the hood?
- **The Dead-Ends & Negative Results:** Document failures honestly and thoroughly. Treat sub-threshold results (e.g., the failures of FiLM, z-scoring, ESM-C bypass, T6-A fingerprints) as critical findings. Explain *why* we tried them, exactly how they failed, and what architectural pivot resulted from that failure.
- **The Validated Results:** What were the concrete, successful outcomes of the experiments in this tier?

# QUALITY GATES
- **Multi-Language Reality:** Do not assume Python is the only language. Ensure you accurately document the logic of any shell scripts, R scripts, C++, or other languages present in the repository.
- **Information Density:** Optimize for high information density and precise technical definitions. Avoid repetitive filler. The document should read like a highly detailed paper supplementary.
- **Mathematical Grounding:** If a method is used (like FBA or sequence alignment), you must explain *how* that specific algorithm operates mathematically or logically before explaining how it was applied in our code.