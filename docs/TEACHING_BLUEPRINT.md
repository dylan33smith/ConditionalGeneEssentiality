# MISSION
Generate a comprehensive teaching document named `PROJECT_TEXTBOOK.md`. This document must serve as a bridge: it needs to explain the fundamental "why" for someone new to the field, but maintain the rigorous technical depth expected of a PhD researcher.

# CONTEXT & SCOPE
Scan the entire codebase, commit history, and any available logs or scripts across all programming languages used in this repository. 
Pay specific attention to:
1. The structural "Tiers" of the project.
2. Formulated hypotheses regarding the data or the models.
3. Specific bioinformatics methodologies implemented (e.g., metabolic modeling, flux balance analysis, Tn-seq data processing, or genomic language model configurations).
4. The architectural and algorithmic decisions made by the AI during development.

# REQUIRED DOCUMENT STRUCTURE
The final `PROJECT_TEXTBOOK.md` must contain the following sections:

## 1. Executive Summary & Glossary
- A high-level overview of the project's ultimate objective.
- A "Term Dictionary": List every major domain-specific term, algorithm, or metric used in the code. For each, provide a simple analogy, followed by the rigorous technical definition, and exactly *why* we chose to use it here instead of an alternative.

## 2. Project Architecture (The Tier System)
- Explain the logic behind how the project is divided into Tiers. What is the overarching progression from Tier 1 to the final Tier?

## 3. Tier-by-Tier Breakdown
For *every single tier* discovered in the project, create a dedicated subsection containing:
- **The Hypotheses:** What exact assumptions about the data or model were we testing?
- **The Methods & Languages:** What scripts, tools, and languages were used to test this? How does the code actually work under the hood?
- **AI Design Decisions:** Why did the AI structure the code or pipelines this way? What dead-ends or alternatives were avoided?
- **The Results:** What were the concrete outcomes of the experiments in this tier? 

# QUALITY GATES
- Do not assume Python is the only language; document the logic of any shell scripts, R scripts, C++, or other languages present.
- If a method is used (like FBA or sequence alignment), you must explain *how* that specific algorithm operates mathematically or logically before explaining how it was applied in our code.