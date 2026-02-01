# aHFR-TokenSHAP  
**Adaptive Hierarchically Feature-Restricted TokenSHAP for Binary Classification with Large Language Models**

This repository provides a **reproducible demonstration** of *feature-level* attribution for LLM-based **binary phenotype classification** prompts rendered from structured (hierarchical) records—together with the accompanying manuscript and technical note.

**Manuscript (PDF):** `paper_aHFR_TokenSHAP.pdf`  
**Technical note (PDF):** `technical_note_complexity_reduction.pdf`  
**Repository:** `stvsever/PAPER_HFR_TokenSHAP`

---

## Motivation

Large language models (LLMs) are increasingly used as inference engines over **structured phenotypic records** rendered into prompt templates (e.g., clinical risk-factor fields). In this regime, **token-level attribution** methods can over-attribute prompt **scaffolding** (headers, separators, boilerplate instructions) that is necessary for instruction-following but **not** the explanatory object of interest.

This repository focuses on **feature-level explanations**: *which structured features (and which feature domains) drive the model’s binary decision?*

---

## Method in one paragraph

We introduce **aHFR-TokenSHAP**, a task-specific extension of TokenSHAP for **binary classification prompts** in which (i) the value function is the model’s binary decision score defined as **label log-odds** (rather than response-similarity between generated texts), and (ii) Shapley “players” are **template-aligned leaf features** organized by a **pre-specified hierarchy** rather than all prompt tokens. aHFR-TokenSHAP further incorporates an **adaptive, hierarchy-constrained permutation generator**: permutations are constructed via mixed-depth hierarchical frontiers, initialized by a short primary-layer calibration and updated across epochs to concentrate sampling on influential subtrees while preserving Shapley–Shubik marginal-contribution semantics. We validate aHFR-TokenSHAP in a controlled random hierarchical feature-injection experiment with 10 parent domains and 30 leaf features across 100 pseudo-profiles, and compare against (a) an internal baseline (**Integrated Gradients** on the same log-odds score, aggregated over value-only spans) and (b) an external knowledge-prior baseline (**LLM-Select**-style feature-name scoring).

---

## Pseudocode of the aHFR-TokenSHAP algorithm

![Pseudocode of algorithm](https://github.com/stvsever/aHFR_TokenSHAP/blob/main/demonstration_results/visuals/pseudocode_aHFR_TokenSHAP_algorithm.png)

---

## Figure: Example prompt overlay

Figure 1 shows a qualitative example prompt with an overlay of weighted feature-importance scores (Integrated Gradients + aHFR-TokenSHAP-style restricted Shapley's), illustrating increased emphasis on clinically relevant features relative to distractor “word-features”.

![Example prompt overlay](https://github.com/stvsever/PAPER_HFR_TokenSHAP/blob/main/demonstration_results/visuals/example_prompt_overlay.png)

---

## Quickstart

### 1) Create an environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
