# Prompted Opinion Summarization with GPT-3.5

This repository contains the artifacts for the paper "Prompted Opinion Summarization with GPT-3.5", ACL (Findings) 2023. 

## Directory Structure

Scripts are found under `scripts/`. Specifically,
- Spreadsheets and python scripts for the Human Evaluation part are in `scripts/human-eval`.
- Code for computing the ROUGE scores in in `scripts/rouge/`.
- Functions for prompting GPT-3 for summarization/split-and-rephrase and calculating metrics (those based on entailment as well as others) are in `scripts/fewsum/` and `scripts/space/`.

The `saved-data/` directory contains the original reviews, produced summaries and entailment scores (+ aggregates where relevant) for both SPACE and FewSum. All of these are saved both as pickle files and as text files (the latter are restricted to those used for human evaluation in some cases).

The metric values obtained are stored in `observations/`.
