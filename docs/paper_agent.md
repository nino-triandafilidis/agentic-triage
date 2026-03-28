# Paper Notes — Agent Guide

This file defines expectations for how `paper_notes.md` should be curated.
Read this before editing `paper_notes.md`.

## Target output

`paper_notes.md` feeds into a **CS224N final project report** (6–8 pages,
LaTeX, NLP research paper style). It is not the paper itself — it is a
structured scratchpad of findings, decisions, and draft-ready prose fragments
that will be copy-edited into the final LaTeX document.

## Report structure (from CS224N Project Report Instructions)

Required sections, in order:

1. **Abstract** (<300 words) — problem, difficulty, contribution, main findings
2. **Introduction** — problem, why hard, how current methods fail, key ideas + results
3. **Related Work** — research context, how papers relate, why our work is a next step
4. **Approach** — architectures, algorithms, baselines; equations/figures; credit original work
5. **Experiments**
   - Data: dataset description (can reference prior work for shared details)
   - Evaluation method: metrics + rationale
   - Experimental details: model configs, hyperparams
   - Results: quantitative, tables/plots, compare against baselines
6. **Analysis** — qualitative evaluation, error analysis, ablation, confusion matrices, subgroup performance
7. **Conclusion** — main findings, achievements, limitations, future work
8. **Team contributions** (required for multi-person teams)
9. **References** (BibTeX)
10. **Appendix** (optional, doesn't count toward page limit)

## Scope constraint

The broader project has multiple workstreams (as described in the proposal):
RAG-augmented triage, RL fine-tuning, agentic orchestration. **This report
covers only the RAG/agentic triage workstream** (Nino's contribution). Other
workstreams will be written up by their respective team members. The paper
notes should focus exclusively on this workstream.

## Space budget

6–8 pages total, shared across the full team. Nino's workstream gets roughly
3–4 pages of body text. This means:

- **Data section**: ~0.25 pages. Reference Gaber et al. for shared setup (MIMIC-IV-ED,
  ESI scale, clinical user setting). Only note differences from their approach.
- **Approach section**: ~1 page. Strategy descriptions, tool-use architecture, case bank.
- **Experiments + Results**: ~1.5 pages. Leaderboard table, key comparisons.
- **Analysis**: ~1 page. Extended metrics argument, confusion matrices, Gaber comparison.

## Writing principles

- **Concise over comprehensive.** If it can be a citation instead of a paragraph, cite.
- **Draft-ready fragments preferred.** Write prose that can be pasted into LaTeX with
  minimal editing, not just bullet points.
- **Data-grounded.** Every claim should trace to a specific experiment run or metric.
  Include the config name and number so the claim can be verified.
- **Audience: CS224N student.** Not a clinician, not an ML researcher. A smart grad
  student who knows transformers but not ED triage.

## What belongs in paper_notes.md vs elsewhere

| Content | Where |
|---|---|
| Draft prose for paper sections | paper_notes.md |
| Experiment results (numbers, verdicts) | experiment_results.md |
| Full per-experiment writeups | experiment_results_archive.md |
| Implementation decisions, engineering notes | code comments / commit messages |
| Bibliography entries for the paper | paper_notes.md § Bibliography |
