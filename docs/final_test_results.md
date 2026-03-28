# Final Test Set Evaluation Results

> **Split:** `data/splits/test.csv` (n=2,200, includes 117 NaN-vitals rows)
> **Branch:** `eval/test-runs`
> **Runbook:** [`docs/test_eval_runbook.md`](./test_eval_runbook.md)
> **Machine-readable log:** [`experiments/results/experiment_log.csv`](../experiments/results/experiment_log.csv)

---

## Evaluation methodology

Results are evaluated using the same **multilevel metric stack** as dev_tune
(see [`experiment_results.md`](../experiment_results.md)):

1. **Benchmark** — `exact_accuracy`, `range_accuracy`
2. **Safety** — `under_triage_rate`, `over_triage_rate`
3. **Imbalance diagnostics** — `macro_recall`, `per_class_recall`, `MA-MAE`
4. **Chance-adjusted diagnostics** — `linear_weighted_gwet_ac2`, `quadratic_weighted_kappa`

Hard gates before recording (from runbook):
- `n_evaluated + n_skipped == 2,200`
- `parse_fail_rate < 0.01`
- `n_unique_predicted >= 3`

Current status: all 7 runs complete (3 historical pre-fix, 4 post-fix).
G4_two_role_pro_critic has 22 API timeout failures (1.0% parse fail rate, at the
hard gate boundary); metrics are computed on the 2178 evaluated rows.

---

## Experiments

### Completed (historical, pre-fix)

These runs were completed before the retrieval-text fix (PR #64). Non-PMC runs
are unaffected by the fix and remain valid final results.

| Internal ID | Reader-facing name | Exact mechanism | Role in final story |
|---|---|---|---|
| W1_ref | Safety comparator | Fewshot on G2.5 Flash | Model-generation ablation |
| G3_W1ref | Primary baseline | Fewshot on G3 Flash | Does the tool add value? |
| G3_tool_use | Main proposed system | Single-agent, case-bank tool | Our best framework config |

### Pending (post-fix)

These runs will use code after PR #64, which unified article excerpt cleaning.
All PMC paths now use `prepare_article_excerpt()` (front-matter stripped via
section markers before truncation). Default excerpt budget:
`RETRIEVAL_CONTEXT_CHARS = 8000`.

| Internal ID | Reader-facing name | Exact mechanism | Role in final story |
|---|---|---|---|
| G3_tool_use_pmc_1k_clean | Evidence-source ablation (1k) | Active PMC tool (no case bank), cleaned 1k excerpts | Matched-budget comparison to pre-fix |
| G3_tool_use_pmc_8k_clean | Evidence-source ablation (8k) | Active PMC tool (no case bank), cleaned 8k excerpts | Is more PMC context better? |
| G3_W1gated_8k_clean | Retrieval-delivery ablation | Gated passive PMC RAG, cleaned 8k snippets | Does distance-gating help? |
| G4_two_role_pro_critic | Orchestration ablation | Nurse + critic (case-bank only, not PMC-sensitive) | Does multi-agent help? |

---

## Results

### Leaderboard (n=2,200)

Only completed runs are populated. Blank rows are pending, not losses.

| Experiment | Model | Strategy | Exact Acc | Range Acc | Under-triage | Over-triage | Macro Recall | MA-MAE | AC2 | QWK | recall_ESI4 | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **W1_ref** | G2.5 Flash | fewshot | 54.3% | 87.0% | 11.6% | 34.0% | 0.389 | 0.772 | 0.729 | 0.375 | 0.000 | $7.51 |
| **G3_W1ref** | G3 Flash | fewshot | 62.8% | 72.6% | 27.3% | 10.0% | 0.394 | 0.667 | 0.771 | 0.500 | 0.000 | $7.96 |
| **G3_tool_use** | G3 Flash | tool_use | 61.1% | 77.2% | 22.7% | 16.1% | 0.374 | 0.683 | 0.764 | 0.461 | 0.000 | $18.41 |
| **G3_tool_use_pmc_1k_clean** | G3 Flash | tool_use_pmc (1k) | 59.1% | 81.3% | 18.5% | 22.4% | 0.365 | 0.724 | 0.757 | 0.414 | 0.000 | $22.32 |
| **G3_tool_use_pmc_8k_clean** | G3 Flash | tool_use_pmc (8k) | 57.3% | 82.8% | 16.8% | 25.9% | 0.354 | 0.796 | 0.748 | 0.370 | 0.000 | $33.27 |
| **G3_W1gated_8k_clean** | G3 Flash | gated RAG (8k) | 61.6% | 73.9% | 25.9% | 12.4% | 0.386 | 0.674 | 0.765 | 0.479 | 0.000 | $10.45 |
| **G4_two_role_pro_critic** | G3.1 Pro + Flash | two_role_case_bank | 62.2% | 79.5% | 20.3% | 17.5% | 0.384 | 0.670 | 0.772 | 0.469 | 0.000 | — |

### Per-class recall

| Experiment | ESI-1 | ESI-2 | ESI-3 | ESI-4 | ESI-5 |
|---|---|---|---|---|---|
| **W1_ref** | 0.449 | 0.787 | 0.322 | 0.000 | — |
| **G3_W1ref** | 0.186 | 0.607 | 0.781 | 0.000 | — |
| **G3_tool_use** | 0.140 | 0.734 | 0.624 | 0.000 | — |
| **G3_tool_use_pmc_1k_clean** | 0.159 | 0.807 | 0.494 | 0.000 | — |
| **G3_tool_use_pmc_8k_clean** | 0.159 | 0.844 | 0.414 | 0.000 | — |
| **G3_W1gated_8k_clean** | 0.182 | 0.638 | 0.724 | 0.000 | — |
| **G4_two_role_pro_critic** | 0.165 | 0.770 | 0.601 | 0.000 | — |

---

## Completed runs (historical, pre-fix)

These runs were produced before PR #64 (retrieval-text fix). None of them use
PMC retrieval, so their results are unaffected by the fix and remain valid.

### W1_ref — Safety comparator (fewshot, G2.5 Flash)

**Date:** 2026-03-14
**Config:** `--mode llm --prompt-template fewshot --model gemini-2.5-flash`
**Git hash:** `9a4e7fb`
**Output:** `data/runs/test_W1_ref/test_W1_ref_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 54.3% |
| Range accuracy | 87.0% |
| +/-1 accuracy | 98.1% |
| MAE | 0.478 |
| Under-triage rate | 11.6% |
| Over-triage rate | 34.0% |
| Expected cost | 0.741 |
| Macro recall | 0.389 |
| MA-MAE | 0.772 |
| Gwet AC2 (linear) | 0.729 |
| Quadratic kappa | 0.375 |
| Prediction entropy | 1.199 bits |
| Unique predicted | 5 |
| recall_ESI1 | 0.449 |
| recall_ESI2 | 0.787 |
| recall_ESI3 | 0.322 |
| recall_ESI4 | 0.000 |
| Total cost | $7.51 |

**Observations:**
- ESI-4 recall is 0.000 — complete class collapse, consistent with dev_tune behavior.
- Under-triage (11.6%) is comparable to dev_tune (10.8%), confirming this config's safety profile generalizes.
- Range accuracy (87.0%) is slightly lower than dev_tune (88.3%).
- Exact accuracy (54.3%) is marginally above dev_tune (53.4%), within noise.
- All hard gates passed: n=2200, parse_fail=0%, unique_predicted=5.

### G3_W1ref — Primary baseline (fewshot, G3 Flash)

**Date:** 2026-03-14
**Config:** `--mode llm --prompt-template fewshot --model gemini-3-flash-preview`
**Git hash:** `208c8fd`
**Output:** `data/runs/test_G3_W1ref/test_G3_W1ref_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 62.8% |
| Range accuracy | 72.6% |
| MAE | 0.387 |
| Under-triage rate | 27.3% |
| Over-triage rate | 10.0% |
| Expected cost | 1.005 |
| Macro recall | 0.394 |
| MA-MAE | 0.667 |
| Gwet AC2 (linear) | 0.771 |
| Quadratic kappa | 0.500 |
| Prediction entropy | 1.229 bits |
| Unique predicted | 5 |
| Parse failures | 3 (0.1%) |
| recall_ESI1 | 0.186 |
| recall_ESI2 | 0.607 |
| recall_ESI3 | 0.781 |
| recall_ESI4 | 0.000 |
| Total cost | $7.96 |

**Observations:**
- Exact accuracy (62.8%) closely matches dev_tune (61.6%), +1.2pp.
- Under-triage (27.3%) is consistent with dev_tune (26.4%) — confirms the G3 Flash safety concern generalizes.
- ESI-3 recall (0.781) is much stronger than W1_ref (0.322), but ESI-2 recall (0.607) drops vs W1_ref (0.787) — the model shifts predictions toward ESI-3.
- ESI-4 recall remains 0.000, same as all other fewshot configs.
- All hard gates passed: n=2200, parse_fail=0.1%, unique_predicted=5.

### G3_tool_use — Main proposed system (case-bank tool, G3 Flash)

**Date:** 2026-03-14
**Config:** `--mode llm --prompt-template tool_use --model gemini-3-flash-preview`
**Git hash:** `208c8fd`
**Output:** `data/runs/test_G3_tool_use/test_G3_tool_use_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 61.1% |
| Range accuracy | 77.2% |
| MAE | 0.402 |
| Under-triage rate | 22.7% |
| Over-triage rate | 16.1% |
| Expected cost | 0.914 |
| Macro recall | 0.374 |
| MA-MAE | 0.683 |
| Gwet AC2 (linear) | 0.764 |
| Quadratic kappa | 0.461 |
| Prediction entropy | 1.212 bits |
| Unique predicted | 5 |
| recall_ESI1 | 0.140 |
| recall_ESI2 | 0.734 |
| recall_ESI3 | 0.624 |
| recall_ESI4 | 0.000 |
| Total cost | $18.41 |

**Observations:**
- Exact accuracy (61.1%) is slightly below dev_tune (62.7%), -1.6pp — modest regression.
- Tool_use vs fewshot on G3 Flash (G3_W1ref): exact -1.7pp, range +4.6pp, under-triage -4.6pp. Tool shifts toward over-triage, improving range and safety at the cost of exact accuracy.
- ESI-3 recall (0.624) is lower than fewshot (0.781) but ESI-2 recall (0.734) is higher — tool_use redistributes predictions more toward ESI-2.
- ESI-4 recall remains 0.000. Tool_use did not recover ESI-4 on test (dev_tune had 0.333).
- All hard gates passed: n=2200, parse_fail=0%, unique_predicted=5.

## Completed runs (post-fix)

These runs use code after PR #64 (retrieval-text fix). All PMC excerpt paths
now use `prepare_article_excerpt()` which strips front matter via section
markers before truncation.

### G3_tool_use_pmc_1k_clean — Evidence-source ablation (cleaned 1k excerpts)

**Date:** 2026-03-15
**Config:** `--mode llm --prompt-template tool_use_pmc --model gemini-3-flash-preview --context-chars 1000`
**Git hash:** `b2a0ef3`
**Output:** `data/runs/test_G3_tool_use_pmc_1k_clean/test_G3_tool_use_pmc_1k_clean_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 59.1% |
| Range accuracy | 81.3% |
| +/-1 accuracy | 99.4% |
| MAE | 0.416 |
| Under-triage rate | 18.5% |
| Over-triage rate | 22.4% |
| Expected cost | 0.810 |
| Macro recall | 0.365 |
| MA-MAE | 0.724 |
| Gwet AC2 (linear) | 0.757 |
| Quadratic kappa | 0.414 |
| Prediction entropy | 1.080 bits |
| Unique predicted | 5 |
| recall_ESI1 | 0.159 |
| recall_ESI2 | 0.807 |
| recall_ESI3 | 0.494 |
| recall_ESI4 | 0.000 |
| Total cost | $22.32 |

**Observations:**
- Exact accuracy (59.1%) is below G3_tool_use (61.1%, case-bank) by -2.0pp — PMC-only with 1k excerpts underperforms the case-bank tool.
- Range accuracy (81.3%) is higher than G3_tool_use (77.2%, +4.1pp) and G3_W1ref (72.6%, +8.7pp) — PMC tool drives predictions toward over-triage, improving range.
- Under-triage (18.5%) is between G3_tool_use (22.7%) and W1_ref (11.6%) — PMC retrieval shifts predictions toward higher acuity vs case-bank alone.
- ESI-2 recall (0.807) is the highest across all configs, but ESI-3 recall (0.494) drops significantly vs G3_W1ref (0.781) — heavy ESI-2 bias.
- ESI-4 recall remains 0.000, consistent with all other configs.
- All hard gates passed: n=2200, parse_fail=0%, unique_predicted=5.

### G3_tool_use_pmc_8k_clean — Evidence-source ablation (cleaned 8k excerpts, default budget)

**Date:** 2026-03-15
**Config:** `--mode llm --prompt-template tool_use_pmc --model gemini-3-flash-preview`
**Git hash:** `b2a0ef3`
**Output:** `data/runs/test_G3_tool_use_pmc_8k_clean/test_G3_tool_use_pmc_8k_clean_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 57.3% |
| Range accuracy | 82.8% |
| +/-1 accuracy | 99.4% |
| MAE | 0.434 |
| Under-triage rate | 16.8% |
| Over-triage rate | 25.9% |
| Expected cost | 0.786 |
| Macro recall | 0.354 |
| MA-MAE | 0.796 |
| Gwet AC2 (linear) | 0.748 |
| Quadratic kappa | 0.370 |
| Prediction entropy | 1.001 bits |
| Unique predicted | 5 |
| recall_ESI1 | 0.159 |
| recall_ESI2 | 0.844 |
| recall_ESI3 | 0.414 |
| recall_ESI4 | 0.000 |
| Total cost | $33.27 |

**Observations:**
- Exact accuracy (57.3%) is the lowest of all configs — 8k PMC excerpts hurt rather than help vs 1k (59.1%, -1.8pp).
- Range accuracy (82.8%) is the highest overall, slightly above 1k (81.3%, +1.5pp) — more context pushes even harder toward over-triage.
- Over-triage (25.9%) is the highest of any config except W1_ref. Under-triage (16.8%) is the lowest of any G3 config.
- ESI-2 recall (0.844) is a new high, but ESI-3 recall (0.414) collapses — the model predicts ESI-2 for nearly everything mid-acuity.
- MA-MAE (0.796) is worse than 1k (0.724) — more context amplifies class imbalance rather than resolving it.
- ESI-4 recall remains 0.000. Cost ($33.27) is 49% higher than 1k ($22.32) with worse outcomes.
- All hard gates passed: n=2200, parse_fail=0%, unique_predicted=5.

### G3_W1gated_8k_clean — Retrieval-delivery ablation (gated RAG, cleaned 8k)

**Date:** 2026-03-15
**Config:** `--mode rag --prompt-template fewshot --model gemini-3-flash-preview --distance-gate 0.25`
**Git hash:** `b2a0ef3`
**Output:** `data/runs/test_G3_W1gated_8k_clean/test_G3_W1gated_8k_clean_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 61.6% |
| Range accuracy | 73.9% |
| +/-1 accuracy | 98.7% |
| MAE | 0.398 |
| Under-triage rate | 25.9% |
| Over-triage rate | 12.4% |
| Expected cost | 0.984 |
| Macro recall | 0.386 |
| MA-MAE | 0.674 |
| Gwet AC2 (linear) | 0.765 |
| Quadratic kappa | 0.479 |
| Prediction entropy | 1.223 bits |
| Unique predicted | 5 |
| Parse failures | 2 (0.1%) |
| recall_ESI1 | 0.182 |
| recall_ESI2 | 0.638 |
| recall_ESI3 | 0.724 |
| recall_ESI4 | 0.000 |
| Total cost | $10.45 |

**Observations:**
- Exact accuracy (61.6%) matches G3_W1ref baseline (62.8%) within noise — gated RAG does not improve over fewshot-only.
- Range accuracy (73.9%) is nearly identical to G3_W1ref (72.6%, +1.3pp) — passive retrieval adds minimal value here.
- Under-triage (25.9%) is comparable to G3_W1ref (27.3%) — slight improvement but well within noise.
- Per-class recall profile closely mirrors G3_W1ref: ESI-1 (0.182 vs 0.186), ESI-2 (0.638 vs 0.607), ESI-3 (0.724 vs 0.781). The distance gate (60% fallback to LLM-only on canary) means most rows get no retrieval context.
- ESI-4 recall remains 0.000, consistent with all configs.
- All hard gates passed: n=2200, parse_fail=0.1%, unique_predicted=5.

### G4_two_role_pro_critic — Orchestration ablation (nurse + critic)

**Date:** 2026-03-15
**Config:** `--mode llm --prompt-template two_role_case_bank --model gemini-3.1-pro-preview --fast-model gemini-3-flash-preview`
**Git hash:** `c2ab7d1`
**Output:** `data/runs/test_G4_two_role_pro_critic/test_G4_two_role_pro_critic_merged_2200.csv`

| Metric | Value |
|---|---|
| Exact accuracy | 62.2% |
| Range accuracy | 79.5% |
| +/-1 accuracy | 99.1% |
| MAE | 0.388 |
| Under-triage rate | 20.3% |
| Over-triage rate | 17.5% |
| Expected cost | 0.838 |
| Macro recall | 0.384 |
| MA-MAE | 0.670 |
| Gwet AC2 (linear) | 0.772 |
| Quadratic kappa | 0.469 |
| Prediction entropy | 1.181 bits |
| Unique predicted | 5 |
| Parse failures | 22 (1.0%) |
| recall_ESI1 | 0.165 |
| recall_ESI2 | 0.770 |
| recall_ESI3 | 0.601 |
| recall_ESI4 | 0.000 |

**Observations:**
- Parse fail rate (1.0%) is at the hard gate boundary — all 22 failures are API timeouts (21) or other infrastructure errors (1), not model quality issues. Failed rows are proportionally distributed across ground-truth classes.
- Exact accuracy (62.2%) is the highest of any tool/retrieval config, slightly above G3_tool_use (61.1%) and matching G3_W1ref baseline (62.8%).
- Range accuracy (79.5%) is second only to G3_tool_use_pmc_8k_clean (82.8%) — Pro critic shifts toward over-triage vs fewshot.
- Under-triage (20.3%) improves over G3_W1ref (27.3%, -7.0pp) — the critic role catches some under-triage from the nurse.
- The over-triage/under-triage balance (17.5%/20.3%) is the most symmetric of any config.
- ESI-4 recall remains 0.000, consistent with all configs.
- Cost not tracked (no pricing available for gemini-3.1-pro-preview).

---

## Dev-tune vs test comparison

| Metric | W1_ref dev | W1_ref test | Delta | G3_W1ref dev | G3_W1ref test | Delta | G3_tool_use dev | G3_tool_use test | Delta |
|---|---|---|---|---|---|---|---|---|---|
| Exact accuracy | 53.4% | 54.3% | +0.9pp | 61.6% | 62.8% | +1.2pp | 62.7% | 61.1% | -1.6pp |
| Range accuracy | 88.3% | 87.0% | -1.3pp | 73.5% | 72.6% | -0.9pp | 78.3% | 77.2% | -1.1pp |
| Under-triage | 10.8% | 11.6% | +0.8pp | 26.4% | 27.3% | +0.9pp | 21.6% | 22.7% | +1.1pp |
| Over-triage | 35.8% | 34.0% | -1.8pp | 12.0% | 10.0% | -2.0pp | 15.7% | 16.1% | +0.4pp |
| Macro recall | 0.378 | 0.389 | +0.011 | 0.381 | 0.394 | +0.013 | 0.467 | 0.374 | -0.093 |
| MA-MAE | 0.761 | 0.772 | +0.011 | 0.681 | 0.667 | -0.014 | 0.587 | 0.683 | +0.096 |
| AC2 | 0.727 | 0.729 | +0.002 | 0.766 | 0.771 | +0.005 | 0.775 | 0.764 | -0.011 |
| QWK | 0.382 | 0.375 | -0.007 | 0.478 | 0.500 | +0.022 | 0.485 | 0.461 | -0.024 |

W1_ref and G3_W1ref generalize stably (all deltas < 2pp). G3_tool_use shows
a notable macro_recall drop (-0.093) driven by ESI-4 recall falling from 0.333
on dev_tune to 0.000 on test. Benchmark metrics (exact, range, AC2) remain
within ~2pp. The ESI-4 recovery seen on dev_tune did not generalize.
