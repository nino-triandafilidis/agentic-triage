# Experiments

> **Status:** Metrics are implemented in `experiments/eval_triage.py` and validated across 18 experiment conditions on dev_tune (150 rows). See `experiment_results.md` for full results.

## Triage Evaluation Metrics

Triage prediction is an ordinal classification task: the model must assign each patient one of five ESI levels (1 = most acute, 5 = least acute). The choice of metrics reflects two properties of this task that plain accuracy misses.

**Plain accuracy** counts a prediction as either right or wrong. For ordinal labels, this is too coarse: predicting ESI 3 when the true label is ESI 2 is meaningfully less wrong than predicting ESI 5. It also rewards models that exploit class imbalance — ESI 3 is the most common level, so a model that always predicts 3 scores non-trivially without learning anything.

### Metrics used

**Exact accuracy** — fraction of patients where the predicted ESI exactly matches the true ESI. Included as a baseline reference; interpreted with caution given the class imbalance.

**±1 accuracy** — fraction of patients where the prediction is within one ESI level of the true value in either direction. Symmetric: treats over-triage (predicting higher acuity) and under-triage (predicting lower acuity) equally. Useful as a permissive ceiling — if this is low, the model is making multi-level errors.

**Range accuracy** (Gaber et al.) — fraction of patients where the prediction is either exact or over-triaged by at most one level (i.e., predicted ESI is equal to or one level more acute than truth). Unlike ±1 accuracy, this is asymmetric: it accepts mild over-triage (clinically safer) but rejects any under-triage. Formula: `0 <= (truth - predicted) <= 1`. This is the clinically motivated metric — under-triage (e.g., ESI 4 for an ESI 2 patient) is a safety event, while slight over-triage triggers a review but rarely causes harm.

**MAE (mean absolute error)** — average absolute difference between predicted and true ESI across all patients. On a 1–5 scale, MAE = 0 is perfect; MAE ≈ 1 means off by one level on average; MAE ≥ 1.5 is approaching random. Simple and interpretable.

**Quadratic weighted kappa (κ)** — the primary metric. Measures agreement between predictions and true labels after correcting for the agreement that would arise by chance given the observed class distribution. The quadratic weighting penalises larger ordinal mistakes more severely: a two-level miss counts four times as much as a one-level miss. This makes it sensitive to clinically dangerous confusions (e.g., ESI 2 predicted as ESI 4) while forgiving small disagreements.

Rough interpretation: κ < 0.2 is barely better than chance; κ 0.4–0.6 is moderate agreement; κ > 0.8 is near-perfect. Published LLM triage benchmarks report κ in the 0.3–0.5 range for general-purpose models.

### Running evaluation

```bash
# Default: evaluates triage_RAG column against triage ground truth in scratch results
.venv/bin/python experiments/eval_triage.py

# Custom file or columns
.venv/bin/python experiments/eval_triage.py \
    --input data/scratch_rag_triage_results.csv \
    --pred triage_RAG \
    --target triage

# Compare multiple prediction columns at once
.venv/bin/python experiments/eval_triage.py --pred triage_RAG triage_baseline
```

## Query Strategy Sweep

Compare different query construction strategies for RAG retrieval. Each strategy builds the retrieval query differently from the patient case data:

- **concat** (baseline): concatenates HPI + patient_info + initial_vitals
- **hpi_only** (ablation): uses only HPI
- **rewrite**: LLM-rewrites the concatenated query into a focused PubMed search query

```bash
# Quick test: concat + hpi_only on 5 rows
.venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/scratch.csv --n-rows 5 \
    --strategies concat hpi_only --top-k 5 \
    --output-prefix sweep_test

# Full sweep with rewrite + split cost caps
.venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/scratch.csv --n-rows 20 \
    --strategies concat hpi_only rewrite \
    --rewrite-model gemini-2.5-flash \
    --max-generation-cost-usd 0.50 --max-rewrite-cost-usd 0.10 \
    --output-prefix sweep_scratch --per-row-diagnostics

# BM25 sparse retrieval backend
RETRIEVAL_BACKEND=bm25 .venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/dev_tune.csv --n-rows 150 \
    --strategies concat --output-prefix E15_bm25

# Hybrid (FAISS + BM25 RRF) backend
RETRIEVAL_BACKEND=hybrid .venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/dev_tune.csv --n-rows 150 \
    --strategies concat --output-prefix E16_hybrid

# Evaluate one strategy output
.venv/bin/python experiments/eval_triage.py \
    --input data/runs/sweep_scratch_concat_*.csv --pred triage_RAG

# With ESI Handbook as system-level context prefix (~99k chars)
.venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/scratch.csv --n-rows 10 \
    --strategies concat --handbook-prefix \
    --output-prefix sweep_handbook
```

### ESI Handbook prefix

`--handbook-prefix` prepends the full ESI Handbook v5 text (~99k chars, 26 pages) as system-level context in the generation prompt. The handbook includes the complete ESI algorithm with decision points A-D, vital sign thresholds, resource counting rules, and worked examples. This is loaded from the pre-extracted plaintext file (see `scripts/extract_esi_handbook_text.py`). Use `--handbook-path` to override the default path.

**Outputs** (all in `data/runs/`):
- `{prefix}_{strategy}_{ts}.csv` — predictions with `triage_RAG`, `triage_query_agent`, `triage_query_text`, `triage_query_hash`, `n_articles_retrieved`
- `{prefix}_{strategy}_{ts}.json` — sidecar manifest with run metadata, split costs, and `execution_events` trace
- `{prefix}_{strategy}_{ts}.diagnostics.jsonl` — per-row diagnostics (with `--per-row-diagnostics`)
- `{prefix}_summary_{ts}.csv` — aggregate comparison across strategies (when >1 strategy)
