# Data

## Directory structure

```
data/
  corpus/   Static pipeline fixtures (git-tracked) — handbook text, reference corpus
  splits/   Input CSVs (gitignored) — regenerate with scripts/prepare_splits.py
  runs/     Per-row prediction outputs and sidecar metadata (gitignored)
  cache/    Retrieval cache and vector store manifest (gitignored)
```

---

## `data/splits/` — Input splits

Produced by `scripts/prepare_splits.py` (see main README for split rationale).

| File | Rows | Role |
| ---- | ---- | ---- |
| `test.csv` | 2 200 | Held out — zero contact until paper submission |
| `val.csv` | 2 000 | Evaluate finished systems only; never iterate on |
| `dev.csv` | 19 049 | Free development pool |
| `dev_tune.csv` | 150 | Fixed tuning subset of dev — used for experiment iterations |
| `dev_holdout.csv` | 18 899 | Remainder of dev after dev_tune extraction |
| `scratch.csv` | 100 | Tiny subset of dev for quick scaffolding runs |

---

## `data/runs/` — Prediction outputs (per-row)

When `experiments/run_rag_triage.py` runs, it writes results here — one
versioned CSV with every patient row plus the model's prediction appended,
and a companion JSON sidecar recording the run configuration.

File naming: `<input_stem>_<mode>_triage_results_<YYYYMMDD_HHMMSS>.csv`

The sidecar `.json` contains: model, mode (`rag`/`llm`), top_k,
retrieval_backend, context_chars, git hash, timestamp, token counts,
cost_usd, and pricing_source.

These are the **raw prediction artifacts**. They answer: *"what did the model
predict for each patient?"* Use `experiments/eval_triage.py` to compute
aggregate metrics from them.

Also contains `retrieval_inspection_*.jsonl` files written by
`scripts/inspect_retrievals.py`.

---

## `data/cache/` — Cached retrieval data

| File | Purpose |
| ---- | ------- |
| `retrieval_cache.parquet` | Pre-computed retrieval results (avoids repeated BQ/FAISS calls per row) |
| `vector_store_manifest.json` | Row-count baseline saved by `scripts/validate_retrieval.py` |

---

## Relationship to `experiments/results/`

| | `data/runs/` | `experiments/results/` |
| --- | --- | --- |
| **Granularity** | One row per patient | One row per experiment run |
| **Question answered** | What did the model predict for each patient? | How well did it do overall? |
| **Example file** | `scratch_rag_triage_results_<timestamp>.csv` | `experiment_log.csv` |
| **Written by** | `experiments/run_rag_triage.py` | `experiments/eval_triage.py` |

The flow is: `run_rag_triage.py` writes predictions to `data/runs/` →
`eval_triage.py` reads from `data/runs/` (auto-detects latest file),
computes metrics, and appends a summary row to
`experiments/results/experiment_log.csv`.
