# MedLLM

RAG-augmented medical LLM evaluation pipeline for emergency department triage. Retrieves PubMed literature via local vector search and uses LLM reasoning to predict ESI (Emergency Severity Index) triage levels on [medLLMbenchmark](https://github.com/BIMSBbioinfo/medLLMbenchmark) cases from MIMIC-IV emergency department data.

> **Zero per-call retrieval cost.** The default backend uses a local FAISS index over 2.18M PubMed embeddings — no cloud API calls per query.

| Retrieval backend | Description | Per-query cost |
| --- | --- | --- |
| `faiss` (default) | Dense: local FAISS IVF index + Vertex AI query embedding | ~$0.00 (search ~48ms + embedding ~1-2s) |
| `bm25` | Sparse: local BM25S index, keyword-based | $0.00 (no API calls) |
| `hybrid` | Dense + sparse fused via Reciprocal Rank Fusion | ~$0.00 (same as FAISS) |
| `bq` (legacy) | BigQuery VECTOR_SEARCH | ~$0.10/query — avoid |

---

## Setup overview

**A Google Cloud project with billing is required.** The pipeline calls Vertex AI APIs at multiple stages of every run — query embedding (`text-embedding-005`), LLM generation (Gemini), and optional steps like query rewriting and reranking. All Google API calls authenticate via Application Default Credentials (ADC).

Setup has two phases. **Section 1** builds the retrieval artifacts (FAISS index, article database) from BigQuery — a one-time step. **Section 2** configures the pipeline to run inference.

If you already have the artifacts (e.g. from a prior build), skip directly to [Section 2](#2-run-the-pipeline).

### Google API calls per inference run

| Stage | API | When | Per-row cost |
| --- | --- | --- | --- |
| Query embedding | Vertex AI Embedding (`text-embedding-005`) | Every query (`faiss`/`hybrid` backends) | ~$0.00 |
| Generation | Vertex AI Generative (`gemini-*`) or Anthropic (`claude-*`) | Every row | ~$0.001-0.003 |
| Query rewriting | Vertex AI Generative | If `--strategies rewrite` | ~$0.0001-0.001 |
| Reranking | Vertex AI Generative or Anthropic | If `--rerank` | ~$0.0005 |
| Boundary review | Vertex AI Generative or Anthropic | If `--boundary-review` and ESI in [2,3] | ~$0.0005 |

Using Anthropic (Claude) for generation requires an `ANTHROPIC_API_KEY` in addition to GCP credentials (embedding still uses Vertex AI). The only fully offline path is `RETRIEVAL_BACKEND=bm25` with a non-Google LLM provider.

---

## 1. Build the retrieval index (one-time)

This section exports PubMed embeddings from BigQuery and builds the local FAISS index. You only need to do this once.

**Prerequisites:**
- Python 3.11+
- A Google Cloud project with billing enabled
- `gcloud` CLI ([install](https://cloud.google.com/sdk/docs/install))
- ~25 GB free disk space
- Budget: ~$1.40 total (~$0.90 export + ~$0.50 optional validation)

### 1.1 Install dependencies

```bash
git clone <this-repo-url>
cd medllm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Your account needs **Editor** (`roles/editor`) and **BigQuery Admin** (`roles/bigquery.admin`) on the project. Grant via [IAM console](https://console.cloud.google.com/iam-admin/iam) or CLI.

### 1.3 Provision BigQuery tables

```python
from src.config import setup
setup()  # idempotent — safe to re-run
```

This enables required GCP APIs (`aiplatform`, `bigquery`, `bigqueryconnection`) and creates the BigQuery dataset, embedding model, and owned tables (`pubmed.pmc_embeddings`, `pubmed.pmc_articles`). Cost: ~$0.82.

### 1.4 Export embeddings to local files

```bash
.venv/bin/python scripts/export_faiss_data.py
```

Exports embeddings (`.npy`), article IDs (`.parquet`), and article text (`.db`) from BigQuery to `~/medllm/faiss/`. Cost: ~$0.90.

| Output file | Size | Description |
| --- | --- | --- |
| `pmc_embeddings.npy` | ~6.5 GB | float32 vectors, shape (N, 768) |
| `pmc_ids.parquet` | ~50 MB | PMC ID + PMID per row (same order as embeddings) |
| `pmc_articles.db` | ~17 GB | SQLite: article text, citations, indexed on `pmc_id` |
| `manifest.json` | <1 KB | Checksums and metadata |

### 1.5 Build the FAISS index

```bash
.venv/bin/python scripts/build_faiss_index.py
```

Builds an IVF4096 index from the exported embeddings. Runs locally, no API cost. Outputs `index.faiss` (~6.3 GB) to the same directory.

### 1.6 Build the BM25 index (optional)

```bash
.venv/bin/python scripts/build_bm25_index.py
```

Only needed if you want to use the `bm25` or `hybrid` retrieval backend. Runs locally (~63 min), outputs to `~/medllm/bm25s/`.

### 1.7 Validate (optional)

```bash
.venv/bin/python scripts/validate_faiss_vs_bq.py   # ~$0.50
```

Sweeps nprobe values and reports recall@10 against BigQuery ground truth. See [scripts/README.md](scripts/README.md) for validation results (recall@10 = 1.00 at nprobe=64).

---

## 2. Run the pipeline

This section assumes you have the retrieval artifacts in `~/medllm/faiss/` (built in Section 1 or obtained elsewhere) and that GCP authentication is already configured (Section 1.2).

### 2.1 Configure API keys (if using Anthropic)

If you plan to use Claude models for generation, add your Anthropic key:

```bash
cp .env.example .env.local
# Edit .env.local:
ANTHROPIC_API_KEY=sk-ant-...
```

Gemini models use the same ADC credentials from Section 1.2 — no additional key needed.

`.env.local` is gitignored and loaded automatically via `python-dotenv` — no manual sourcing needed.

### 2.2 Configure artifact paths (if non-default)

By default the pipeline reads from `~/medllm/faiss/`. If your artifacts are elsewhere, set the path in `.env.local`:

```bash
FAISS_LOCAL_DIR=/your/custom/path
```

See `.env.example` for all available path variables.

### 2.3 Prepare evaluation data

```bash
.venv/bin/python scripts/prepare_splits.py
```

This copies benchmark CSVs from `third_part_code/medLLMbenchmark/` and generates the train/dev/val/test splits in `data/splits/`. Fixed seed (42) guarantees reproducible output. See [Data splits](#data-splits) below for details.

### 2.4 Verify installation

```bash
.venv/bin/python -m pytest tests/test_retrieval.py -v
```

All tests pass without cloud credentials — they use mocked backends.

### 2.5 Run inference

```python
from src.rag import rag_pipeline

# Free-text query
answer = rag_pipeline(query="What are the causes of acute chest pain with diaphoresis?")

# From medLLMbenchmark patient fields
answer = rag_pipeline(
    hpi="Patient presents with acute chest pain radiating to the left arm.",
    patient_info="Gender: M, Race: White, Age: 62",
)
```

### 2.6 Run experiments

```bash
# Compare query strategies on a small subset
.venv/bin/python experiments/query_strategy_sweep.py \
    --input data/splits/scratch.csv \
    --strategies concat hpi_only \
    --output-prefix sweep_test

# Evaluate predictions
.venv/bin/python experiments/eval_triage.py \
    --input data/runs/sweep_test_*.csv \
    --pred triage_RAG \
    --target triage
```

---

## Configuration

### Retrieval backend

Set via environment variable (default: `faiss`):

```bash
export RETRIEVAL_BACKEND=faiss   # dense retrieval (default)
export RETRIEVAL_BACKEND=bm25    # sparse retrieval
export RETRIEVAL_BACKEND=hybrid  # dense + sparse via RRF
```

### Environment variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes (if using Claude) | — | Anthropic API key |
| `MOONSHOT_API_KEY` | No | — | Moonshot/Kimi API key |
| `RETRIEVAL_BACKEND` | No | `faiss` | `faiss`, `bm25`, or `hybrid` |
| `FAISS_LOCAL_DIR` | No | `~/medllm/faiss` | Path to FAISS artifacts |
| `FAISS_STORE_DIR` | No | `~/medllm/faiss` | Where build scripts write artifacts |
| `BM25_LOCAL_DIR` | No | `~/medllm/bm25s` | Path to BM25 index |
| `FAISS_NPROBE` | No | `64` | IVF nprobe (higher = more recall, slower) |
| `ESI_HANDBOOK_PATH` | No | `data/corpus/esi_handbook_clean.txt` | ESI Handbook text for prompts |

---

## Repository structure

```
src/
├── config.py              # Credentials, constants, one-time setup()
├── llm/                   # Provider-agnostic LLM interface (Anthropic, Google, Kimi)
│   ├── base.py            # Abstract base class
│   ├── registry.py        # Provider routing by model ID prefix
│   ├── types.py           # Shared types (LLMResponse, GenerationConfig, etc.)
│   └── providers/         # Concrete implementations (anthropic.py, google.py, kimi.py)
└── rag/
    ├── pipeline.py         # Entry point: query -> retrieve -> generate
    ├── agentic_pipeline.py # Multi-step agentic RAG orchestrator
    ├── retrieval.py        # FAISS, BM25, hybrid RRF, or BQ retrieval
    ├── generation.py       # LLM response generation with retrieved context
    ├── query_agents.py     # Query construction agents (rewrite, HPI-only, etc.)
    ├── triage_core.py      # ESI prompt templates and triage parsing
    ├── text_cleaning.py    # Text preprocessing for BM25 tokenization
    ├── esi_handbook.py     # ESI Handbook text loader
    └── case_bank.py        # Few-shot case storage
data/                      # See data/README.md
├── corpus/                # Static pipeline fixtures (git-tracked)
├── splits/                # Evaluation CSVs (gitignored — regenerate with scripts/prepare_splits.py)
├── runs/                  # Prediction outputs (gitignored)
└── cache/                 # Retrieval cache (gitignored)
scripts/                   # See scripts/README.md
├── prepare_splits.py      # Generate train/dev/val/test splits
├── export_faiss_data.py   # BQ export -> local .npy/.parquet/.db (~$0.90)
├── build_faiss_index.py   # Build IVF4096 FAISS index
├── build_bm25_index.py    # Build BM25S sparse index
└── validate_faiss_vs_bq.py # Recall@10 validation
experiments/               # See experiments/README.md
├── query_strategy_sweep.py # Unified experiment runner
├── eval_triage.py         # Metric computation
└── results/               # Experiment log CSV
tests/                     # Unit + integration tests
third_part_code/           # Upstream dependencies (git subtrees)
├── medLLMbenchmark/       # Benchmark data + creation scripts
└── pubmed-rag/            # Google's PubMed RAG reference
```

## Vector store

### Local FAISS (default)

The default backend uses a local FAISS IVF4096 index over 2.18M PubMed embeddings. All data lives on disk — no BQ call per query. Validated recall@10 = 1.00 at nprobe=64, latency ~48ms per search. See [scripts/README.md](scripts/README.md) for validation results.

### BigQuery tables (legacy)

Two owned BQ tables exist for the `RETRIEVAL_BACKEND=bq` path:

| Table                   | Purpose                                                                        | Size                 |
| ----------------------- | ------------------------------------------------------------------------------ | -------------------- |
| `pubmed.pmc_embeddings` | Drives vector search. Has the IVF index. Contains embedding vectors + IDs.     | ~14 GB               |
| `pubmed.pmc_articles`   | Document store. Fetched by ID after search. Contains text, citation, title.    | ~40-60 GB compressed |

These are provisioned by `scripts/create_vector_store.py` (~$0.82) during Section 1.3. The FAISS artifacts are exported from these tables. See [scripts/README.md](scripts/README.md) for full BQ validation findings.

## Tests

Two tiers: unit tests that run anywhere (no GCP credentials required), and integration tests that hit real BigQuery.

### Unit tests — no credentials, instant, zero cost

```bash
.venv/bin/python -m pytest tests/ -v
```

These use `unittest.mock` to replace the BigQuery client and FAISS index, so they run offline. Key regression guards:

- `test_distance_is_cosine` / `test_sorted_by_distance` — FAISS backend returns correct cosine distances in ascending order
- `test_same_schema_as_bq` — FAISS and BQ backends return identical column sets
- `test_handles_empty_results` — graceful handling of FAISS returning no results
- `test_uses_owned_tables` — BQ backend queries owned tables, not the public PMC table
- `test_query_is_parameterized` — BQ query text goes through `ScalarQueryParameter`, not f-string interpolation (SQL injection guard)

### Integration tests — requires BQ credentials, ~$0.01

```bash
INTEGRATION=1 .venv/bin/python -m pytest tests/ -v -m integration
```

Skipped by default (`INTEGRATION` env var not set). Run once after provisioning or after any change to `retrieval.py` query structure.

## Data splits

The original paper acknowledges prompt experimentation in passing but provides no formal split. This reproduction makes the split explicit and reproducible: any prompt engineering or hyperparameter decisions are traceable to dev-set observations and never touch the test set.

### Split structure

All splits live under `data/splits/` with short, consistent names. They are gitignored and regenerated by running `scripts/prepare_splits.py`.

```
23 249 unique rows
├── data/splits/test.csv         2 200  — held out for final paper reporting
├── data/splits/val.csv          2 000  — one-shot comparison between finished RAG variants
└── data/splits/dev.csv         19 049  — free development pool (source minus val)
        ├── data/splits/dev_tune.csv       150  — fixed tuning subset for experiment iterations
        ├── data/splits/dev_holdout.csv  18 899  — remainder of dev after dev_tune
        └── data/splits/scratch.csv        100  — subset of dev; immediate scaffolding runs
```

**test.csv — zero contact until paper submission.** Never load this file during development. Any inspection, even checking distributions, is data leakage that invalidates the final numbers.

**val.csv — evaluate, don't iterate.** You may run inference on val to compare two finished RAG systems. Do not look at val failures and adjust prompts or hyperparameters in response. Use dev for all iterative work.

`test.csv` is copied from `third_part_code/medLLMbenchmark/`. `val.csv` and `dev.csv` are produced by splitting the upstream dev source (21 049 rows): val is a random 2 000-row sample (seed 42); dev is the remainder. `scratch.csv` is a 100-row subset of dev (seed 42).

### Rationale: no stratification

Plain random sampling is used (no `stratify=` argument). Triage-level proportions in the dev pool are stable (<=1.7 pp deviation between dev and val, verified empirically), so forced stratification adds no statistical benefit.

### Running the split

```bash
python scripts/prepare_splits.py          # copies test/dev, writes val and scratch
python scripts/prepare_splits.py --help   # show --val-size / --scratch-size / --seed options
```

All CSVs are gitignored. Re-run the script to regenerate them; the fixed seed guarantees identical output.

## Sub-READMEs

| Location | Contents |
| --- | --- |
| [data/README.md](data/README.md) | Data splits, per-row prediction outputs |
| [analytics/README.md](analytics/README.md) | PubMed content analytics findings (3-stage analysis) |
| [experiments/README.md](experiments/README.md) | Triage evaluation metrics, experiment log format |
| [scripts/README.md](scripts/README.md) | Script reference, FAISS/BM25 index building, validation results |
| [third_part_code/README.md](third_part_code/README.md) | medLLMbenchmark preprocessing notes |
