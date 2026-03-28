# MedLLM

> **Local retrieval — zero per-call cost (default backend)**
>
> `retrieval.py` supports multiple backends via `RETRIEVAL_BACKEND` env var:
>
> | Backend | Description | Per-call cost |
> | ------- | ----------- | ------------- |
> | `faiss` (default) | Dense: local FAISS IVF index + Vertex AI embedding | ~$0.00 (search ~48ms + embedding ~1-2s) |
> | `bm25` | Sparse: local BM25S index, keyword-based | $0.00 (no API calls) |
> | `hybrid` | FAISS + BM25 fused via Reciprocal Rank Fusion (RRF) | ~$0.00 (same as FAISS) |
> | `bq` (legacy) | BigQuery VECTOR_SEARCH | ~$0.10/call — avoid |

## New user setup

All large artifacts (embeddings, FAISS index, SQLite articles database) are already built and live on the shared Google Drive. **You do not need to re-run the export or build scripts.**

**Step 1 — Point `FAISS_STORE_DIR` at the shared drive.**

Add this to your shell profile (`~/.zshrc` or `~/.bash_profile`), replacing the path with wherever Google Drive mounts on your machine:

```bash
export FAISS_STORE_DIR="/path/to/your/Shared drives/CS224n/faiss_store"
```

**Step 2 — Copy the artifacts to local disk (recommended, takes a few minutes).**

The pipeline auto-copies files from the shared drive to `~/medllm/faiss` on first run, but reading through the Google Drive FUSE layer is slow (~12-15 min for 14.5 GB). Copy them directly now for a much faster transfer (local disk to local disk):

```bash
mkdir -p ~/medllm/faiss
cp "$FAISS_STORE_DIR/index.faiss" "$FAISS_STORE_DIR/pmc_articles.db" "$FAISS_STORE_DIR/pmc_ids.parquet" "$FAISS_STORE_DIR/manifest.json" ~/medllm/faiss/
```

Once copied, the pipeline reads only from `~/medllm/faiss` and skips the shared drive on all subsequent runs (checksum-validated on startup).

**Step 3 — Verify:**

```bash
.venv/bin/python -m pytest tests/test_retrieval.py -v
```

## README index

| Location | Contents |
| -------- | -------- |
| [README.md](README.md) | Setup, usage, vector store, tests, data splits |
| [data/README.md](data/README.md) | Data splits, per-row prediction outputs, relationship to experiment logs |
| [analytics/README.md](analytics/README.md) | PubMed content analytics findings (3-stage analysis) |
| [experiments/README.md](experiments/README.md) | Triage evaluation metrics, experiment log format |
| [scripts/README.md](scripts/README.md) | Script reference, FAISS/BM25 index building, validation results (Feb 2026) |
| [third_part_code/README.md](third_part_code/README.md) | medLLMbenchmark preprocessing notes, `df` vs `df_small` analysis |

---

RAG-augmented medical LLM evaluation pipeline. Uses BigQuery vector search over PubMed literature to ground LLM predictions on the subset of [medLLMbenchmark](https://github.com/BIMSBbioinfo/medLLMbenchmark) tasks (i.e., triage and diagnosis), using MIMIC-IV emergency department data.

## Repository structure

```
src/
├── config.py              # GCP credentials, constants, one-time setup()
├── llm/                   # Provider-agnostic LLM interface (Anthropic, Google, Kimi)
│   ├── base.py            # Abstract base class for LLM providers
│   ├── registry.py        # Provider registry and factory
│   ├── types.py           # Shared types (EmbedResponse, etc.)
│   └── providers/         # Concrete provider implementations
└── rag/
    ├── retrieval.py        # PubMed retrieval — FAISS, BM25, hybrid RRF, or BQ
    ├── text_cleaning.py    # Text preprocessing for BM25 tokenization
    ├── triage_core.py      # Core triage prediction logic and prompt building
    ├── agentic_pipeline.py # Multi-step agentic RAG pipeline
    ├── generation.py       # LLM response generation with retrieved context
    ├── query_agents.py     # Query construction agents (rewrite, HPI-only, etc.)
    ├── esi_handbook.py     # ESI Handbook text loader for system-level context
    └── pipeline.py         # End-to-end: query → retrieve → generate
data/                      # See data/README.md
├── README.md
├── corpus/                # Static pipeline fixtures (git-tracked)
│   ├── esi_v4_fewshot_bank.md     # 5 gold-standard ESI cases for few-shot prompts
│   └── esi_v4_practice_cases.md   # ESI v4 practice cases reference
├── splits/                # Input CSVs (gitignored) — regenerate with scripts/prepare_splits.py
│   ├── test.csv           # 2 200-row — zero contact until paper submission
│   ├── val.csv            # 2 000-row — evaluate finished systems, do not iterate
│   ├── dev.csv            # 19 049-row free development pool
│   ├── dev_tune.csv       # 150-row tuning subset of dev (experiment iterations)
│   ├── dev_holdout.csv    # Remainder of dev after dev_tune extraction
│   └── scratch.csv        # 100-row subset of dev for quick scaffolding runs
├── runs/                  # Per-row prediction outputs and sidecar metadata (gitignored)
└── cache/                 # Retrieval cache and vector store manifest (gitignored)
analytics/                 # See analytics/README.md
├── README.md              # PubMed content analytics findings
├── _gcp.py                # Shared BigQuery client (reads ADC credentials)
├── 01_schema_discovery.py
├── 02_content_landscape.py
└── 03_triage_relevance.py
experiments/               # See experiments/README.md
├── README.md              # Triage evaluation metrics, experiment log format
├── query_strategy_sweep.py # Unified experiment runner (RAG + LLM-only)
├── eval_triage.py         # Computes metrics, appends summary to results/experiment_log.csv
├── tracking.py            # Experiment tracking utilities
├── run_rag_triage.py      # Legacy single-strategy inference runner
├── measure_retrieval_cost.py
└── results/
    └── experiment_log.csv # One row per experiment run (aggregate metrics)
scripts/                   # See scripts/README.md
├── README.md              # Script reference, FAISS validation results, BQ validation findings
├── prepare_splits.py      # Single entry point: copies + splits all benchmark data
├── build_bm25_index.py    # Build BM25S sparse index from SQLite corpus
├── build_faiss_index.py   # Build IVF4096 FAISS index from exported embeddings
├── export_faiss_data.py   # One-time BQ export → local .npy/.parquet/.db (~$0.90)
├── validate_faiss_vs_bq.py # Recall@10 sweep across nprobe values (~$0.50)
├── build_esi_handbook_index.py  # Build FAISS index from ESI Handbook PDF
├── extract_esi_handbook_text.py # Extract ESI Handbook PDF to plaintext
├── inspect_retrievals.py  # Inspect retrieval results for debugging
├── create_vector_store.py # One-time BQ table provisioning (legacy)
├── validate_retrieval.py  # Post-provisioning integrity checks (legacy)
└── populate_retrieval_cache.py  # Pre-compute retrieval cache (legacy)
docs/
└── paper_notes.md         # Permanent findings record linked to GitHub issues
third_part_code/           # See third_part_code/README.md
├── README.md              # Preprocessing notes for medLLMbenchmark
├── pubmed-rag/            # Upstream: google/pubmed-rag (Git subtree)
└── medLLMbenchmark/       # Upstream: BIMSBbioinfo/medLLMbenchmark (Git subtree)
rag_scaffolding.ipynb      # Demo notebook for the RAG pipeline
```

## Prerequisites

- Python 3.11+
- A Google Cloud project with billing enabled
- `gcloud` CLI ([install](https://cloud.google.com/sdk/docs/install)) — only needed for initial `gcloud auth application-default login`; not required at runtime (Python clients read ADC directly)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Authenticate with Google Cloud

**Google Colab** — handled automatically. The code calls `google.colab.auth.authenticate_user()` on import.

**Local** — run once in your terminal:

```bash
gcloud auth login
gcloud auth application-default login
```

This saves credentials to `~/.config/gcloud/application_default_credentials.json`. They persist across kernel restarts; re-run only if you switch accounts or the token expires.

### 3. Set your project

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

The project ID is auto-detected from your gcloud config at runtime — no hardcoding needed.

### 4. Environment variables

All machine-specific config lives in a single `.env.local` file (gitignored). Copy the template and fill in your values:

```bash
cp .env.example .env.local
```

This file holds API keys (Anthropic, Moonshot) and, if you need to re-run dataset creation, MIMIC data paths. See `.env.example` for all variables and descriptions.

`.env.local` is loaded automatically via `python-dotenv` by both `src/config.py` and the dataset creation script — no manual sourcing needed.

Most collaborators only need API keys. MIMIC paths are only required for the one-time dataset creation step (see `third_part_code/README.md`).

### 5. IAM roles

Your account needs these roles on the project (grant once via [IAM console](https://console.cloud.google.com/iam-admin/iam) or CLI):

- **Editor** (`roles/editor`)
- **BigQuery Admin** (`roles/bigquery.admin`)

### 6. Run setup

```python
from src.config import setup
setup()
```

This enables the required APIs (`aiplatform`, `bigquery`, `bigqueryconnection`), creates the BigQuery dataset and embedding model. Safe to call repeatedly — every step is idempotent.

## Usage

```python
from src.config import setup
from src.rag import rag_pipeline

setup()

# Free-text query
answer = rag_pipeline(query="What are the causes of acute chest pain with diaphoresis?")

# From medLLMbenchmark fields
answer = rag_pipeline(
    hpi="Patient presents with acute chest pain radiating to the left arm.",
    patient_info="Gender: M, Race: White, Age: 62",
)
```

## Vector store

### Local FAISS (default)

The default backend uses a local FAISS IVF4096 index over 2.18M PubMed embeddings. All data lives on disk — no BQ call per query. Validated recall@10 = 1.00 at nprobe=64, latency ~48ms per search. See [scripts/README.md](scripts/README.md) for validation results.

### BigQuery tables (legacy / used for BQ backend)

Two owned BQ tables exist for the `RETRIEVAL_BACKEND=bq` path:

| Table                   | Purpose                                                                        | Size                 |
| ----------------------- | ------------------------------------------------------------------------------ | -------------------- |
| `pubmed.pmc_embeddings` | Drives vector search. Has the IVF index. Contains embedding vectors + IDs.     | ~14 GB               |
| `pubmed.pmc_articles`   | Document store. Fetched by ID after search. Contains text, citation, title.    | ~40–60 GB compressed |

These were provisioned once (~$0.82) via `scripts/create_vector_store.py`. New users do not need to re-run this — the FAISS artifacts on the shared drive were exported from these tables. See [scripts/README.md](scripts/README.md) for full BQ validation findings.

## Tests

Two tiers: unit tests that run anywhere (no GCP credentials required), and integration tests that hit real BigQuery.

### Unit tests — no credentials, instant, zero cost

```bash
.venv/bin/python -m pytest tests/ -v
```

These use `unittest.mock` to replace the BigQuery client and FAISS index, so they run offline. `src/config.py` uses lazy credential resolution (credentials are fetched on first access, not at import time), and `tests/conftest.py` pre-seeds fake credentials so no test triggers `google.auth.default()`. Key regression guards:

- `test_distance_is_cosine` / `test_sorted_by_distance` — FAISS backend returns correct cosine distances in ascending order
- `test_same_schema_as_bq` — FAISS and BQ backends return identical column sets
- `test_handles_empty_results` — graceful handling of FAISS returning no results
- `test_uses_owned_tables` — BQ backend queries owned tables, not the public PMC table
- `test_query_is_parameterized` — BQ query text goes through `ScalarQueryParameter`, not f-string interpolation (SQL injection guard)

### Integration tests — requires BQ credentials, ~$0.01

```bash
INTEGRATION=1 .venv/bin/python -m pytest tests/ -v -m integration
```

Skipped by default (`INTEGRATION` env var not set). These issue real VECTOR_SEARCH calls and validate that the owned tables are accessible and return non-empty results. Run once after provisioning or after any change to `retrieval.py` query structure.


## Dev / Val / Scratch Split Strategy

The original paper acknowledges prompt experimentation in passing but provides no formal split. Our reproduction makes the split explicit and reproducible: any prompt engineering or hyperparameter decisions are traceable to dev-set observations and never touch the test set.

### Split structure

All splits live under `data/splits/` with short, consistent names. They are gitignored (large files) and regenerated by running `scripts/prepare_splits.py`, which is the single entry point for all data setup.

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

**val.csv — evaluate, don't iterate.** You may run inference on val to compare two finished RAG systems. What you must not do is look at val failures and adjust prompts or hyperparameters in response — that makes val a training signal and the comparison meaningless. Use dev for all iterative work.

`test.csv` is copied from `third_part_code/medLLMbenchmark/`. `val.csv` and `dev.csv` are produced by splitting the upstream dev source (21 049 rows): val is a random 2 000-row sample (seed 42); dev is the remainder. `scratch.csv` is a 100-row subset of dev (seed 42). Total unique rows is 23 249.

### Rationale: no stratification

Plain random sampling is used (no `stratify=` argument). Triage-level proportions in the dev pool are stable (≤1.7 pp deviation between dev and val, verified empirically), so forced stratification adds no statistical benefit. ESI 4 is too rare (<0.5%) to meaningfully balance via stratification regardless.

### Running the split

```bash
python scripts/prepare_splits.py          # copies test/dev, writes val and scratch
python scripts/prepare_splits.py --help   # show --val-size / --scratch-size / --seed options
```

All four CSVs are gitignored. Re-run the script to regenerate them; the fixed seed guarantees identical output.


