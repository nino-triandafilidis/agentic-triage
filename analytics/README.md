# PubMed Content Analytics

> **Disclaimer:** The analyses in this section are AI-driven and exploratory in nature. Findings reflect the characteristics of the PMC Open Access corpus and heuristic classification outputs

Before building the full RAG pipeline we ran a three-stage analysis to understand what the PMC Open Access vector store actually contains and how relevant it is for ED triage decisions. Scripts live in `analytics/`; all outputs land in `analytics/results/` which is gitignored.

## Stage 1 — Schema discovery (`01_schema_discovery.py`)

Queries `INFORMATION_SCHEMA.COLUMNS` to enumerate the exact columns available in `bigquery-public-data.pmc_open_access_commercial.articles`, counts total rows, spot-samples five rows to observe data shape, and measures non-null rates per column across a 10 k-row sample.

Key finding: the table has **no `journal_title` or `pub_year` columns**. Both must be extracted from `article_citation` (format: `"Journal Name. YYYY Mon DD; Vol:Page"`). All columns are non-null except `author` (~0.4% null).

## Stage 2 — Content landscape (`02_content_landscape.py`)

Five aggregation queries over the full 2.31 M-row table, using hash-based sampling (`MOD(ABS(FARM_FINGERPRINT(pmc_id)), N) < K`) wherever a full scan would exceed BQ's memory slot:


| Query                 | Method                                                                               | Result                            |
| --------------------- | ------------------------------------------------------------------------------------ | --------------------------------- |
| Journal distribution  | Full GROUP BY on `REGEXP_EXTRACT(article_citation, …)`                               | Top 50 journals by article count  |
| Year distribution     | Full GROUP BY on extracted year; filtered to 1990–2026 to remove regex noise         | Publication volume by year        |
| Article length        | Full GROUP BY on `LENGTH(article_text)` into five buckets                            | Length distribution               |
| License distribution  | Full GROUP BY on `license`                                                           | License breakdown                 |
| ED keyword prevalence | `COUNTIF(LOWER(article_text) LIKE '%keyword%')` on a ~4% hash sample (~100 k rows) | Hit rate for 19 ED-relevant terms |


**Key findings:**

- **2,311,736 articles** total
- Top journals are broad open-access venues (PLoS One 5.9%, Sci Rep 4.0%, Cureus 2.2%) — not ED-specialised.
- Article bulk is in the 20k–100k character range (full papers, not abstracts).
- ED keyword hit rates in a random 100 k-article sample: `trauma` 11%, `stroke` 9%, `emergency department` 3.3%, `triage` 1.1%, `emergency severity index` 0.02%.

The 19 keywords were selected heuristically by LLM and grouped into four categories:


| Category                | Keywords                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| ED setting / workflow   | `emergency department`, `emergency room`, `triage`, `emergency severity index`, `acute care`, `clinical decision support` |
| Severity / monitoring   | `critical care`, `intensive care`, `vital signs`, `glasgow coma`                                                          |
| Common chief complaints | `chest pain`, `dyspnea`, `abdominal pain`, `trauma`                                                                       |
| High-acuity diagnoses   | `sepsis`, `stroke`, `myocardial infarction`, `cardiac arrest`, `respiratory failure`                                      |


## Stage 3 — Triage relevance via vector search (`03_triage_relevance.py`)

Constructs a free-text query from each case's HPI and patient demographics (`build_query_from_case`, first 500 chars of HPI + 200 chars of patient info), and issues a `VECTOR_SEARCH` call against the PMC table using the `pubmed.textembed` BigQuery ML model (Google `text-embedding-005`). Samples from `MIMIC-IV-Ext-Dev.csv` (rows `df[2200:]` from the creation script) — the eval CSV is never touched.

Default run: 20 dev cases × top-10 articles = 200 retrieved articles.

**Topic classification methodology.** Each retrieved article is classified by `classify_snippet_heuristic()` — keyword matching against the **first 500 characters** of article text across 12 predefined categories (emergency medicine, critical care, cardiology, pulmonology, neurology, infectious disease, surgery, oncology, general medicine, pediatrics, pharmacology, basic science). Classification is heuristic and non-exclusive: a single article can match multiple categories. Topic percentages in the results are computed by exploding multi-tag rows and normalising by total articles retrieved, not by unique articles.

**Clinical relevance definition.** Two tiers are defined explicitly in the script:

- **Directly ED-relevant**: article matched `emergency_medicine` or `critical_care` tags
- **Clinically relevant**: article matched any clinical tag — the above plus cardiology, pulmonology, neurology, infectious disease, surgery, general medicine, pediatrics, or pharmacology
- **Basic science / other**: no clinical tag matched (includes `basic_science` tag and untagged articles)

**Key findings** *(20 dev cases × top-10 articles = 200 retrieved articles; `random_state=42`):*

- Mean closest distance: **0.75**; mean average distance: **0.78**; mean farthest: **0.80** — moderate but not tight semantic alignment.
- Topic breakdown (non-exclusive tags, normalised over 200 articles): unclassified 26.5%, neurology 18%, surgery 17%, emergency medicine 14.5%, cardiology 12%, general medicine 9.5%, infectious disease & oncology 8.5% each, pulmonology 7.5%, critical care 6%.
- **71% of retrieved articles are clinically relevant** (any clinical domain); **20% are directly ED-relevant** (emergency medicine or critical care); 29% are basic science or otherwise off-topic.

## Running the analytics locally

```bash
# All scripts accept --project to override GCP project detection
python analytics/01_schema_discovery.py
python analytics/02_content_landscape.py --out analytics/results
python analytics/03_triage_relevance.py --sample-size 20 --top-k 10 --out analytics/results
```

No `gcloud` CLI is required. Auth is resolved from `~/.config/gcloud/application_default_credentials.json` (set via `gcloud auth application-default login`). `analytics/_gcp.py` provides the shared BigQuery client used by all three scripts.
