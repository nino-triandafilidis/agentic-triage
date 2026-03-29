# Scripts

**Retrieval index pipeline:**

| Script                    | Purpose                                                                | Cost            |
| ------------------------- | ---------------------------------------------------------------------- | --------------- |
| `export_faiss_data.py`    | One-time BQ export: embeddings (.npy), IDs (.parquet), articles (.db) | approximately $0.90 once |
| `build_faiss_index.py`    | Build IVF4096 FAISS index from exported embeddings                     | free (local)    |
| `build_bm25_index.py`    | Build BM25S sparse index from SQLite corpus (~63 min, 6.3 GB output)  | free (local)    |
| `validate_faiss_vs_bq.py` | Recall@10 sweep across nprobe values vs BQ ground truth                | approximately $0.50 once |

**Legacy (BigQuery backend):**

| Script                       | Purpose                                                             | Cost                    |
| ---------------------------- | ------------------------------------------------------------------- | ----------------------- |
| `create_vector_store.py`     | Provisions `pmc_embeddings` + `pmc_articles` BQ tables with IVF index | approximately $0.82 once |
| `validate_retrieval.py`      | Verifies the owned BQ tables match the public PMC source            | approximately $0.35 once |
| `populate_retrieval_cache.py`| Pre-computes BQ retrieval results for a set of queries              | varies                  |

> These legacy scripts are only needed if switching to `RETRIEVAL_BACKEND=bq`. For new users on the FAISS backend, ignore them.

## `validate_retrieval.py`

Run once after provisioning (or after any schema change to the owned tables):

```bash
.venv/bin/python scripts/validate_retrieval.py
```

**What it checks:**

| Check                         | Cost   | What it verifies                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 — Row counts**            | free   | `owned_n == COUNT(DISTINCT pmc_id)` in the public table. QUALIFY dedup keeps exactly one row per article; this must hold exactly. Also reports how many `pmid='0'` placeholder rows were dropped vs. kept as singletons. Saves a baseline to `data/vector_store_manifest.json`; subsequent runs compare `pub_n` against that baseline and warn if the public table has grown (re-provisioning recommended). |
| **2B — Coverage**             | free   | No `pmc_id` is entirely absent from owned. Flags the 23 articles that have two real-PMID rows in the public table (PMID corrections); QUALIFY kept one — not data loss.                                                                                                                                                                                                                                     |
| **2A — Embedding integrity**  | ~$0.14 | Embeddings in owned match the source. Hash-samples ~218 rows (`1-in-10,000`); inner-joins on `(pmc_id, pmid)` and tests `MIN(ML.DISTANCE) < 1e-6` per group.                                                                                                                                                                                                                                                |
| **3 — Retrieval consistency** | ~$0.21 | VECTOR_SEARCH on owned (IVF, `fraction_lists_to_search=0.1`) returns the same top-1 as brute-force search on the public table for 3 canonical ED queries.                                                                                                                                                                                                                                                   |

**Validation status (Feb 2026):** All checks PASS. Check 1: dedup verified (131,869 rows dropped = `pmid='0'` duplicates; 111,324 `pmid='0'` singletons kept). Check 2: 0 corrupted embeddings; 0 articles entirely absent (23 alternate-PMID rows noted — PMID corrections in source, not data loss). Check 3: top-1 match on all 3 canonical ED queries with `fraction_lists_to_search=0.1`. See [validation findings](#validation-findings-feb-2026) below for details.

## `create_vector_store.py`

Idempotent — safe to re-run if a step was interrupted. Prompts for confirmation at each step.

```bash
.venv/bin/python scripts/create_vector_store.py
```

**Ongoing storage cost:** ~$1.10–1.50/month while both tables are active. Tables move to long-term storage pricing (half price) automatically after 90 days without modification.

## `export_faiss_data.py`

Exports BQ data for local FAISS search. Run once; outputs are cached.

```bash
.venv/bin/python scripts/export_faiss_data.py
```

**Outputs** (written to `FAISS_STORE_DIR`, defaults to `~/medllm/faiss/`):
- `pmc_embeddings.npy` (~6.5 GB) — float32 vectors, shape (N, 768)
- `pmc_ids.parquet` (~50 MB) — pmc_id + pmid per row (same order as embeddings)
- `pmc_articles.db` (~17 GB) — SQLite database with `articles` table (pmc_id, pmid, article_text, article_citation). `article_text` stores the first 8000 characters from the owned `pmc_articles` table (`SUBSTR(article_text, 1, 8000)` in `create_vector_store.py`), matching the default `config.RETRIEVAL_CONTEXT_CHARS`. The pipeline cleans retrieved text before prompt insertion, then truncates to the run's `context_chars`. Full article text is only available in the BQ source table. B-tree index on `pmc_id` for O(log N) lookups at ~0 MB steady-state RAM.

At runtime, `_ensure_local_copy()` verifies the required files exist in `FAISS_LOCAL_DIR` (defaults to `~/medllm/faiss/`). If `FAISS_STORE_DIR` differs from `FAISS_LOCAL_DIR`, files are copied automatically. Checksums are validated on startup.

## `build_faiss_index.py`

Builds an IVF4096 FAISS index from exported embeddings.

```bash
.venv/bin/python scripts/build_faiss_index.py
```

**IVF training parameters:**
- `NLIST = 4096` — number of IVF clusters
- `TRAIN_SAMPLE = 256,000` — ~62 points per centroid

The training sample size follows the heuristic of ≥40× nlist for stable centroids (40 × 4096 = 163,840; we use 256k for extra margin). Increasing beyond this gives diminishing returns on centroid quality. To re-tune: adjust `TRAIN_SAMPLE` in the script and re-validate recall with `validate_faiss_vs_bq.py`.

## `validate_faiss_vs_bq.py`

Sweeps nprobe values (32, 64, 128, 256) and reports recall@10, Jaccard similarity, distance correlation, and latency against BQ VECTOR_SEARCH ground truth.

```bash
.venv/bin/python scripts/validate_faiss_vs_bq.py
```

Recommends the lowest nprobe achieving ≥90% recall@10.

**FAISS validation results (Feb 2026):** All nprobe values ≥64 achieve perfect recall. Distance correlations are +1.000 across the board, confirming FAISS and BQ return identical rankings.

| nprobe | recall@10 | jaccard | dist corr | latency (ms) |
| ------ | --------- | ------- | --------- | ------------ |
| 32     | 0.94      | 0.90    | +1.000    | 65           |
| 64     | 1.00      | 1.00    | +1.000    | 48           |
| 128    | 1.00      | 1.00    | +1.000    | 91           |
| 256    | 1.00      | 1.00    | +1.000    | 137          |

**Selected: nprobe=64** — 100% recall at 48ms per query. nprobe=32 achieves 94% recall but misses 1-2 results on some queries. nprobe=128/256 add latency with no recall gain.

## `build_esi_handbook_index.py`

Builds a **standalone** FAISS index from the AHRQ/ENA Emergency Severity Index (ESI) Handbook PDF. Use this to augment or compare with PMC retrieval for triage decision support.

```bash
.venv/bin/python scripts/build_esi_handbook_index.py --pdf PATH [--out DIR]
```

- **--pdf** — Path to the ESI Handbook PDF (required).
- **--out** — Output directory for `index.faiss`, `chunks.json`, `manifest.json` (default: `data/corpus/esi_handbook_index/`).

**Steps:** (1) Parse PDF, drop cover/TOC/blank/acknowledgments and other non–decision-support pages. (2) Chunk retained text (~2000 chars, 400 char overlap). (3) Embed with Vertex AI `text-embedding-005` (same as main corpus). (4) Build FAISS `IndexFlatIP` and write index + chunk metadata.

**Output:** `index.faiss`, `chunks.json` (chunk_id, text, page_start, page_end), `manifest.json` (checksums, model, num_chunks). Cost: Vertex embedding only (no BQ).

## `extract_esi_handbook_text.py`

Extracts the ESI Handbook PDF into a single plaintext file for use as a system-level context prefix in LLM triage queries (no embedding or FAISS). Uses the same page-filtering logic as `build_esi_handbook_index.py`.

```bash
.venv/bin/python scripts/extract_esi_handbook_text.py --pdf PATH [--out PATH]
```

- **--pdf** — Path to the ESI Handbook PDF (required).
- **--out** — Output plaintext file (default: `data/corpus/esi_handbook_text.txt`).

**Output:** One plaintext file with page markers (`--- Page N ---`) between pages so the LLM can cite specific pages. After running, use `src.rag.esi_handbook.load_esi_handbook_prefix(path)` (or `load_esi_handbook_prefix()` for the default path) to load the text for injection into prompts. Not wired into the RAG pipeline — available for future use.

---

## Validation findings (Feb 2026)

`scripts/validate_retrieval.py` was run after re-provisioning `pmc_embeddings` + `pmc_articles` with `QUALIFY` deduplication. Three checks were performed.

**Check 1 — Row counts (free):** PASS. Public table: 2,311,740 rows across 2,179,871 distinct `pmc_id`s. Owned table: 2,179,871 rows (131,869 fewer). Check 1 passes when `own_n == COUNT(DISTINCT pmc_id)` in the public table — QUALIFY keeps exactly one row per distinct `pmc_id`, so this must hold exactly. Zero duplicate `pmc_id` entries in the owned table after dedup.

The 131,869 dropped rows are all `pmid='0'` placeholders that have a corresponding real-PMID row for the same `pmc_id`. The remaining 111,324 `pmid='0'` rows in the public table are **singletons** — articles deposited but not yet assigned a PMID — and are kept by QUALIFY (they rank first as the only row for their `pmc_id`). Total `pmid='0'` rows in public: 243,193 = 131,869 dropped + 111,324 singletons kept.

**Why the public table has 131,869 duplicate `pmc_id` rows — empirical evidence.** Querying the public table directly for one affected `pmc_id`:

```sql
SELECT pmc_id, version, last_updated
FROM `bigquery-public-data.pmc_open_access_commercial.articles`
WHERE pmc_id = 'PMC12440108'
```

| pmc_id      | version | last_updated        |
| ----------- | ------- | ------------------- |
| PMC12440108 | 1.0     | 2025-09-23T23:17:36 |
| PMC12440108 | 1.0     | 2025-09-17T23:25:19 |

Two rows for the same `pmc_id` already exist in the public table. Our owned table inherits them via `CREATE TABLE ... AS SELECT`. Including `pmid` reveals the mechanism:

```sql
SELECT pmid, pmc_id, article_citation, version, last_updated
FROM `bigquery-public-data.pmc_open_access_commercial.articles`
WHERE pmc_id = 'PMC12440108'
```

| pmid     | pmc_id      | article_citation                    | version | last_updated        |
| -------- | ----------- | ----------------------------------- | ------- | ------------------- |
| 0        | PMC12440108 | Res Sq. 2025 Sep 9;:rs.3.rs-7347858 | 1.0     | 2025-09-17T23:25:19 |
| 40964036 | PMC12440108 | Res Sq. 2025 Sep 9;:rs.3.rs-7347858 | 1.0     | 2025-09-23T23:17:36 |

Same citation and version, different `pmid` and `last_updated`. The `pmid=0` row was ingested 6 days before the real-PMID row, consistent with PMID assignment at NLM happening after PMC deposit. `QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY pmid DESC) = 1` in `create_vector_store.py` keeps only the real-PMID version.

In most cases the two rows have identical embeddings (no text change during PMID assignment). For 2 of the 218 sampled articles in Check 2A the embeddings differed (`max_dist=0.24`), meaning the article text was also updated between deposit and PMID assignment. Keeping `ORDER BY pmid DESC` (the newer, real-PMID row) therefore keeps the more complete version in those cases.

**Check 2 — Coverage + embedding integrity (Part B free, Part A ~$0.14):** PASS (both parts). Check 2 is a two-step chain that completes what Check 1 started.

Check 1 verified the *count* of dropped rows was correct (`own_n == pub_distinct_n`). Check 2 completes the chain:

- **Part B — right articles present (free):** verifies no `pmc_id` is entirely absent from owned. A `LEFT JOIN` on `pmc_id` (not the pair) counts distinct `pmc_id`s with a real PMID in public that have no row at all in owned. Result: **0 entirely absent articles.** Every article is present.
  Separately, 23 `(pmc_id, pmid)` pairs exist in public but not in owned — but in every case the `pmc_id` IS in owned with a different `pmid`. These are articles where the public table has two real-PMID rows for the same `pmc_id` (PMID corrections or reassignments — e.g. `PMC8291212` has both `pmid=34308251` and `pmid=34308252`, differing by 1). QUALIFY kept the lexicographically higher `pmid` and dropped the other. Not data loss: the article content is present.
- **Part A — remaining rows uncorrupted (~$0.14):** verifies that the rows that *do* exist in the owned table have embeddings matching the source. A deterministic hash sample of ~218 rows (`MOD(ABS(FARM_FINGERPRINT(pmc_id)), 10_000) = 0` on the public table, ~1-in-10,000) is inner-joined to owned on `(pmc_id, pmid)` and `MIN(ML.DISTANCE(..., 'EUCLIDEAN'))` is tested per group. The inner join skips rows absent from owned by design; Part B already confirmed no articles are missing. Result: **0 corrupted rows** across 218 sampled groups (`max_dist=0.0`).

Together: Check 1 verifies the right *count* was dropped → Part B verifies all *articles* are present → Part A verifies the rows that remain are uncorrupted.

**Check 3 — VECTOR_SEARCH retrieval consistency (~$0.47):** PASS. The public table has no vector index (it is read-only and Google has not indexed it), so every call against it is a brute-force exact scan. The owned table uses the IVF index.

BigQuery's default `fraction_lists_to_search` is auto-tuned aggressively — at ~0.5% of IVF lists for a 2.3M-row table, it scans only a small fraction of clusters and gave **2/10 top-10 overlap** on the first post-index-ready run. Setting `fraction_lists_to_search=0.1` (10% of lists) recovers **≥9/10 overlap** at low marginal cost (the dominant per-call cost is the ~4.4 GB `pmc_articles` snippet JOIN, not the index search itself). This setting is applied in both `validate_retrieval.py` and `retrieval.py`.

Final results with `fraction_lists_to_search=0.1`: top-1 match on all 3 canonical ED queries; top-10 overlap ≥9/10 on each.

**Overall status:** PASS. The copy is faithful; the duplicate pattern is understood and handled; IVF recall is calibrated.
