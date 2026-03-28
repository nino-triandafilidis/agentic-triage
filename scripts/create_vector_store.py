"""One-time BQ provisioning: create pmc_embeddings and pmc_articles tables.

Run with:
    .venv/bin/python scripts/create_vector_store.py

Each step prints the estimated cost and waits for your confirmation before
touching BigQuery.  Steps are idempotent — safe to re-run after a failure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402 (path insert must come first)

# ── Cost constants ─────────────────────────────────────────────────────
# BQ on-demand: $5 / TB processed
# pmc_embeddings scan: 768 dims × 8 bytes × 2.31M rows ≈ 13.85 GB → ~$0.07
# pmc_articles scan: public table article_text column ~150 GB → ~$0.75
#   (SUBSTR in SELECT does not reduce bytes read from public table source)
# Storage: $0.020/GB/month (active), $0.010/GB/month (long-term)
#   pmc_embeddings ≈ 14.5 GB compressed  → ~$0.29/month active
#   pmc_articles   larger with 8k snippets (estimated — confirm after build)
#     (stores 8000-char snippets, not full article_text)

EMBEDDINGS_TABLE = f"{config.PROJECT_ID}.{config.USER_DATASET}.pmc_embeddings"
ARTICLES_TABLE   = f"{config.PROJECT_ID}.{config.USER_DATASET}.pmc_articles"
SOURCE_TABLE     = config.PUBMED_TABLE


def _confirm(prompt: str) -> None:
    """Print prompt and abort unless user types 'y'."""
    answer = input(f"\n{prompt}\nProceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def _table_exists(table_id: str) -> bool:
    try:
        config.bq_client.get_table(table_id)
        return True
    except Exception:
        return False


def _index_exists(table_id: str, index_name: str) -> bool:
    dataset = table_id.split(".")[1]
    project = table_id.split(".")[0]
    sql = f"""
    SELECT COUNT(*) AS cnt
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.VECTOR_INDEXES`
    WHERE index_name = '{index_name}'
    """
    result = config.bq_client.query(sql).to_dataframe()
    return int(result["cnt"].iloc[0]) > 0


# ── Step 1: pmc_embeddings ─────────────────────────────────────────────
def create_embeddings_table() -> None:
    if _table_exists(EMBEDDINGS_TABLE):
        print(f"[SKIP] {EMBEDDINGS_TABLE} already exists.")
        return

    _confirm(
        f"STEP 1 — Create {EMBEDDINGS_TABLE}\n"
        f"  Scans embedding column of {SOURCE_TABLE} (13.85 GB)\n"
        f"  Estimated one-time BQ cost: ~$0.07\n"
        f"  Ongoing storage: ~$0.29/month"
    )

    # QUALIFY deduplicates articles that appear twice under the same pmc_id.
    # The public table contains ~65k pmc_ids with two rows each:
    #   Row 1 — pmid=0:    article deposited to PMC before NLM assigned a PMID
    #   Row 2 — pmid=REAL: same article after PMID was assigned (higher last_updated)
    #
    # ORDER BY pmid DESC keeps the real-PMID row (real_pmid > 0 always).
    # Assumptions this relies on:
    #   (A) pmid=0 is always a placeholder, not a valid PMID — true by NCBI convention
    #   (B) the real-PMID row is the later/more complete version — supported by two
    #       independent observations: (i) last_updated on real-PMID rows is consistently
    #       later than on pmid=0 rows (confirmed empirically for PMC12440108:
    #       pmid=0 last_updated 2025-09-17, pmid=40964036 last_updated 2025-09-23);
    #       (ii) validate_retrieval.py Check 2 found non-zero embedding distances for
    #       2/218 sampled articles when comparing pmid=0 against real-PMID versions,
    #       meaning the article_text was updated between deposit and PMID assignment —
    #       the real-PMID version is the updated one
    #   (C) no pmc_id has more than two rows — true in the current dataset (Check 1
    #       showed COUNT(*) - COUNT(DISTINCT pmc_id) = 131,869 = 2 × 65,934 unique
    #       duplicated pmc_ids, consistent with exactly 2 rows each)
    sql = f"""
    CREATE TABLE `{EMBEDDINGS_TABLE}` AS
    SELECT
        pmc_id,
        pmid,
        ml_generate_embedding_result,
        license
    FROM `{SOURCE_TABLE}`
    WHERE ml_generate_embedding_result IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY pmid DESC) = 1
    """
    print("Creating pmc_embeddings table … (this may take a few minutes)")
    config.bq_client.query(sql).result()
    print(f"[OK] {EMBEDDINGS_TABLE} created.")


# ── Step 2: vector index on pmc_embeddings ─────────────────────────────
def create_vector_index() -> None:
    index_name = "pmc_emb_idx"
    if _index_exists(EMBEDDINGS_TABLE, index_name):
        print(f"[SKIP] Vector index '{index_name}' already exists.")
        return

    _confirm(
        f"STEP 2 — Create IVF vector index '{index_name}' on {EMBEDDINGS_TABLE}\n"
        f"  This is a DDL operation — no additional data scan cost.\n"
        f"  Index build runs asynchronously in BQ; status will become READY\n"
        f"  within minutes.  No charge until VECTOR_SEARCH is called."
    )

    sql = f"""
    CREATE VECTOR INDEX {index_name}
    ON `{EMBEDDINGS_TABLE}`(ml_generate_embedding_result)
    OPTIONS(index_type = 'IVF', distance_type = 'COSINE')
    """
    print(f"Creating vector index '{index_name}' … (asynchronous in BQ)")
    config.bq_client.query(sql).result()
    print(f"[OK] Vector index '{index_name}' creation submitted.")
    print("     Run the check below to confirm status = READY:")
    print(f"     SELECT index_name, coverage_percentage, last_refresh_time")
    print(f"     FROM `{config.PROJECT_ID}.{config.USER_DATASET}.INFORMATION_SCHEMA.VECTOR_INDEXES`")


# ── Step 3: pmc_articles ───────────────────────────────────────────────
def create_articles_table() -> None:
    if _table_exists(ARTICLES_TABLE):
        print(f"[SKIP] {ARTICLES_TABLE} already exists.")
        return

    _confirm(
        f"STEP 3 — Create {ARTICLES_TABLE}\n"
        f"  Scans full {SOURCE_TABLE} (article_text column, ~150 GB)\n"
        f"  Estimated one-time BQ cost: ~$0.75\n"
        f"  Ongoing storage: higher than the old 2k setup (estimate; confirm after build)\n"
        f"  article_text stored as SUBSTR(article_text, 1, 8000) to match\n"
        f"  RETRIEVAL_CONTEXT_CHARS in the local pipeline."
    )

    # QUALIFY deduplicates on the same logic as pmc_embeddings — see comment above.
    # article_text is stored truncated to 8000 chars (= config.RETRIEVAL_CONTEXT_CHARS).
    # Storing the full text caused ~$0.68/retrieval-call because BigQuery columnar
    # storage reads the entire article_text column (~109 GB) even when JOINing only
    # top-k rows. SUBSTR in the source SELECT does not reduce the provisioning scan
    # cost (still reads full column from public table), but keeps the owned table
    # materially smaller than full text while preserving the first 8k chars.
    sql = f"""
    CREATE TABLE `{ARTICLES_TABLE}`
    CLUSTER BY pmc_id
    AS
    SELECT
        pmc_id,
        pmid,
        SUBSTR(article_text, 1, 8000) AS article_text,
        article_citation,
        title,
        author,
        license
    FROM `{SOURCE_TABLE}`
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pmc_id ORDER BY pmid DESC) = 1
    """
    print("Creating pmc_articles table … (this may take several minutes)")
    config.bq_client.query(sql).result()
    print(f"[OK] {ARTICLES_TABLE} created.")


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("BQ Vector Store Provisioning")
    print(f"Project : {config.PROJECT_ID}")
    print(f"Dataset : {config.USER_DATASET}")
    print("=" * 60)
    print(
        "\nTotal estimated one-time cost: ~$0.82"
        "\nTotal ongoing storage cost:    ~$0.31–0.33/month"
        "\nAfter setup, retrieval cost per call: ~$0.10 (~12.6 GB embedding scan + ~4.4 GB snippet join)"
    )

    config.setup_clients(bq_guardrail=False)  # provisioning scans ~150 GB legitimately

    create_embeddings_table()
    create_vector_index()
    create_articles_table()

    print("\n[DONE] All steps complete.")
    print("Run the retrieval smoke-test to verify:")
    print("  .venv/bin/python -c \"import src.config as c; c.setup_clients(); from src.rag.retrieval import search_pubmed_articles; print(search_pubmed_articles('chest pain triage', top_k=3))\"")


if __name__ == "__main__":
    main()
