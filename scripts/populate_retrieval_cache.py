"""Populate retrieval cache for a CSV file using one batched BQ query.

Instead of calling search_pubmed_articles once per row (~$0.10 each),
this script uploads all query strings to a temp BQ table and runs a
single VECTOR_SEARCH — scanning the embedding column once regardless
of row count.

Cost: ~$0.10 for all 100 scratch rows (vs ~$10 sequential).

Usage:
    .venv/bin/python scripts/populate_retrieval_cache.py
    .venv/bin/python scripts/populate_retrieval_cache.py --input data/splits/dev.csv --top-k 10

    # Fetch 8k chars from the public table (bypasses local 2k-char pmc_articles):
    .venv/bin/python scripts/populate_retrieval_cache.py --context-chars 8000 --use-public-table
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config

CACHE_PATH = ROOT / "data" / "cache" / "retrieval_cache.parquet"
TEMP_TABLE = f"{config.PROJECT_ID}.{config.USER_DATASET}.scratch_queries_temp"


def build_query_text(row: pd.Series) -> str:
    hpi = str(row.get("HPI", ""))
    patient_info = str(row.get("patient_info", ""))
    initial_vitals = str(row.get("initial_vitals", ""))
    return f"{hpi} {patient_info} {initial_vitals}".strip()


def make_hash(query_text: str) -> str:
    return hashlib.sha256(query_text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data" / "splits" / "scratch.csv"))
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of articles to retrieve per query (default 10).")
    parser.add_argument("--context-chars", type=int, default=None,
                        help="Max chars per article. If set, article_text is truncated "
                             "via SUBSTR in SQL. Default: no truncation (uses stored text).")
    parser.add_argument("--use-public-table", action="store_true",
                        help="JOIN against the public PMC table instead of the local "
                             "pmc_articles table. Use this when you need more article "
                             "text than the owned pmc_articles table currently stores. "
                             "Costs more (~$0.82) due to full article_text column scan.")
    args = parser.parse_args()

    # Provisioning script — needs BQ client and may scan >30 GB, disable guardrail
    config.setup_clients(bq_guardrail=False, force_bq=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    df["_query_text"] = df.apply(build_query_text, axis=1)
    df["_query_hash"] = df["_query_text"].apply(make_hash)

    unique = df[["_query_hash", "_query_text"]].drop_duplicates("_query_hash")
    print(f"{len(unique)} unique queries")

    # Upload query strings to a temp BQ table
    # ML.GENERATE_EMBEDDING requires the text column to be named 'content'
    upload_df = unique.rename(columns={"_query_text": "content", "_query_hash": "query_hash"})
    schema = [
        bigquery.SchemaField("query_hash", "STRING"),
        bigquery.SchemaField("content", "STRING"),
    ]
    load_cfg = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    config.bq_client.load_table_from_dataframe(
        upload_df, TEMP_TABLE, job_config=load_cfg
    ).result()
    print(f"Uploaded {len(upload_df)} rows to temp table")

    PUBLIC_TABLE = "bigquery-public-data.pmc_open_access_commercial.articles"
    articles_table = f"`{PUBLIC_TABLE}`" if args.use_public_table else f"`{config.PMC_ARTICLES_TABLE}`"

    # Build article_text expression: apply SUBSTR if --context-chars is set
    if args.context_chars:
        article_text_expr = f"SUBSTR(docs.article_text, 1, {args.context_chars})"
    else:
        article_text_expr = "docs.article_text"

    cost_note = (
        f"~$0.82 (public table scan)" if args.use_public_table
        else f"~$0.10 (local pmc_articles)"
    )
    print(f"Articles source: {articles_table}  (est. cost: {cost_note})")
    if args.context_chars:
        print(f"Truncating article_text to {args.context_chars} chars")

    sql = f"""
    SELECT
        query.query_hash,
        base.pmc_id,
        base.pmid,
        distance,
        {article_text_expr} AS article_text,
        docs.article_citation
    FROM VECTOR_SEARCH(
        TABLE `{config.PMC_EMBEDDINGS_TABLE}`,
        'ml_generate_embedding_result',
        (SELECT ml_generate_embedding_result, query_hash
         FROM ML.GENERATE_EMBEDDING(
             MODEL `{config.EMBEDDING_MODEL}`,
             TABLE `{TEMP_TABLE}`
         )),
        top_k => {args.top_k},
        options => '{{"fraction_lists_to_search": 0.1}}'
    )
    JOIN {articles_table} docs ON base.pmc_id = docs.pmc_id
    ORDER BY query.query_hash, distance
    """

    print("Running batched VECTOR_SEARCH...")
    results = config.bq_client.query(sql).to_dataframe()
    print(f"Got {len(results)} result rows")

    results["top_k"] = args.top_k
    results["context_chars"] = args.context_chars or 0  # 0 = no truncation (stored text)

    # Merge with existing cache (preserves entries for other top_k / context_chars combos)
    if CACHE_PATH.exists():
        existing = pd.read_parquet(CACHE_PATH)
        # Add context_chars column to old caches that lack it
        if "context_chars" not in existing.columns:
            existing["context_chars"] = 0
        mask = (existing["top_k"] == args.top_k) & (existing["context_chars"] == results["context_chars"].iloc[0])
        existing = existing[~mask]
        results = pd.concat([existing, results], ignore_index=True)

    results.to_parquet(CACHE_PATH, index=False)
    print(f"Cache saved to {CACHE_PATH} ({len(results)} rows)")

    config.bq_client.delete_table(TEMP_TABLE, not_found_ok=True)
    print("Temp table dropped")


if __name__ == "__main__":
    main()
