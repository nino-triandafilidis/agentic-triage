"""Measure per-call BQ cost of the current two-table VECTOR_SEARCH retrieval.

Runs N_QUERIES independent retrieval calls (no cache) and reports:
  - bytes processed / billed per call (from the QueryJob object)
  - wall-clock time per call
  - cost estimate at $6.25 / TiB

Note: `total_bytes_billed` from the job API is the pre-index-optimization
estimate for VECTOR_SEARCH (see config.py comment).  Actual billing is lower
and will appear in pubmed-rag-pipeline.billing_export once that export
populates (typically within a few hours of usage).

Usage:
    .venv/bin/python experiments/measure_retrieval_cost.py
    .venv/bin/python experiments/measure_retrieval_cost.py --n-queries 3 --top-k 10
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config

BQ_PRICE_PER_TIB = 6.25  # USD, on-demand


def build_query_text(row: pd.Series) -> str:
    return f"{row.get('HPI', '')} {row.get('patient_info', '')} {row.get('initial_vitals', '')}".strip()


def run_single_retrieval(query_text: str, top_k: int) -> dict:
    """Run one VECTOR_SEARCH call and return job stats."""
    sql = f"""
    SELECT
        base.pmc_id,
        base.pmid,
        distance,
        docs.article_text,
        docs.article_citation
    FROM VECTOR_SEARCH(
        TABLE `{config.PMC_EMBEDDINGS_TABLE}`,
        'ml_generate_embedding_result',
        (SELECT ml_generate_embedding_result
         FROM ML.GENERATE_EMBEDDING(
             MODEL `{config.EMBEDDING_MODEL}`,
             (SELECT @query AS content)
         )),
        top_k => {top_k},
        options => '{{"fraction_lists_to_search": 0.1}}'
    )
    JOIN `{config.PMC_ARTICLES_TABLE}` docs ON base.pmc_id = docs.pmc_id
    ORDER BY distance
    LIMIT {top_k}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("query", "STRING", query_text)],
        maximum_bytes_billed=config.BQ_MAX_BYTES_BILLED,
    )

    t0 = time.time()
    job = config.bq_client.query(sql, job_config=job_config)
    rows = job.result().to_dataframe()
    elapsed = time.time() - t0

    # Re-fetch job to get billing stats (populated after result())
    job = config.bq_client.get_job(job.job_id, location=config.BQ_LOCATION)

    gb_processed = (job.total_bytes_processed or 0) / 1024 ** 3
    gb_billed    = (job.total_bytes_billed    or 0) / 1024 ** 3
    cost_usd     = gb_billed / 1024 * BQ_PRICE_PER_TIB

    return {
        "job_id":        job.job_id,
        "elapsed_s":     elapsed,
        "rows_returned": len(rows),
        "gb_processed":  gb_processed,
        "gb_billed":     gb_billed,
        "cost_usd":      cost_usd,
        "query_preview": query_text[:80],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--top-k",    type=int, default=10)
    parser.add_argument("--input",    type=Path, default=ROOT / "data" / "splits" / "scratch.csv")
    args = parser.parse_args()

    config.setup_clients(bq_guardrail=True)

    df = pd.read_csv(args.input, nrows=args.n_queries)
    queries = [build_query_text(row) for _, row in df.iterrows()]

    print(f"Experiment: {args.n_queries} retrieval calls, top_k={args.top_k}")
    print(f"Start time (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(f"Table: {config.PMC_EMBEDDINGS_TABLE}")
    print(f"fraction_lists_to_search: 0.1")
    print(f"guardrail: {config.BQ_MAX_BYTES_BILLED / 1024**3:.0f} GB max per query")
    print("-" * 80)

    results = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{args.n_queries}] Running... ", end="", flush=True)
        stats = run_single_retrieval(query, top_k=args.top_k)
        results.append(stats)
        print(f"{stats['elapsed_s']:.1f}s | "
              f"processed={stats['gb_processed']:.2f} GiB | "
              f"billed(API)={stats['gb_billed']:.2f} GiB | "
              f"~${stats['cost_usd']:.4f}")

    print("-" * 80)
    total_billed = sum(r["gb_billed"] for r in results)
    total_cost   = sum(r["cost_usd"]  for r in results)
    avg_billed   = total_billed / len(results)
    avg_elapsed  = sum(r["elapsed_s"] for r in results) / len(results)

    print(f"Calls:              {len(results)}")
    print(f"Avg GiB billed/call (API estimate): {avg_billed:.2f} GiB")
    print(f"Avg cost/call (API estimate):       ${total_cost / len(results):.4f}")
    print(f"Avg wall-clock time:                {avg_elapsed:.1f}s")
    print()
    print("NOTE: 'billed(API)' is the pre-index estimate from the job API and is")
    print("      inflated vs actual GCP billing.  Check billing_export dataset in")
    print("      a few hours for the true per-call cost.")
    print()

    # Save results for later cross-reference with billing export
    out = ROOT / "data" / "retrieval_cost_experiment.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Raw results saved to {out.name}")
    print(f"End time (UTC): {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
