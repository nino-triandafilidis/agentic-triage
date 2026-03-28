"""One-call BigQuery cost comparison: brute-force vs IVF retrieval paths.

Runs exactly one query per strategy and reports:
  - total bytes billed (GiB / TiB)
  - no-free-tier cost estimate at $6.25/TiB
  - wall-clock latency

Strategies:
  1) brute_force_public_vector_only
  2) ivf_owned_vector_only
  3) ivf_owned_with_snippet_join  (matches current retrieval pipeline shape)

Usage:
  .venv/bin/python experiments/compare_bruteforce_vs_ivf_one_call.py
  .venv/bin/python experiments/compare_bruteforce_vs_ivf_one_call.py --query "acute chest pain diaphoresis" --top-k 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config

BQ_PRICE_PER_TIB = 6.25


def _query_sql(strategy: str, top_k: int) -> str:
    if strategy == "brute_force_public_vector_only":
        return f"""
        -- experiment: brute_force_vs_ivf_one_call
        SELECT
            base.pmc_id,
            base.pmid,
            distance
        FROM VECTOR_SEARCH(
            TABLE `{config.PUBMED_TABLE}`,
            'ml_generate_embedding_result',
            (SELECT ml_generate_embedding_result
             FROM ML.GENERATE_EMBEDDING(
                 MODEL `{config.EMBEDDING_MODEL}`,
                 (SELECT @query AS content)
             )),
            top_k => {top_k}
        )
        ORDER BY distance
        LIMIT {top_k}
        """

    if strategy == "ivf_owned_vector_only":
        return f"""
        -- experiment: brute_force_vs_ivf_one_call
        SELECT
            base.pmc_id,
            base.pmid,
            distance
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
        ORDER BY distance
        LIMIT {top_k}
        """

    if strategy == "ivf_owned_with_snippet_join":
        return f"""
        -- experiment: brute_force_vs_ivf_one_call
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

    raise ValueError(f"Unknown strategy: {strategy}")


def _run_once(strategy: str, query_text: str, top_k: int) -> dict:
    sql = _query_sql(strategy=strategy, top_k=top_k)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("query", "STRING", query_text)],
        maximum_bytes_billed=config.BQ_MAX_BYTES_BILLED,
        use_query_cache=False,
    )

    t0 = time.time()
    job = config.bq_client.query(sql, job_config=job_config)
    _ = job.result()
    elapsed_s = time.time() - t0

    job = config.bq_client.get_job(job.job_id, location=config.BQ_LOCATION)
    bytes_billed = int(job.total_bytes_billed or 0)
    gib_billed = bytes_billed / (1024 ** 3)
    tib_billed = bytes_billed / (1024 ** 4)
    usd_no_free_tier = tib_billed * BQ_PRICE_PER_TIB

    return {
        "strategy": strategy,
        "job_id": job.job_id,
        "gib_billed": gib_billed,
        "tib_billed": tib_billed,
        "usd_no_free_tier": usd_no_free_tier,
        "elapsed_s": elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="adult ED triage chest pain dyspnea emergency severity index",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # Cost-risk assessment: one call per strategy, each typically < 30 GiB cap.
    # Typical observed no-free-tier total is around $0.25-$0.30 for all 3 calls.
    print("Cost-risk assessment:")
    print("  This will run 3 BigQuery queries (1 per strategy), uncached.")
    print("  Expected no-free-tier total: roughly $0.25-$0.30 (varies by query shape).")
    print("  Guardrail: maximum_bytes_billed = 30 GiB per query.")
    answer = input("Proceed with BigQuery calls? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    config.setup_clients(bq_guardrail=True)

    strategies = [
        "brute_force_public_vector_only",
        "ivf_owned_vector_only",
        "ivf_owned_with_snippet_join",
    ]

    rows = []
    for i, strategy in enumerate(strategies, start=1):
        print(f"[{i}/{len(strategies)}] Running {strategy} ...", flush=True)
        rows.append(_run_once(strategy=strategy, query_text=args.query, top_k=args.top_k))

    df = pd.DataFrame(rows).sort_values("usd_no_free_tier", ascending=False)

    for col in ["gib_billed", "tib_billed", "usd_no_free_tier", "elapsed_s"]:
        df[col] = df[col].round(6)

    print("\n=== One-call comparison (no free tier) ===")
    print(df[["strategy", "gib_billed", "tib_billed", "usd_no_free_tier", "elapsed_s", "job_id"]].to_string(index=False))

    if "brute_force_public_vector_only" in set(df["strategy"]) and "ivf_owned_vector_only" in set(df["strategy"]):
        bf = float(df.loc[df["strategy"] == "brute_force_public_vector_only", "usd_no_free_tier"].iloc[0])
        ivf = float(df.loc[df["strategy"] == "ivf_owned_vector_only", "usd_no_free_tier"].iloc[0])
        delta = ivf - bf
        pct = (delta / bf * 100.0) if bf else 0.0
        print("\nVector-only delta (IVF - brute-force):")
        print(f"  ${delta:.6f} ({pct:+.2f}%)")

    if "ivf_owned_vector_only" in set(df["strategy"]) and "ivf_owned_with_snippet_join" in set(df["strategy"]):
        ivf_only = float(df.loc[df["strategy"] == "ivf_owned_vector_only", "usd_no_free_tier"].iloc[0])
        ivf_join = float(df.loc[df["strategy"] == "ivf_owned_with_snippet_join", "usd_no_free_tier"].iloc[0])
        join_delta = ivf_join - ivf_only
        print("\nJoin overhead inside IVF pipeline:")
        print(f"  ${join_delta:.6f} per call")


if __name__ == "__main__":
    main()

