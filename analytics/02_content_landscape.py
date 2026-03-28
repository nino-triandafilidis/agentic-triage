"""Broad content analytics on the PMC Open Access Commercial dataset.

Produces summary statistics that help assess what kinds of articles exist
in the vector store — journal distribution, publication years, article
length distribution, and MeSH-level topic categories (approximated from
journal names since MeSH terms are not in this table).

Usage:
    python analytics/02_content_landscape.py [--out analytics/results]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _gcp import get_bq_client
import pandas as pd

# Client initialised in main() after arg parsing
bq = None

TABLE = "bigquery-public-data.pmc_open_access_commercial.articles"


def run_query(sql: str) -> pd.DataFrame:
    return bq.query(sql).to_dataframe()


def total_articles() -> int:
    df = run_query(f"SELECT COUNT(*) AS n FROM `{TABLE}`")
    return int(df.iloc[0]["n"])


def journal_distribution(top_n: int = 50) -> pd.DataFrame:
    """Top journals by article count (extracted from article_citation)."""
    # article_citation format: "Journal Name. YYYY Mon DD; Vol:Page"
    sql = f"""
    SELECT
        REGEXP_EXTRACT(article_citation, r'^([^.]+)') AS journal_title,
        COUNT(*) AS article_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
    FROM `{TABLE}`
    WHERE article_citation IS NOT NULL
    GROUP BY journal_title
    ORDER BY article_count DESC
    LIMIT {top_n}
    """
    return run_query(sql)


def year_distribution() -> pd.DataFrame:
    """Publication year distribution (extracted from article_citation).

    Filters to plausible publication years (1990-2026) to remove noise
    from 4-digit numbers that appear in journal names (e.g. "Lond 1886")
    or other citation fields.
    """
    sql = f"""
    SELECT
        CAST(REGEXP_EXTRACT(article_citation, r'(\\d{{4}})') AS INT64) AS year,
        COUNT(*) AS article_count
    FROM `{TABLE}`
    WHERE article_citation IS NOT NULL
      AND REGEXP_EXTRACT(article_citation, r'(\\d{{4}})') IS NOT NULL
      AND CAST(REGEXP_EXTRACT(article_citation, r'(\\d{{4}})') AS INT64) BETWEEN 1990 AND 2026
    GROUP BY year
    ORDER BY year
    """
    return run_query(sql)


def article_length_distribution() -> pd.DataFrame:
    """Distribution of article_text length (char count), sampled for cost."""
    sql = f"""
    SELECT
        CASE
            WHEN LENGTH(article_text) < 5000 THEN '<5k chars'
            WHEN LENGTH(article_text) < 20000 THEN '5k-20k chars'
            WHEN LENGTH(article_text) < 50000 THEN '20k-50k chars'
            WHEN LENGTH(article_text) < 100000 THEN '50k-100k chars'
            ELSE '100k+ chars'
        END AS length_bucket,
        COUNT(*) AS article_count
    FROM `{TABLE}`
    WHERE article_text IS NOT NULL
    GROUP BY length_bucket
    ORDER BY
        CASE length_bucket
            WHEN '<5k chars' THEN 1
            WHEN '5k-20k chars' THEN 2
            WHEN '20k-50k chars' THEN 3
            WHEN '50k-100k chars' THEN 4
            ELSE 5
        END
    """
    return run_query(sql)


def license_distribution() -> pd.DataFrame:
    """Distribution of license types."""
    sql = f"""
    SELECT
        license,
        COUNT(*) AS article_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
    FROM `{TABLE}`
    WHERE license IS NOT NULL
    GROUP BY license
    ORDER BY article_count DESC
    """
    return run_query(sql)


def null_field_summary(sample_size: int = 50_000) -> pd.DataFrame:
    """Percentage of nulls for key metadata columns (sampled for cost/memory)."""
    sql = f"""
    SELECT
        COUNT(*) AS total,
        COUNTIF(article_citation IS NULL) AS null_citation,
        COUNTIF(pmid IS NULL) AS null_pmid,
        COUNTIF(pmc_id IS NULL) AS null_pmcid,
        COUNTIF(article_text IS NULL) AS null_text,
        COUNTIF(license IS NULL) AS null_license,
        COUNTIF(title IS NULL) AS null_title,
        COUNTIF(author IS NULL) AS null_author
    FROM (
        SELECT article_citation, pmid, pmc_id, article_text, license, title, author
        FROM `{TABLE}`
        LIMIT {sample_size}
    )
    """
    return run_query(sql)


def keyword_prevalence(keywords: list[str], sample_size: int = 100_000) -> pd.DataFrame:
    """How often domain-relevant keywords appear in article_text (sampled)."""
    cases = ",\n        ".join(
        f"COUNTIF(LOWER(article_text) LIKE '%{kw.lower()}%') AS `{kw}`"
        for kw in keywords
    )
    # Sample by hashing pmc_id — avoids ORDER BY on 2.3M rows (memory efficient)
    sample_pct = max(1, round(sample_size / 2_311_736 * 100))
    sql = f"""
    SELECT
        COUNT(*) AS sample_size,
        {cases}
    FROM `{TABLE}`
    WHERE article_text IS NOT NULL
      AND MOD(ABS(FARM_FINGERPRINT(pmc_id)), 100) < {sample_pct}
    """
    return run_query(sql)


def main():
    global bq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="GCP project ID override")
    parser.add_argument("--out", default="analytics/results", help="Output directory")
    args = parser.parse_args()
    bq = get_bq_client(args.project)

    os.makedirs(args.out, exist_ok=True)

    # ── Total ────────────────────────────────────────────────────────────
    total = total_articles()
    print(f"Total articles in PMC OA Commercial: {total:,}\n")

    # ── Journal distribution ─────────────────────────────────────────────
    print("Top 30 journals by article count:")
    journals = journal_distribution(50)
    print(journals.head(30).to_string(index=False))
    journals.to_csv(os.path.join(args.out, "journal_distribution.csv"), index=False)
    print()

    # ── Year distribution ────────────────────────────────────────────────
    print("Year distribution:")
    years = year_distribution()
    # Show only last 25 years for readability
    recent = years[years["year"] >= 2000]
    print(recent.to_string(index=False))
    years.to_csv(os.path.join(args.out, "year_distribution.csv"), index=False)
    print()

    # ── Article length ───────────────────────────────────────────────────
    print("Article length distribution:")
    lengths = article_length_distribution()
    print(lengths.to_string(index=False))
    lengths.to_csv(os.path.join(args.out, "length_distribution.csv"), index=False)
    print()

    # ── License distribution ─────────────────────────────────────────────
    print("License distribution:")
    licenses = license_distribution()
    print(licenses.to_string(index=False))
    licenses.to_csv(os.path.join(args.out, "license_distribution.csv"), index=False)
    print()

    # ── Null field summary ───────────────────────────────────────────────
    print("Null field summary:")
    nulls = null_field_summary()
    print(nulls.to_string(index=False))
    print()

    # ── ED / Triage keyword prevalence ───────────────────────────────────
    ed_keywords = [
        "emergency department",
        "emergency room",
        "triage",
        "emergency severity index",
        "acute care",
        "critical care",
        "intensive care",
        "chest pain",
        "dyspnea",
        "abdominal pain",
        "trauma",
        "sepsis",
        "stroke",
        "myocardial infarction",
        "cardiac arrest",
        "respiratory failure",
        "clinical decision support",
        "vital signs",
        "glasgow coma",
    ]

    print(f"Keyword prevalence in {100_000:,}-article sample:")
    kw_df = keyword_prevalence(ed_keywords, sample_size=100_000)
    # Transpose for readability
    kw_long = kw_df.drop(columns=["sample_size"]).T.reset_index()
    kw_long.columns = ["keyword", "count"]
    kw_long["pct_of_sample"] = (kw_long["count"] / 100_000 * 100).round(2)
    kw_long = kw_long.sort_values("count", ascending=False)
    print(kw_long.to_string(index=False))
    kw_long.to_csv(os.path.join(args.out, "ed_keyword_prevalence.csv"), index=False)

    # ── Summary JSON ─────────────────────────────────────────────────────
    summary = {
        "total_articles": total,
        "top_10_journals": journals.head(10).to_dict(orient="records"),
        "year_range": [int(years["year"].min()), int(years["year"].max())],
        "ed_keyword_hit_rate": {
            row["keyword"]: row["pct_of_sample"]
            for _, row in kw_long.iterrows()
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
