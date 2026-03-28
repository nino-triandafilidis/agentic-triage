"""Discover the schema of the PMC Open Access table in BigQuery.

Queries INFORMATION_SCHEMA to understand available columns, then samples
a few rows to see what the actual data looks like.

Usage:
    python analytics/01_schema_discovery.py [--project YOUR_PROJECT_ID]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _gcp import get_bq_client

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", default=None, help="GCP project ID override")
args = parser.parse_args()

bq = get_bq_client(args.project)

DATASET = "bigquery-public-data.pmc_open_access_commercial"
TABLE = f"{DATASET}.articles"

# ── 1. Column schema ────────────────────────────────────────────────────
print("=" * 70)
print("COLUMN SCHEMA")
print("=" * 70)

schema_sql = f"""
SELECT column_name, data_type, is_nullable
FROM `{DATASET}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'articles'
ORDER BY ordinal_position
"""
schema_df = bq.query(schema_sql).to_dataframe()
print(schema_df.to_string(index=False))
print()

# ── 2. Table size ────────────────────────────────────────────────────────
print("=" * 70)
print("TABLE SIZE")
print("=" * 70)

count_sql = f"SELECT COUNT(*) AS total_rows FROM `{TABLE}`"
total = bq.query(count_sql).to_dataframe().iloc[0]["total_rows"]
print(f"Total rows: {total:,}")
print()

# ── 3. Sample rows (text columns truncated) ──────────────────────────────
print("=" * 70)
print("SAMPLE ROWS (5)")
print("=" * 70)

# Get all column names, truncate long text columns for readability
cols = schema_df["column_name"].tolist()
select_exprs = []
for col in cols:
    dtype = schema_df.loc[schema_df["column_name"] == col, "data_type"].iloc[0]
    if dtype == "STRING":
        select_exprs.append(f"LEFT({col}, 200) AS {col}")
    else:
        select_exprs.append(col)

sample_sql = f"""
SELECT {', '.join(select_exprs)}
FROM `{TABLE}`
LIMIT 5
"""
sample_df = bq.query(sample_sql).to_dataframe()
for i, row in sample_df.iterrows():
    print(f"\n--- Row {i+1} ---")
    for col in cols:
        val = row[col]
        if val is not None:
            print(f"  {col}: {val}")

# ── 4. Non-null counts per column ────────────────────────────────────────
print("\n" + "=" * 70)
print("NON-NULL COUNTS PER COLUMN (approx from 10k sample)")
print("=" * 70)

null_exprs = [f"COUNTIF({col} IS NOT NULL) AS {col}" for col in cols]
null_sql = f"""
SELECT {', '.join(null_exprs)}
FROM (SELECT * FROM `{TABLE}` LIMIT 10000)
"""
null_df = bq.query(null_sql).to_dataframe()
for col in cols:
    print(f"  {col}: {null_df[col].iloc[0]:,} / 10,000")
