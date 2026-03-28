"""One-time validation of pmc_embeddings. Run once after provisioning.

Three checks:

  Check 1 — Row counts (free)
    Compares row counts between the public table and our owned copy.
    Two sub-checks:
      (a) Full-copy mode (no QUALIFY): own_n == pub_n.
      (b) Dedup mode (QUALIFY applied): own_n == pub_distinct_n, where
          pub_distinct_n is COUNT(DISTINCT pmc_id) in the public table.
          QUALIFY keeps exactly one row per distinct pmc_id (the one with
          the highest pmid DESC), so owned_n must equal pub_distinct_n.
          Note: pub_pmid0_n (total pmid='0' rows) is NOT the right metric
          because many pmid='0' rows are singletons (no real-PMID counterpart)
          that QUALIFY keeps rather than drops.
    Also reports duplicate pmc_id count in owned table as a sanity check
    (should be 0 after dedup, >0 after full copy).

  Check 2 — Embedding vectors (~$0.14) + coverage check (free)
    Part A — embedding integrity:
      Hash-samples ~200 owned rows, JOINs on (pmc_id, pmid), and tests
      MIN(ML.DISTANCE(..., 'EUCLIDEAN')) per group.  JOIN is inner, so rows
      absent from owned are excluded rather than flagged — this is by design
      for the dedup case (pmid=0 rows are intentionally absent).
    Part B — coverage check (reads only pmc_id + pmid, no embeddings, free):
      LEFT JOINs public onto owned on (pmc_id, pmid) and counts any
      public rows with a real PMID (pmid != '0') that are absent from owned.
      Expected: 0 in both full-copy and dedup modes.  A non-zero count means
      real articles were silently dropped beyond the expected pmid=0 removal.

  Check 3 — VECTOR_SEARCH retrieval consistency (~$0.07/query × N_QUERIES)
    Runs the same query against both tables and reports top-1 match (PASS/FAIL)
    and top-10 overlap (diagnostic).
    Important: the public table has no vector index
    (INFORMATION_SCHEMA.VECTOR_INDEXES returns no rows for it — it is
    read-only and Google has not indexed it).  The public side therefore
    always executes as a brute-force exact scan (~$0.07/query).  The owned
    table uses the IVF index with fraction_lists_to_search=0.1 (same setting
    as production retrieval.py) so this check tests real production recall,
    not a theoretical best-case.

Total estimated cost: ~$0.35
  Check 1: free (row counts + pmc_id/pmid scans only)
  Check 2: ~$0.14 embedding scan + free coverage check
  Check 3: ~$0.07 × N_QUERIES (public brute-force dominates)

Usage:
    .venv/bin/python scripts/validate_retrieval.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from google.cloud import bigquery

import src.config as config

# Manifest written after every successful validation; read on next run to detect
# whether the public table has changed since provisioning.
MANIFEST_PATH = ROOT / "data" / "vector_store_manifest.json"


def _load_manifest() -> dict | None:
    """Return the saved manifest dict, or None if it does not exist yet."""
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _save_manifest(pub_n: int, pub_distinct_n: int, own_n: int) -> None:
    """Persist current table counts so the next run can detect public-table drift."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(
            {
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "pub_n": pub_n,
                "pub_distinct_n": pub_distinct_n,
                "own_n": own_n,
            },
            f,
            indent=2,
        )

SAMPLE_MODULO    = 10_000   # ~230 sampled rows out of 2.3M
DISTANCE_EPSILON = 1e-6     # float round-trip noise threshold
TOP_K = 10

QUERIES = [
    "chest pain triage emergency department",
    "stroke neurological deficit acute onset",
    "sepsis fever hypotension infection workup",
]


def _confirm(msg: str) -> None:
    ans = input(f"\n{msg}\nProceed? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted.")
        sys.exit(0)


# ── Check 1: row counts ────────────────────────────────────────────────
def check_row_counts() -> tuple[int, int, int, int, int]:
    """Return (pub_n, own_n, own_dup_n, pub_pmid0_n, pub_distinct_n).

    pub_n          — public table rows with non-null embedding (reference total)
    own_n          — owned table row count
    own_dup_n      — duplicate pmc_id rows in owned (0 after dedup, >0 after full copy)
    pub_pmid0_n    — pmid='0' rows in public (informational; NOT the QUALIFY drop
                     count — see note below)
    pub_distinct_n — COUNT(DISTINCT pmc_id) in public; after QUALIFY dedup, owned
                     should have exactly this many rows.  pub_n - pub_distinct_n is
                     the actual number of rows QUALIFY drops.

    Note on pub_pmid0_n vs pub_distinct_n: pub_pmid0_n counts ALL pmid='0' rows,
    including singletons — pmc_ids that have a pmid='0' row but no real-PMID row
    yet.  QUALIFY ORDER BY pmid DESC keeps singletons (they rank first as the only
    row) and only drops pmid='0' rows that have a real-PMID counterpart.  Therefore
    pub_pmid0_n > (pub_n - pub_distinct_n) when singletons exist.
    """
    pub_sql = (
        f"SELECT COUNT(*) AS n FROM `{config.PUBMED_TABLE}`"
        f" WHERE ml_generate_embedding_result IS NOT NULL"
    )
    own_sql  = f"SELECT COUNT(*) AS n FROM `{config.PMC_EMBEDDINGS_TABLE}`"
    dup_sql  = (
        f"SELECT COUNT(*) - COUNT(DISTINCT pmc_id) AS dup"
        f" FROM `{config.PMC_EMBEDDINGS_TABLE}`"
    )
    # Informational only — total pmid='0' rows in public.  IS NOT NULL has no
    # effect in practice (all rows have embeddings; confirmed: no_embedding=0).
    pmid0_sql = (
        f"SELECT COUNT(*) AS n FROM `{config.PUBMED_TABLE}`"
        f" WHERE pmid = '0' AND ml_generate_embedding_result IS NOT NULL"
    )
    # The correct dedup check metric: QUALIFY keeps exactly one row per distinct
    # pmc_id, so owned_n must equal COUNT(DISTINCT pmc_id) from public.
    distinct_sql = (
        f"SELECT COUNT(DISTINCT pmc_id) AS n FROM `{config.PUBMED_TABLE}`"
        f" WHERE ml_generate_embedding_result IS NOT NULL"
    )
    pub_n       = int(config.bq_client.query(pub_sql).to_dataframe()["n"].iloc[0])
    own_n       = int(config.bq_client.query(own_sql).to_dataframe()["n"].iloc[0])
    own_dup_n   = int(config.bq_client.query(dup_sql).to_dataframe()["dup"].iloc[0])
    pub_pmid0_n = int(config.bq_client.query(pmid0_sql).to_dataframe()["n"].iloc[0])
    pub_distinct_n = int(config.bq_client.query(distinct_sql).to_dataframe()["n"].iloc[0])
    return pub_n, own_n, own_dup_n, pub_pmid0_n, pub_distinct_n


# ── Check 2a: embedding vector integrity ──────────────────────────────
def check_embeddings() -> dict:
    # Strategy: JOIN on (pmc_id, pmid) — inner join — so rows absent from owned
    # are excluded rather than flagged.  This is correct for the dedup case:
    # pmid=0 rows are intentionally absent and should not be compared.
    # For the full-copy case, all pub rows have a match in owned so nothing is
    # silently skipped.
    # GROUP BY (pmc_id, pmid) handles the rare edge case of two pmid=0 rows
    # for the same pmc_id in the public table (takes MIN dist per group).
    sql = f"""
    SELECT
        COUNT(*)                               AS rows_checked,
        COUNTIF(min_dist > {DISTANCE_EPSILON}) AS corrupted_rows,
        MAX(min_dist)                          AS max_distance,
        AVG(min_dist)                          AS avg_distance
    FROM (
        SELECT
            pub.pmc_id,
            pub.pmid,
            MIN(ML.DISTANCE(
                pub.ml_generate_embedding_result,
                own.ml_generate_embedding_result,
                'EUCLIDEAN'
            )) AS min_dist
        FROM `{config.PUBMED_TABLE}` pub
        JOIN `{config.PMC_EMBEDDINGS_TABLE}` own
            ON pub.pmc_id = own.pmc_id AND pub.pmid = own.pmid
        WHERE MOD(ABS(FARM_FINGERPRINT(pub.pmc_id)), {SAMPLE_MODULO}) = 0
          AND pub.ml_generate_embedding_result IS NOT NULL
        GROUP BY pub.pmc_id, pub.pmid
    )
    """
    row = config.bq_client.query(sql).to_dataframe().iloc[0]
    return {
        "rows_checked":   int(row["rows_checked"]),
        "corrupted_rows": int(row["corrupted_rows"]),
        "max_distance":   float(row["max_distance"] or 0.0),
        "avg_distance":   float(row["avg_distance"] or 0.0),
    }


# ── Check 2b: coverage — are any articles entirely missing from owned? ──
def check_coverage() -> tuple[int, int]:
    """Return (entirely_absent, alternate_pmid_rows).

    The PASS/FAIL metric is entirely_absent: distinct pmc_ids with at least one
    real-PMID row in public that have NO row at all in owned.  Expected: 0.

    alternate_pmid_rows is informational: (pmc_id, pmid) pairs in public that are
    absent from owned by exact match, yet the pmc_id IS present in owned with a
    different pmid.  This arises when the public table has two real-PMID rows for
    the same pmc_id (data quality issue in the source — PMID corrections or
    reassignments); QUALIFY ORDER BY pmid DESC keeps one and drops the other.
    Observed: ~23 rows (Feb 2026).  Not data loss — the article is present.

    Note: IS NOT NULL has no effect (all rows have embeddings; no_embedding=0).
    Kept for explicit intent in case this changes.
    """
    # PASS/FAIL: pmc_ids entirely absent from owned (join on pmc_id alone)
    absent_sql = f"""
    SELECT COUNT(*) AS n FROM (
      SELECT DISTINCT pub.pmc_id
      FROM `{config.PUBMED_TABLE}` pub
      LEFT JOIN `{config.PMC_EMBEDDINGS_TABLE}` own ON pub.pmc_id = own.pmc_id
      WHERE pub.pmid != '0'
        AND pub.ml_generate_embedding_result IS NOT NULL
        AND own.pmc_id IS NULL
    )
    """
    # INFO: (pmc_id, pmid) pairs missing — superset of above; difference = alternate PMIDs
    pair_sql = f"""
    SELECT COUNT(*) AS n
    FROM `{config.PUBMED_TABLE}` pub
    LEFT JOIN `{config.PMC_EMBEDDINGS_TABLE}` own
        ON pub.pmc_id = own.pmc_id AND pub.pmid = own.pmid
    WHERE pub.pmid != '0'
      AND pub.ml_generate_embedding_result IS NOT NULL
      AND own.pmc_id IS NULL
    """
    entirely_absent   = int(config.bq_client.query(absent_sql).to_dataframe().iloc[0]["n"])
    alternate_pmid_rows = int(config.bq_client.query(pair_sql).to_dataframe().iloc[0]["n"])
    return entirely_absent, alternate_pmid_rows


# ── Check 3: VECTOR_SEARCH retrieval consistency ───────────────────────
def _vector_search_top_ids(table: str, query_text: str) -> list[str]:
    # Note on asymmetry: the public table has NO vector index.
    # INFORMATION_SCHEMA.VECTOR_INDEXES returns zero rows for it because it is
    # a read-only Google-managed table.  Every VECTOR_SEARCH call against the
    # public table therefore falls back to a full brute-force scan of the
    # embedding column (~14 GB, ~$0.07/call) regardless of any options passed.
    # The owned table uses the IVF index with fraction_lists_to_search=0.1
    # (same as production retrieval.py), so Check 3 measures real production
    # recall against exact search — not ideal vs. ideal.
    sql = f"""
    SELECT base.pmc_id, distance
    FROM VECTOR_SEARCH(
        TABLE `{table}`,
        'ml_generate_embedding_result',
        (SELECT ml_generate_embedding_result
         FROM ML.GENERATE_EMBEDDING(
             MODEL `{config.EMBEDDING_MODEL}`,
             (SELECT @query AS content)
         )),
        top_k => {TOP_K},
        -- fraction_lists_to_search only takes effect when the table has an IVF
        -- index.  For the public table this option is silently ignored and
        -- brute-force is used.  For owned table it sets recall vs. cost tradeoff.
        options => '{{"fraction_lists_to_search": 0.1}}'
    )
    ORDER BY distance
    LIMIT {TOP_K}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("query", "STRING", query_text)]
    )
    df = config.bq_client.query(sql, job_config=job_config).to_dataframe()
    return list(df["pmc_id"])


def check_vector_search() -> list[dict]:
    results = []
    for query in QUERIES:
        pub_ids = _vector_search_top_ids(config.PUBMED_TABLE, query)
        own_ids = _vector_search_top_ids(config.PMC_EMBEDDINGS_TABLE, query)
        overlap = len(set(pub_ids) & set(own_ids))
        top1_ok = bool(pub_ids and own_ids and pub_ids[0] == own_ids[0])
        results.append({
            "query":    query,
            "top1_ok":  top1_ok,
            "overlap":  overlap,
            "pub_top1": pub_ids[0] if pub_ids else "—",
            "own_top1": own_ids[0] if own_ids else "—",
        })
    return results


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    n = len(QUERIES)
    _confirm(
        f"Validation plan (run once after provisioning):\n"
        f"  Check 1 — row counts + dedup verification   free\n"
        f"  Check 2 — embedding vectors                  ~$0.14  (reads both embedding columns)\n"
        f"          — coverage check                     free    (pmc_id + pmid only)\n"
        f"  Check 3 — VECTOR_SEARCH consistency          ~${0.07 * n:.2f}  ({n} queries × ~$0.07 public brute-force)\n"
        f"  Total estimated cost: ~${0.14 + 0.07 * n:.2f}"
    )

    config.setup_clients()
    passed = True

    # ── Check 1 ──────────────────────────────────────────────────────
    print("\n[Check 1] Row counts (free)")
    pub_n, own_n, own_dup_n, pub_pmid0_n, pub_distinct_n = check_row_counts()
    qualify_dropped = pub_n - pub_distinct_n  # rows QUALIFY actually drops

    # Public-table drift check: compare current pub_n against the baseline saved
    # by the previous successful validation run.  A change in pub_n means Google
    # updated the public table; the owned table is now a stale snapshot and
    # re-provisioning should be considered.
    manifest = _load_manifest()
    if manifest:
        delta = pub_n - manifest["pub_n"]
        if delta == 0:
            print(
                f"  INFO  public table stable since {manifest['validated_at']} "
                f"({pub_n:,} rows unchanged)"
            )
        else:
            print(
                f"  WARN  public table changed since {manifest['validated_at']}: "
                f"pub_n was {manifest['pub_n']:,}, now {pub_n:,} ({delta:+,} rows). "
                f"Owned table is a stale snapshot — re-provisioning recommended."
            )
    else:
        print("  INFO  no manifest found — first validation run; baseline will be saved on PASS")
    pmid0_singletons = pub_pmid0_n - qualify_dropped  # pmid='0' rows kept as singletons

    if own_n == pub_n:
        # Full-copy mode: every public row (including pmid=0 duplicates) was copied.
        print(f"  PASS  public={pub_n:,}  owned={own_n:,}  (faithful full copy, duplicates included)")
    elif own_dup_n == 0 and own_n < pub_n:
        # Dedup mode: QUALIFY keeps one row per distinct pmc_id, so own_n must
        # equal pub_distinct_n.  pub_pmid0_n is NOT the right metric because
        # singleton pmid='0' rows (no real-PMID counterpart) are kept, not dropped.
        if own_n == pub_distinct_n:
            print(
                f"  PASS  public={pub_n:,}  owned={own_n:,}  "
                f"({qualify_dropped:,} rows dropped by QUALIFY; "
                f"pub_distinct_pmc_ids={pub_distinct_n:,} ✓)"
            )
        else:
            diff = pub_distinct_n - own_n
            print(
                f"  WARN  public={pub_n:,}  pub_distinct_pmc_ids={pub_distinct_n:,}  "
                f"owned={own_n:,}  "
                f"← {abs(diff):,} rows {'missing from' if diff > 0 else 'extra in'} owned"
            )
            passed = False
    else:
        diff = pub_n - own_n
        print(f"  WARN  public={pub_n:,}  owned={own_n:,}  ← {abs(diff):,} rows {'missing' if diff > 0 else 'extra'}")
        passed = False

    print(
        f"  INFO  duplicate pmc_ids in owned: {own_dup_n:,}\n"
        f"        pmid='0' rows in public: {pub_pmid0_n:,}  "
        f"({qualify_dropped:,} dropped by QUALIFY, {pmid0_singletons:,} singletons kept)\n"
        f"        (public table inserts a pmid='0' row on deposit, then a real-PMID row\n"
        f"         when NLM assigns the PMID; rows with no real-PMID counterpart yet\n"
        f"         are kept as singletons by QUALIFY ORDER BY pmid DESC)"
    )

    # ── Check 2 ──────────────────────────────────────────────────────
    print("\n[Check 2] Embedding vectors (~$0.14) + coverage check (free)")

    # Part A: embedding integrity (sampled, inner join — missing rows excluded by design)
    r = check_embeddings()
    ok2 = r["corrupted_rows"] == 0
    print(
        f"  {'PASS' if ok2 else 'FAIL'}  embedding integrity  "
        f"rows_checked={r['rows_checked']:,}  corrupted={r['corrupted_rows']}  "
        f"max_dist={r['max_distance']:.2e}  avg_dist={r['avg_distance']:.2e}"
    )
    if not ok2:
        print(f"  FAIL: {r['corrupted_rows']} sampled row(s) have mismatched embedding vectors.")
        passed = False

    # Part B: coverage — are any articles (pmc_ids) entirely absent from owned?
    # PASS/FAIL on entirely_absent; alternate_pmid_rows is informational.
    entirely_absent, alternate_pmid_rows = check_coverage()
    ok_cov = entirely_absent == 0
    print(
        f"  {'PASS' if ok_cov else 'FAIL'}  coverage check       "
        f"articles_entirely_absent={entirely_absent}"
        + ("" if ok_cov else "  ← pmc_ids with no row at all in owned")
    )
    if alternate_pmid_rows > 0:
        # pmc_id IS in owned but with a different pmid — the public table has two
        # real-PMID rows for the same article (PMID correction or reassignment).
        # QUALIFY ORDER BY pmid DESC kept the lexicographically higher one.
        # This is not data loss: the article is present, just the alternate PMID row
        # (lower pmid) was dropped.  Observed: ~23 rows (Feb 2026).
        print(
            f"  INFO  alternate_pmid_rows={alternate_pmid_rows}  "
            f"(pmc_id present in owned with a different pmid — PMID corrections "
            f"in the public table; QUALIFY kept the lexicographically higher pmid)"
        )
    if not ok_cov:
        passed = False

    # ── Check 3 ──────────────────────────────────────────────────────
    # PASS/FAIL is binary top-1 match.
    # top10_overlap is diagnostic: measures how many of the top-10 result sets
    # intersect between public (exact brute-force) and owned (IVF, recall ~95%).
    # Public table: no vector index → always brute-force exact search (~$0.07/call).
    # Owned table:  IVF index with fraction_lists_to_search=0.1 → approximate.
    print(f"\n[Check 3] VECTOR_SEARCH retrieval consistency (~${0.07 * n:.2f})")
    print(f"  NOTE  public side = brute-force exact (no index); owned side = IVF approximate")
    top1_failures = 0
    for row in check_vector_search():
        ok3 = row["top1_ok"]
        status = "PASS" if ok3 else "FAIL"
        print(
            f"  {status}  top1_match={'yes' if ok3 else 'no '}  "
            f"top10_overlap={row['overlap']}/{TOP_K}  "
            f"pub_top1={row['pub_top1']}  own_top1={row['own_top1']}"
        )
        print(f"       query: {row['query']!r}")
        if not ok3:
            top1_failures += 1
    if top1_failures:
        print(
            f"  WARN: {top1_failures} query/queries had top-1 mismatch "
            f"(IVF approximation — expected occasionally at fraction_lists_to_search=0.1)."
        )

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    if passed:
        print("[DONE] Core checks passed.")
        if top1_failures:
            print("       VECTOR_SEARCH top-1 differed on some queries — IVF approximation.")
        _save_manifest(pub_n, pub_distinct_n, own_n)
        print(f"       Manifest saved to {MANIFEST_PATH} — next run will detect public-table drift.")
    else:
        print("[FAIL] One or more checks failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
