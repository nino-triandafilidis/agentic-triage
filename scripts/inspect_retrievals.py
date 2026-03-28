"""Inspect top-k retrieved articles for the first N cases of scratch.csv.

Retrieval only — no LLM calls, no Gemini cost/rate limits.
Logs full results to a JSONL file (one JSON object per case).

Usage:
    .venv/bin/python scripts/inspect_retrievals.py [--n-rows 10] [--top-k 5]
"""

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import src.config as config
from src.rag.retrieval import search_pubmed_articles

DATA_PATH = ROOT / "data" / "splits" / "scratch.csv"
CACHE_PATH = ROOT / "data" / "cache" / "retrieval_cache.parquet"
LOG_DIR = ROOT / "data" / "runs"


def extract_title(citation: str | None) -> str:
    """Best-effort title extraction from article_citation."""
    if not citation:
        return "(no citation)"
    return citation[:120].rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config.setup_clients()

    # Load retrieval cache (same logic as run_rag_triage.py)
    retrieval_cache: dict[str, pd.DataFrame] = {}
    if CACHE_PATH.exists():
        raw_cache = pd.read_parquet(CACHE_PATH)
        raw_cache = raw_cache[raw_cache["top_k"] >= args.top_k]
        cols = ["pmc_id", "pmid", "distance", "article_text", "article_citation"]
        for qhash, group in raw_cache.groupby("query_hash"):
            retrieval_cache[qhash] = group[cols].head(args.top_k).reset_index(drop=True)
        print(f"Retrieval cache loaded: {len(retrieval_cache)} queries")
    else:
        print("No retrieval cache found — all queries will hit FAISS + Vertex AI")

    df = pd.read_csv(DATA_PATH, nrows=args.n_rows)
    print(f"Loaded {len(df)} cases from scratch.csv\n")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"retrieval_inspection_{ts}.jsonl"

    cache_hits = 0
    with open(log_path, "w") as log_f:
        for idx, row in df.iterrows():
            hpi = str(row.get("HPI", ""))
            patient_info = str(row.get("patient_info", ""))
            initial_vitals = str(row.get("initial_vitals", ""))
            chief_complaint = str(row.get("chiefcomplaint", ""))
            ground_truth = row.get("triage", "?")

            query_text = f"{hpi} {patient_info} {initial_vitals}".strip()
            qhash = hashlib.sha256(query_text.encode()).hexdigest()

            if qhash in retrieval_cache:
                articles = retrieval_cache[qhash]
                from_cache = True
                cache_hits += 1
            else:
                articles = search_pubmed_articles(query_text, top_k=args.top_k)
                from_cache = False

            # Build log record
            retrieved = []
            for rank, (_, art) in enumerate(articles.iterrows(), 1):
                retrieved.append({
                    "rank": rank,
                    "pmc_id": str(art.get("pmc_id", "")),
                    "pmid": str(art.get("pmid", "")),
                    "distance": float(art["distance"]) if pd.notna(art.get("distance")) else None,
                    "citation": str(art.get("article_citation", "")),
                    "article_text": str(art.get("article_text", "")),
                })

            record = {
                "case_index": int(idx),
                "stay_id": int(row["stay_id"]),
                "chief_complaint": chief_complaint,
                "ground_truth_esi": int(ground_truth) if pd.notna(ground_truth) else None,
                "query": query_text,
                "from_cache": from_cache,
                "top_k": args.top_k,
                "articles": retrieved,
            }
            log_f.write(json.dumps(record) + "\n")

            # Console output (truncated for readability)
            print("=" * 80)
            print(f"CASE {idx + 1}  |  stay_id={row['stay_id']}  |  "
                  f"chief_complaint={chief_complaint}  |  ground_truth_ESI={ground_truth}")
            print("-" * 80)

            for art_rec in retrieved:
                snippet_display = textwrap.shorten(art_rec["article_text"], width=200, placeholder="...")
                print(f"\n  [{art_rec['rank']}] PMID={art_rec['pmid']}  PMC={art_rec['pmc_id']}  "
                      f"distance={art_rec['distance']:.4f}")
                print(f"      Citation: {extract_title(art_rec['citation'])}")
                print(f"      Snippet:  {snippet_display}")

            print()

    print(f"Done — inspected {len(df)} cases x {args.top_k} articles each.")
    print(f"Cache hits: {cache_hits}/{len(df)}")
    print(f"Full log: {log_path}")


if __name__ == "__main__":
    main()
