"""One-time correctness check: compare FAISS results against BQ VECTOR_SEARCH.

Run with:
    .venv/bin/python scripts/validate_faiss_vs_bq.py

Tests several nprobe values (32, 64, 128, 256) and reports:
  - Recall@10 (fraction of BQ ground-truth results recovered)
  - Jaccard similarity (for reference)
  - Distance correlation (Spearman)
  - Latency per query

Recommends an nprobe setting based on recall@10/latency tradeoff.

Estimated BQ cost: ~$0.50 (5 queries × 1 call each × ~$0.10/call)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402

CANONICAL_QUERIES = [
    "chest pain triage emergency department",
    "sepsis fever early recognition",
    "stroke neurological assessment acute",
    "trauma injury severity score",
    "pediatric respiratory distress",
]

NPROBE_VALUES = [32, 64, 128, 256]
TOP_K = 10


def _search_bq(query: str) -> pd.DataFrame:
    """Run BQ VECTOR_SEARCH."""
    from src.rag.retrieval import _search_bq
    return _search_bq(query, top_k=TOP_K)


def _search_faiss(query: str, nprobe: int) -> pd.DataFrame:
    """Run FAISS search with a specific nprobe."""
    from src.rag.retrieval import _load_faiss_index, _embed_query
    index, pmc_ids_df, _articles_db = _load_faiss_index()

    old_nprobe = index.nprobe
    index.nprobe = nprobe

    query_vec = _embed_query(query)
    t0 = time.time()
    similarities, indices = index.search(query_vec, TOP_K)
    latency = time.time() - t0

    index.nprobe = old_nprobe

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        pmc_id = pmc_ids_df.iloc[idx]["pmc_id"]
        distance = 1.0 - float(sim)
        results.append({"pmc_id": pmc_id, "distance": distance})

    return pd.DataFrame(results), latency


def compute_recall(bq_df: pd.DataFrame, faiss_df: pd.DataFrame) -> float:
    """Recall@K: fraction of BQ ground-truth results recovered by FAISS."""
    bq_set = set(bq_df["pmc_id"])
    faiss_set = set(faiss_df["pmc_id"])
    if not bq_set:
        return 1.0
    return len(bq_set & faiss_set) / len(bq_set)


def compute_jaccard(bq_df: pd.DataFrame, faiss_df: pd.DataFrame) -> float:
    """Jaccard similarity of top-k pmc_id sets (for reference)."""
    bq_set = set(bq_df["pmc_id"])
    faiss_set = set(faiss_df["pmc_id"])
    if not bq_set and not faiss_set:
        return 1.0
    return len(bq_set & faiss_set) / len(bq_set | faiss_set)


def compute_distance_correlation(bq_df: pd.DataFrame, faiss_df: pd.DataFrame) -> float:
    """Spearman correlation of distances for common pmc_ids."""
    common = set(bq_df["pmc_id"]) & set(faiss_df["pmc_id"])
    if len(common) < 3:
        return float("nan")

    bq_dists = bq_df.set_index("pmc_id").loc[list(common), "distance"]
    faiss_dists = faiss_df.set_index("pmc_id").loc[list(common), "distance"]
    # Align indices
    common_sorted = sorted(common)
    bq_vals = bq_dists.loc[common_sorted].values
    faiss_vals = faiss_dists.loc[common_sorted].values
    corr, _ = stats.spearmanr(bq_vals, faiss_vals)
    return corr


def main() -> None:
    print("=" * 70)
    print("FAISS vs BigQuery Validation")
    print(f"Queries  : {len(CANONICAL_QUERIES)}")
    print(f"nprobe   : {NPROBE_VALUES}")
    print(f"top_k    : {TOP_K}")
    print("=" * 70)

    # Init both backends
    config.setup_clients(force_bq=True)

    # Get BQ ground truth for all queries
    print("\n--- BQ ground truth ---")
    bq_results = {}
    for q in CANONICAL_QUERIES:
        print(f"  BQ: {q[:50]}… ", end="", flush=True)
        t0 = time.time()
        bq_results[q] = _search_bq(q)
        bq_lat = time.time() - t0
        print(f"({len(bq_results[q])} results, {bq_lat:.2f}s)")

    # Test each nprobe value
    print("\n--- FAISS nprobe sweep ---")
    summary_rows = []

    for nprobe in NPROBE_VALUES:
        print(f"\n  nprobe = {nprobe}")
        recalls = []
        jaccards = []
        correlations = []
        latencies = []

        for q in CANONICAL_QUERIES:
            faiss_df, lat = _search_faiss(q, nprobe)
            bq_df = bq_results[q]

            recall = compute_recall(bq_df, faiss_df)
            jaccard = compute_jaccard(bq_df, faiss_df)
            corr = compute_distance_correlation(bq_df, faiss_df)

            recalls.append(recall)
            jaccards.append(jaccard)
            correlations.append(corr)
            latencies.append(lat)

            print(f"    {q[:40]:40s}  recall={recall:.2f}  jaccard={jaccard:.2f}  corr={corr:+.3f}  lat={lat*1000:.0f}ms")

        mean_recall = np.mean(recalls)
        mean_jaccard = np.mean(jaccards)
        mean_corr = np.nanmean(correlations)
        mean_lat = np.mean(latencies) * 1000

        summary_rows.append({
            "nprobe": nprobe,
            "mean_recall": mean_recall,
            "mean_jaccard": mean_jaccard,
            "mean_corr": mean_corr,
            "mean_latency_ms": mean_lat,
        })

        print(f"    --- MEAN: recall={mean_recall:.3f}  jaccard={mean_jaccard:.3f}  corr={mean_corr:+.3f}  lat={mean_lat:.0f}ms")

    # Summary and recommendation
    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    # Recommend: first nprobe with recall >= 0.9, or the highest recall
    good = summary[summary["mean_recall"] >= 0.9]
    if not good.empty:
        rec = good.iloc[0]
        print(f"\nRecommendation: nprobe = {int(rec['nprobe'])}")
        print(f"  (recall={rec['mean_recall']:.3f}, latency={rec['mean_latency_ms']:.0f}ms)")
    else:
        rec = summary.sort_values("mean_recall", ascending=False).iloc[0]
        print(f"\nBest available: nprobe = {int(rec['nprobe'])}")
        print(f"  (recall={rec['mean_recall']:.3f}, latency={rec['mean_latency_ms']:.0f}ms)")
        print("  WARNING: No nprobe achieved ≥90% recall. Consider increasing nlist or using brute-force.")

    print(f"\nTo apply: set FAISS_NPROBE={int(rec['nprobe'])} in environment or config.py")


if __name__ == "__main__":
    main()
