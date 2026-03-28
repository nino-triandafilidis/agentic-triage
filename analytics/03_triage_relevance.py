"""Assess PMC article relevance for ED triage decision support.

Takes a sample of cases from the medLLMbenchmark dataset and:
  1. Extracts chief-complaint / HPI keywords
  2. Runs vector searches against PMC for each case
  3. Measures retrieval distance distribution (proxy for relevance)
  4. Classifies retrieved articles by topic overlap with ED triage

This helps answer: "Will the PMC vector store actually return useful
articles when our RAG agents try to support triage decisions?"

Requires:
  - GCP auth (gcloud auth application-default login)
  - The embedding model created by src/config.setup()

Usage:
    python analytics/03_triage_relevance.py [--sample-size 20] [--top-k 10] [--out analytics/results]
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _gcp import get_bq_client, get_project_id

bq = None          # initialised in main() after arg parsing
EMBEDDING_MODEL = None  # set in main() once project_id is known

PUBMED_TABLE = "bigquery-public-data.pmc_open_access_commercial.articles"
USER_DATASET = "pubmed"

DEV_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "third_part_code",
    "medLLMbenchmark",
    "MIMIC-IV-Ext-Dev.csv",
)


def load_benchmark_sample(n: int) -> pd.DataFrame:
    """Load n random rows from the dev dataset."""
    df = pd.read_csv(DEV_CSV)
    if n >= len(df):
        return df
    return df.sample(n, random_state=42).reset_index(drop=True)


def build_query_from_case(row: pd.Series) -> str:
    """Build a search query from HPI + patient_info (mirrors the triage prompt)."""
    parts = []
    if pd.notna(row.get("HPI")):
        parts.append(str(row["HPI"])[:500])
    if pd.notna(row.get("patient_info")):
        parts.append(str(row["patient_info"])[:200])
    return " ".join(parts)


def vector_search(query_text: str, top_k: int = 10) -> pd.DataFrame:
    """Run a single vector search, returning articles + distances."""
    # Escape triple quotes in query text
    safe_query = query_text.replace('"""', '\\"\\"\\"').replace("\\", "\\\\")

    sql = f"""
    DECLARE query_text STRING;
    SET query_text = \"\"\"{safe_query}\"\"\";

    WITH vector_results AS (
        SELECT
            base.pmc_id          AS pmcid,
            base.pmid            AS pmid,
            base.article_citation AS journal_title,
            LEFT(base.article_text, 500) AS snippet,
            distance
        FROM VECTOR_SEARCH(
            TABLE `{PUBMED_TABLE}`,
            'ml_generate_embedding_result',
            (SELECT ml_generate_embedding_result
             FROM ML.GENERATE_EMBEDDING(
                 MODEL `{EMBEDDING_MODEL}`,
                 (SELECT query_text AS content)
             )),
            top_k => {top_k}
        )
    )
    SELECT * FROM vector_results
    ORDER BY distance
    LIMIT {top_k}
    """
    return bq.query(sql).to_dataframe()


def classify_snippet_heuristic(snippet: str) -> str:
    """Rough heuristic topic classification from article snippet."""
    snippet_lower = snippet.lower() if snippet else ""
    categories = {
        "emergency_medicine": [
            "emergency department", "emergency room", "triage",
            "acute care", "emergency medicine",
        ],
        "critical_care": [
            "intensive care", "critical care", "icu", "ventilat",
            "mechanical ventilation",
        ],
        "cardiology": [
            "cardiac", "heart", "myocardial", "coronary", "arrhythmia",
            "echocardiography",
        ],
        "pulmonology": [
            "pulmonary", "respiratory", "lung", "pneumonia", "asthma",
            "copd", "dyspnea",
        ],
        "neurology": [
            "neurolog", "stroke", "seizure", "brain", "cerebr",
            "headache",
        ],
        "infectious_disease": [
            "infect", "sepsis", "antibiotic", "bacteria", "viral",
            "pathogen",
        ],
        "surgery": [
            "surgical", "surgery", "operation", "incision",
            "laparoscop",
        ],
        "oncology": [
            "cancer", "tumor", "oncolog", "neoplasm", "malignant",
            "chemotherapy",
        ],
        "general_medicine": [
            "diabetes", "hypertension", "chronic", "primary care",
            "outpatient",
        ],
        "pediatrics": [
            "pediatric", "child", "infant", "neonatal", "adolescent",
        ],
        "pharmacology": [
            "drug", "pharmacol", "dosage", "medication", "adverse effect",
        ],
        "basic_science": [
            "cell line", "in vitro", "mouse model", "protein",
            "gene expression", "molecular",
        ],
    }
    matched = []
    for cat, keywords in categories.items():
        if any(kw in snippet_lower for kw in keywords):
            matched.append(cat)
    return ", ".join(matched) if matched else "other"


def main():
    global bq
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=None, help="GCP project ID override")
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of benchmark cases to probe")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Articles to retrieve per query")
    parser.add_argument("--out", default="analytics/results",
                        help="Output directory")
    args = parser.parse_args()
    bq = get_bq_client(args.project)
    project_id = get_project_id(args.project)
    global EMBEDDING_MODEL
    EMBEDDING_MODEL = f"{project_id}.{USER_DATASET}.textembed"

    os.makedirs(args.out, exist_ok=True)

    # ── Load benchmark cases ─────────────────────────────────────────────
    print(f"Loading {args.sample_size} benchmark cases...")
    cases = load_benchmark_sample(args.sample_size)

    # Show ESI distribution in sample
    if "acuity" in cases.columns:
        esi_col = "acuity"
    elif "ESI" in cases.columns:
        esi_col = "ESI"
    else:
        esi_col = None

    if esi_col:
        print(f"\nESI distribution in sample:")
        print(cases[esi_col].value_counts().sort_index().to_string())
        print()

    # ── Run vector searches ──────────────────────────────────────────────
    all_results = []
    distance_stats = []

    for i, row in cases.iterrows():
        query = build_query_from_case(row)
        esi = row.get(esi_col, "?") if esi_col else "?"
        print(f"[{i+1}/{len(cases)}] ESI={esi}  query={query[:80]}...")

        try:
            results = vector_search(query, top_k=args.top_k)
            results["case_idx"] = i
            results["esi_level"] = esi
            results["stay_id"] = row.get("stay_id")
            results["query_preview"] = query[:120]

            # Classify each retrieved article
            results["topic_tags"] = results["snippet"].apply(classify_snippet_heuristic)

            all_results.append(results)
            distance_stats.append({
                "case_idx": i,
                "esi_level": esi,
                "min_dist": results["distance"].min(),
                "max_dist": results["distance"].max(),
                "mean_dist": results["distance"].mean(),
                "median_dist": results["distance"].median(),
                "n_retrieved": len(results),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            distance_stats.append({
                "case_idx": i, "esi_level": esi,
                "min_dist": None, "max_dist": None,
                "mean_dist": None, "median_dist": None,
                "n_retrieved": 0,
            })

    # ── Aggregate results ────────────────────────────────────────────────
    if not all_results:
        print("No results retrieved. Check your GCP auth and embedding model.")
        return

    all_df = pd.concat(all_results, ignore_index=True)
    dist_df = pd.DataFrame(distance_stats)

    # Save raw results
    all_df.to_csv(os.path.join(args.out, "triage_retrieval_raw.csv"), index=False)
    dist_df.to_csv(os.path.join(args.out, "triage_distance_stats.csv"), index=False)

    # ── Distance analysis ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RETRIEVAL DISTANCE STATISTICS")
    print("=" * 70)
    print(f"  Mean closest distance:  {dist_df['min_dist'].mean():.4f}")
    print(f"  Mean average distance:  {dist_df['mean_dist'].mean():.4f}")
    print(f"  Mean farthest distance: {dist_df['max_dist'].mean():.4f}")
    print()

    if esi_col:
        print("Distance by ESI level (lower = more relevant):")
        esi_dist = dist_df.groupby("esi_level").agg(
            mean_closest=("min_dist", "mean"),
            mean_avg=("mean_dist", "mean"),
            n_cases=("case_idx", "count"),
        ).round(4)
        print(esi_dist.to_string())
        print()

    # ── Topic distribution of retrieved articles ─────────────────────────
    print("=" * 70)
    print("TOPIC DISTRIBUTION OF RETRIEVED ARTICLES")
    print("=" * 70)

    # Explode multi-tag rows
    tag_counts = {}
    for tags in all_df["topic_tags"]:
        for tag in tags.split(", "):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    tag_df = pd.DataFrame(
        sorted(tag_counts.items(), key=lambda x: -x[1]),
        columns=["topic", "count"],
    )
    tag_df["pct"] = (tag_df["count"] / len(all_df) * 100).round(1)
    print(tag_df.to_string(index=False))
    tag_df.to_csv(os.path.join(args.out, "triage_topic_distribution.csv"), index=False)
    print()

    # ── Journal distribution in retrieved articles ───────────────────────
    print("=" * 70)
    print("TOP JOURNALS IN RETRIEVED ARTICLES")
    print("=" * 70)
    journal_counts = all_df["journal_title"].value_counts().head(20)
    print(journal_counts.to_string())
    print()

    # ── ED-relevance score ───────────────────────────────────────────────
    ed_tags = {"emergency_medicine", "critical_care"}
    ed_relevant = all_df["topic_tags"].apply(
        lambda t: any(tag in ed_tags for tag in t.split(", "))
    )
    ed_pct = ed_relevant.mean() * 100

    clinical_tags = {"cardiology", "pulmonology", "neurology",
                     "infectious_disease", "surgery", "general_medicine",
                     "pediatrics", "pharmacology"}
    clinically_relevant = all_df["topic_tags"].apply(
        lambda t: any(tag in (ed_tags | clinical_tags) for tag in t.split(", "))
    )
    clinical_pct = clinically_relevant.mean() * 100

    print("=" * 70)
    print("RELEVANCE SUMMARY")
    print("=" * 70)
    print(f"  Directly ED-relevant (EM + critical care): {ed_pct:.1f}%")
    print(f"  Clinically relevant (any clinical domain):  {clinical_pct:.1f}%")
    print(f"  Basic science / other:                      {100 - clinical_pct:.1f}%")
    print()

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "sample_size": args.sample_size,
        "top_k": args.top_k,
        "total_retrieved": len(all_df),
        "distance": {
            "mean_closest": round(dist_df["min_dist"].mean(), 4),
            "mean_average": round(dist_df["mean_dist"].mean(), 4),
        },
        "relevance": {
            "ed_relevant_pct": round(ed_pct, 1),
            "clinically_relevant_pct": round(clinical_pct, 1),
            "basic_science_other_pct": round(100 - clinical_pct, 1),
        },
        "top_topics": tag_df.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(args.out, "triage_relevance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
