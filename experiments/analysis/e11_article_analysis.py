"""Deep-dive analysis of E11 distance-gated retrieval: articles behind correct vs incorrect RAG predictions.

Joins E11 post-hoc analysis (gating decisions + predictions) with E07 per-row diagnostics
(retrieved PMC IDs) and the retrieval cache (article text + citations) to examine what
the model was reading when it got ESI-3 cases right vs wrong.

Usage:
    .venv/bin/python experiments/analysis/e11_article_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Load data sources ──────────────────────────────────────────────────

POSTHOC_CSV = ROOT / "experiments" / "analysis" / "e11_posthoc_analysis.csv"
# Use E07 top-10 diagnostics (superset of top-5; rank 1-5 = same as top-5 run)
DIAG_JSONL = ROOT / "data" / "runs" / "E07_8k_strip_top10" / "E07_8k_strip_top10_strip_20260305_070703.diagnostics.jsonl"
E07_CSV = ROOT / "data" / "runs" / "E07_8k_strip_top10" / "E07_8k_strip_top10_strip_20260305_070703.csv"
CACHE_PARQUET = ROOT / "data" / "cache" / "retrieval_cache.parquet"
OUTPUT_MD = ROOT / "experiments" / "analysis" / "e11_article_deep_dive.md"


def load_diagnostics(path: Path) -> dict[int, list[dict]]:
    """Load diagnostics JSONL → {stay_id: [article_stats]}."""
    result = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            result[row["stay_id"]] = row["article_stats"]
    return result


def load_cache(path: Path) -> pd.DataFrame:
    """Load retrieval cache with article text and citations."""
    df = pd.read_parquet(path)
    # Deduplicate on pmc_id (same article may appear for different queries)
    return df.drop_duplicates(subset=["pmc_id"]).set_index("pmc_id")


def truncate(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    # Load all data
    posthoc = pd.read_csv(POSTHOC_CSV)
    e07 = pd.read_csv(E07_CSV)
    diagnostics = load_diagnostics(DIAG_JSONL)
    cache = load_cache(CACHE_PARQUET)

    print(f"Loaded: {len(posthoc)} posthoc rows, {len(diagnostics)} diagnostics, {len(cache)} cached articles")

    # Merge posthoc with E07 patient data (chief complaint, HPI)
    merged = posthoc.merge(
        e07[["stay_id", "chiefcomplaint", "HPI", "patient_info", "initial_vitals"]],
        on="stay_id",
        how="left",
    )

    # ── Focus groups ──────────────────────────────────────────────────

    # Group 1: ESI-3 cases where RAG was used (top1 <= 0.25)
    esi3_rag = merged[(merged["triage"] == 3) & (merged["used_rag"] == True)]
    esi3_rag_correct = esi3_rag[esi3_rag["triage_RAG"] == 3]
    esi3_rag_wrong = esi3_rag[esi3_rag["triage_RAG"] != 3]

    # Group 2: ESI-2 cases where RAG was used
    esi2_rag = merged[(merged["triage"] == 2) & (merged["used_rag"] == True)]
    esi2_rag_correct = esi2_rag[esi2_rag["triage_RAG"] == 2]
    esi2_rag_wrong = esi2_rag[esi2_rag["triage_RAG"] != 2]

    # Group 3: All RAG-used cases — correct vs wrong
    all_rag = merged[merged["used_rag"] == True]
    all_rag_correct = all_rag[all_rag["triage_RAG"] == all_rag["triage"]]
    all_rag_wrong = all_rag[all_rag["triage_RAG"] != all_rag["triage"]]

    # ── Build markdown report ─────────────────────────────────────────

    lines = []
    lines.append("# E11 Post-Hoc Deep Dive: Retrieved Articles Analysis")
    lines.append("")
    lines.append("Examines the PMC articles retrieved for cases where RAG predictions were")
    lines.append("correct vs incorrect, focusing on ESI-3 cases (90.5% over-triage rate).")
    lines.append("")
    lines.append("Data sources:")
    lines.append("- E11 post-hoc analysis (gating at threshold=0.25)")
    lines.append("- E07 top-10 retrieval diagnostics (PMC IDs per row)")
    lines.append("- Retrieval cache (article text + citations)")
    lines.append("")

    # ── Section 1: Aggregate article statistics ───────────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 1. Aggregate article characteristics by prediction outcome")
    lines.append("")

    for label, subset in [
        ("RAG correct (all ESI levels, n=%d)" % len(all_rag_correct), all_rag_correct),
        ("RAG wrong (all ESI levels, n=%d)" % len(all_rag_wrong), all_rag_wrong),
        ("ESI-3 RAG correct (n=%d)" % len(esi3_rag_correct), esi3_rag_correct),
        ("ESI-3 RAG wrong/over-triaged (n=%d)" % len(esi3_rag_wrong), esi3_rag_wrong),
    ]:
        lines.append(f"### {label}")
        lines.append("")

        # Collect article-level stats
        all_citations = []
        all_snippets = []
        all_ranks = []
        unique_pmcids = set()
        duplicate_counts = []

        for _, row in subset.iterrows():
            sid = row["stay_id"]
            articles = diagnostics.get(sid, [])
            top5 = [a for a in articles if a["rank"] <= 5]
            pmcids = [a["pmc_id"] for a in top5]
            duplicate_counts.append(len(pmcids) - len(set(pmcids)))
            for art in top5:
                pmc_id = art["pmc_id"]
                unique_pmcids.add(pmc_id)
                all_snippets.append(art["snippet_len"])
                all_ranks.append(art["rank"])
                if pmc_id in cache.index:
                    all_citations.append(cache.loc[pmc_id, "article_citation"])

        lines.append(f"- Cases: {len(subset)}")
        lines.append(f"- Unique articles retrieved (top-5 per case): {len(unique_pmcids)}")
        lines.append(f"- Mean snippet length: {np.mean(all_snippets):.0f} chars")
        lines.append(f"- Mean duplicate articles per case (in top-5): {np.mean(duplicate_counts):.2f}")
        lines.append("")

        # Extract journal names from citations
        journals = []
        for cit in all_citations:
            if isinstance(cit, str) and ". " in cit:
                # Citation format: "Journal Name. YYYY Mon DD; Vol:Page"
                journal = cit.split(". ")[0]
                journals.append(journal)

        if journals:
            from collections import Counter
            j_counts = Counter(journals).most_common(10)
            lines.append("Top journals:")
            lines.append("")
            lines.append("| Journal | Count |")
            lines.append("| --- | --- |")
            for j, c in j_counts:
                lines.append(f"| {j} | {c} |")
            lines.append("")

    # ── Section 2: ESI-3 correct cases (deep dive) ───────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 2. ESI-3 cases where RAG predicted correctly (n=%d)" % len(esi3_rag_correct))
    lines.append("")
    lines.append("These are the rare cases where RAG retrieved articles AND the model still")
    lines.append("correctly predicted ESI-3 (instead of over-triaging to ESI-2).")
    lines.append("")

    for _, row in esi3_rag_correct.iterrows():
        sid = row["stay_id"]
        lines.append(f"### stay_id: {sid}")
        lines.append("")
        lines.append(f"- **Chief complaint:** {row.get('chiefcomplaint', 'N/A')}")
        lines.append(f"- **Top-1 distance:** {row['top1_distance']:.4f}")
        lines.append(f"- **RAG prediction:** ESI {int(row['triage_RAG'])} (correct)")
        lines.append(f"- **LLM-only prediction:** ESI {int(row['triage_LLM'])}")
        lines.append("")

        # HPI excerpt
        hpi = str(row.get("HPI", ""))
        if hpi and hpi != "nan":
            lines.append(f"**HPI excerpt:** {truncate(hpi, 400)}")
            lines.append("")

        # Retrieved articles
        articles = diagnostics.get(sid, [])
        top5 = [a for a in articles if a["rank"] <= 5]
        lines.append("**Retrieved articles (top-5):**")
        lines.append("")
        lines.append("| Rank | PMC ID | Citation | Snippet (first 200 chars) |")
        lines.append("| --- | --- | --- | --- |")

        for art in top5:
            pmc_id = art["pmc_id"]
            citation = ""
            snippet = ""
            if pmc_id in cache.index:
                row_cache = cache.loc[pmc_id]
                citation = str(row_cache.get("article_citation", ""))[:80]
                raw_text = str(row_cache.get("article_text", ""))
                # Try to find body content
                snippet = truncate(raw_text, 200).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {art['rank']} | {pmc_id} | {citation} | {snippet} |")

        lines.append("")

    # ── Section 3: ESI-3 incorrect cases (sample) ────────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 3. ESI-3 cases where RAG over-triaged to ESI-2 (sample of 5/%d)" % len(esi3_rag_wrong))
    lines.append("")
    lines.append("Representative sample of the dominant error pattern: true ESI-3 predicted as ESI-2.")
    lines.append("")

    # Pick 5 spread across the distance range
    wrong_sorted = esi3_rag_wrong.sort_values("top1_distance")
    sample_indices = np.linspace(0, len(wrong_sorted) - 1, min(5, len(wrong_sorted)), dtype=int)
    wrong_sample = wrong_sorted.iloc[sample_indices]

    for _, row in wrong_sample.iterrows():
        sid = row["stay_id"]
        lines.append(f"### stay_id: {sid}")
        lines.append("")
        lines.append(f"- **Chief complaint:** {row.get('chiefcomplaint', 'N/A')}")
        lines.append(f"- **Top-1 distance:** {row['top1_distance']:.4f}")
        lines.append(f"- **RAG prediction:** ESI {int(row['triage_RAG'])} (over-triaged)")
        lines.append(f"- **LLM-only prediction:** ESI {int(row['triage_LLM'])}")
        lines.append("")

        hpi = str(row.get("HPI", ""))
        if hpi and hpi != "nan":
            lines.append(f"**HPI excerpt:** {truncate(hpi, 400)}")
            lines.append("")

        articles = diagnostics.get(sid, [])
        top5 = [a for a in articles if a["rank"] <= 5]
        lines.append("**Retrieved articles (top-5):**")
        lines.append("")
        lines.append("| Rank | PMC ID | Citation | Snippet (first 200 chars) |")
        lines.append("| --- | --- | --- | --- |")

        for art in top5:
            pmc_id = art["pmc_id"]
            citation = ""
            snippet = ""
            if pmc_id in cache.index:
                row_cache = cache.loc[pmc_id]
                citation = str(row_cache.get("article_citation", ""))[:80]
                raw_text = str(row_cache.get("article_text", ""))
                snippet = truncate(raw_text, 200).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {art['rank']} | {pmc_id} | {citation} | {snippet} |")

        lines.append("")

    # ── Section 4: ESI-2 correct cases (for contrast) ────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 4. ESI-2 cases where RAG predicted correctly (sample of 5/%d)" % len(esi2_rag_correct))
    lines.append("")
    lines.append("Contrast group: cases where retrieved articles aligned with the correct")
    lines.append("ESI-2 prediction. Are the articles qualitatively different?")
    lines.append("")

    esi2_sample = esi2_rag_correct.head(5)
    for _, row in esi2_sample.iterrows():
        sid = row["stay_id"]
        lines.append(f"### stay_id: {sid}")
        lines.append("")
        lines.append(f"- **Chief complaint:** {row.get('chiefcomplaint', 'N/A')}")
        lines.append(f"- **Top-1 distance:** {row['top1_distance']:.4f}")
        lines.append(f"- **RAG prediction:** ESI {int(row['triage_RAG'])} (correct)")
        lines.append(f"- **LLM-only prediction:** ESI {int(row['triage_LLM'])}")
        lines.append("")

        articles = diagnostics.get(sid, [])
        top5 = [a for a in articles if a["rank"] <= 5]
        lines.append("**Retrieved articles (top-5):**")
        lines.append("")
        lines.append("| Rank | PMC ID | Citation | Snippet (first 200 chars) |")
        lines.append("| --- | --- | --- | --- |")

        for art in top5:
            pmc_id = art["pmc_id"]
            citation = ""
            snippet = ""
            if pmc_id in cache.index:
                row_cache = cache.loc[pmc_id]
                citation = str(row_cache.get("article_citation", ""))[:80]
                raw_text = str(row_cache.get("article_text", ""))
                snippet = truncate(raw_text, 200).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {art['rank']} | {pmc_id} | {citation} | {snippet} |")

        lines.append("")

    # ── Section 5: Cross-group comparison ─────────────────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 5. Cross-group comparison: article content themes")
    lines.append("")

    # Analyze article text for keyword presence
    keyword_groups = {
        "triage": ["triage", "emergency severity", "ESI", "acuity"],
        "treatment/management": ["treatment", "management", "therapy", "intervention"],
        "diagnosis": ["diagnosis", "diagnostic", "differential"],
        "epidemiology": ["prevalence", "incidence", "epidemiology", "cohort", "retrospective"],
        "pathophysiology": ["pathophysiology", "mechanism", "pathogenesis"],
        "case report": ["case report", "case presentation", "we present"],
        "ED/emergency": ["emergency department", "emergency room", "ED visit", "ED presentation"],
        "review/meta": ["systematic review", "meta-analysis", "literature review"],
    }

    lines.append("Keyword presence in top-5 retrieved articles (% of articles containing keyword group):")
    lines.append("")
    lines.append("| Keyword group | ESI-3 correct (n=%d articles) | ESI-3 wrong (n=%d articles) | ESI-2 correct (n=%d articles) |" % (
        len(esi3_rag_correct) * 5, len(esi3_rag_wrong) * 5, len(esi2_rag_correct) * 5))
    lines.append("| --- | --- | --- | --- |")

    for group_name, keywords in keyword_groups.items():
        rates = []
        for subset in [esi3_rag_correct, esi3_rag_wrong, esi2_rag_correct]:
            total = 0
            hits = 0
            for _, row in subset.iterrows():
                sid = row["stay_id"]
                articles = diagnostics.get(sid, [])
                top5 = [a for a in articles if a["rank"] <= 5]
                for art in top5:
                    pmc_id = art["pmc_id"]
                    if pmc_id in cache.index:
                        text = str(cache.loc[pmc_id, "article_text"]).lower()
                        total += 1
                        if any(kw.lower() in text for kw in keywords):
                            hits += 1
            rate = (hits / total * 100) if total > 0 else 0
            rates.append(f"{rate:.0f}%")
        lines.append(f"| {group_name} | {rates[0]} | {rates[1]} | {rates[2]} |")

    lines.append("")

    # ── Section 6: Duplicate article analysis ─────────────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 6. Article overlap and duplicate analysis")
    lines.append("")

    # Check how many articles appear in both correct and wrong groups
    correct_pmcids = set()
    wrong_pmcids = set()
    for _, row in esi3_rag_correct.iterrows():
        arts = diagnostics.get(row["stay_id"], [])
        correct_pmcids.update(a["pmc_id"] for a in arts if a["rank"] <= 5)
    for _, row in esi3_rag_wrong.iterrows():
        arts = diagnostics.get(row["stay_id"], [])
        wrong_pmcids.update(a["pmc_id"] for a in arts if a["rank"] <= 5)

    overlap = correct_pmcids & wrong_pmcids
    lines.append(f"- Unique articles in ESI-3 correct group: {len(correct_pmcids)}")
    lines.append(f"- Unique articles in ESI-3 wrong group: {len(wrong_pmcids)}")
    lines.append(f"- Overlap: {len(overlap)} articles appear in both groups")
    lines.append("")

    if overlap:
        lines.append("Shared articles:")
        lines.append("")
        for pmc_id in sorted(overlap):
            if pmc_id in cache.index:
                cit = str(cache.loc[pmc_id, "article_citation"])[:100]
                lines.append(f"- {pmc_id}: {cit}")
        lines.append("")

    # ── Section 7: FAISS duplicate retrieval ──────────────────────────

    lines.append("---")
    lines.append("")
    lines.append("## 7. FAISS duplicate retrieval (same article multiple ranks)")
    lines.append("")
    lines.append("Cases where the same PMC ID appears at multiple ranks in the top-5/10,")
    lines.append("indicating the FAISS index contains duplicate embeddings for the same article.")
    lines.append("")

    dup_cases = []
    for _, row in all_rag.iterrows():
        sid = row["stay_id"]
        articles = diagnostics.get(sid, [])
        top5 = [a for a in articles if a["rank"] <= 5]
        pmcids = [a["pmc_id"] for a in top5]
        if len(pmcids) != len(set(pmcids)):
            from collections import Counter
            dups = {k: v for k, v in Counter(pmcids).items() if v > 1}
            dup_cases.append((sid, int(row["triage"]), int(row["triage_RAG"]), dups))

    lines.append(f"Cases with duplicate articles in top-5: {len(dup_cases)} / {len(all_rag)}")
    lines.append("")
    if dup_cases:
        lines.append("| stay_id | True ESI | RAG pred | Duplicate PMC IDs (count) |")
        lines.append("| --- | --- | --- | --- |")
        for sid, true_esi, pred_esi, dups in dup_cases[:15]:
            dup_str = ", ".join(f"{k} (×{v})" for k, v in dups.items())
            lines.append(f"| {sid} | {true_esi} | {pred_esi} | {dup_str} |")
        lines.append("")

    # ── Write output ──────────────────────────────────────────────────

    output_text = "\n".join(lines) + "\n"
    OUTPUT_MD.write_text(output_text, encoding="utf-8")
    print(f"\nAnalysis written to: {OUTPUT_MD}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
