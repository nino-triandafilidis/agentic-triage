"""Analyze the 9 cases where RAG predicted correctly but LLM-only did not.

Contrasts with the 10 reverse cases (LLM correct, RAG wrong).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.rag.text_cleaning import extract_body

posthoc = pd.read_csv(ROOT / "experiments/analysis/e11_posthoc_analysis.csv")
e07 = pd.read_csv(ROOT / "data/runs/E07_8k_strip_top10/E07_8k_strip_top10_strip_20260305_070703.csv")
cache = pd.read_parquet(ROOT / "data/cache/retrieval_cache.parquet").drop_duplicates("pmc_id").set_index("pmc_id")

diag = {}
with open(ROOT / "data/runs/E07_8k_strip_top10/E07_8k_strip_top10_strip_20260305_070703.diagnostics.jsonl") as f:
    for line in f:
        row = json.loads(line)
        diag[row["stay_id"]] = row["article_stats"]

rag_used = posthoc[posthoc["used_rag"] == True]
fallback = posthoc[posthoc["used_rag"] == False]

# RAG wins
g1 = rag_used[(rag_used["triage_RAG"] == rag_used["triage"]) & (rag_used["triage_LLM"] != rag_used["triage"])]
g2 = fallback[(fallback["triage_RAG"] == fallback["triage"]) & (fallback["triage_LLM"] != fallback["triage"])]
wins = pd.concat([g1, g2])

# RAG losses (LLM correct, RAG wrong)
r1 = rag_used[(rag_used["triage_LLM"] == rag_used["triage"]) & (rag_used["triage_RAG"] != rag_used["triage"])]
r2 = fallback[(fallback["triage_LLM"] == fallback["triage"]) & (fallback["triage_RAG"] != fallback["triage"])]
losses = pd.concat([r1, r2])

merged_wins = wins.merge(e07[["stay_id", "chiefcomplaint", "HPI"]], on="stay_id", how="left")
merged_losses = losses.merge(e07[["stay_id", "chiefcomplaint", "HPI"]], on="stay_id", how="left")

lines = []
lines.append("# E11 RAG-wins analysis: cases where RAG correct, LLM-only wrong")
lines.append("")
lines.append(f"Total RAG wins: {len(wins)} (4 in RAG-used subset + 5 in fallback subset)")
lines.append(f"Total RAG losses (reverse): {len(losses)} (2 in RAG-used + 8 in fallback)")
lines.append("")

# Error direction summary
lines.append("## Error direction summary")
lines.append("")
lines.append("### RAG wins (n=%d)" % len(wins))
lines.append("")
lines.append("| stay_id | GT | RAG pred | LLM pred | LLM error direction | Chief complaint | Top-1 dist | RAG used? |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
for _, r in merged_wins.iterrows():
    gt, rp, lp = int(r["triage"]), int(r["triage_RAG"]), int(r["triage_LLM"])
    err = "over-triage" if lp < gt else "under-triage"
    cc = str(r.get("chiefcomplaint", "N/A"))[:40]
    lines.append(f"| {r['stay_id']} | {gt} | {rp} | {lp} | {err} | {cc} | {r['top1_distance']:.4f} | {'Yes' if r['used_rag'] else 'No'} |")
lines.append("")

# LLM error breakdown
over = len(merged_wins[merged_wins["triage_LLM"] < merged_wins["triage"]])
under = len(merged_wins[merged_wins["triage_LLM"] > merged_wins["triage"]])
lines.append(f"LLM-only errors on these cases: {over} over-triage, {under} under-triage")
lines.append("")

lines.append("### RAG losses (n=%d)" % len(losses))
lines.append("")
lines.append("| stay_id | GT | RAG pred | LLM pred | RAG error direction | Chief complaint | Top-1 dist | RAG used? |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
for _, r in merged_losses.iterrows():
    gt, rp, lp = int(r["triage"]), int(r["triage_RAG"]), int(r["triage_LLM"])
    err = "over-triage" if rp < gt else "under-triage"
    cc = str(r.get("chiefcomplaint", "N/A"))[:40]
    lines.append(f"| {r['stay_id']} | {gt} | {rp} | {lp} | {err} | {cc} | {r['top1_distance']:.4f} | {'Yes' if r['used_rag'] else 'No'} |")
lines.append("")

over_l = len(merged_losses[merged_losses["triage_RAG"] < merged_losses["triage"]])
under_l = len(merged_losses[merged_losses["triage_RAG"] > merged_losses["triage"]])
lines.append(f"RAG errors on these cases: {over_l} over-triage, {under_l} under-triage")
lines.append("")

# Per-case article deep dive for wins
lines.append("---")
lines.append("")
lines.append("## Per-case article analysis: RAG wins")
lines.append("")

for _, r in merged_wins.iterrows():
    sid = r["stay_id"]
    gt, rp, lp = int(r["triage"]), int(r["triage_RAG"]), int(r["triage_LLM"])
    err = "over-triage" if lp < gt else "under-triage"
    lines.append(f"### stay_id: {sid}")
    lines.append("")
    lines.append(f"- **GT:** ESI {gt} | **RAG:** ESI {rp} (correct) | **LLM:** ESI {lp} (wrong — {err})")
    lines.append(f"- **Chief complaint:** {r.get('chiefcomplaint', 'N/A')}")
    lines.append(f"- **Top-1 distance:** {r['top1_distance']:.4f}")
    lines.append(f"- **RAG used at gate=0.25:** {'Yes' if r['used_rag'] else 'No (fallback)'}")
    lines.append("")

    hpi = str(r.get("HPI", ""))
    if hpi and hpi != "nan":
        lines.append(f"**HPI:** {hpi[:350]}...")
        lines.append("")

    arts = diag.get(sid, [])
    top5 = [a for a in arts if a["rank"] <= 5]
    lines.append("**Top-5 retrieved articles:**")
    lines.append("")
    for a in top5:
        pmcid = a["pmc_id"]
        if pmcid in cache.index:
            cit = str(cache.loc[pmcid, "article_citation"])[:90]
            raw = str(cache.loc[pmcid, "article_text"])
            cleaned = extract_body(raw, 300) or raw[:300]
            cleaned = cleaned.replace("\n", " ")
            lines.append(f"- **Rank {a['rank']}** ({pmcid}, {cit})")
            lines.append(f"  > {cleaned}...")
            lines.append("")
    lines.append("")

# Keyword analysis comparing wins vs losses
lines.append("---")
lines.append("")
lines.append("## Keyword analysis: RAG wins vs losses")
lines.append("")

keyword_groups = {
    "triage/acuity": ["triage", "emergency severity", "ESI", "acuity"],
    "ED/emergency": ["emergency department", "emergency room", "ED visit", "ED presentation"],
    "case report": ["case report", "case presentation", "we present"],
    "treatment": ["treatment", "management", "therapy"],
    "complication": ["complication", "adverse", "mortality", "fatal", "death"],
    "diagnosis": ["diagnosis", "diagnostic", "differential"],
}

lines.append("| Keyword group | RAG wins (n=%d, %d articles) | RAG losses (n=%d, %d articles) |" % (
    len(wins), len(wins) * 5, len(losses), len(losses) * 5))
lines.append("| --- | --- | --- |")

for group_name, keywords in keyword_groups.items():
    rates = []
    for subset in [wins, losses]:
        total = 0
        hits = 0
        for _, row in subset.iterrows():
            arts = diag.get(row["stay_id"], [])
            for a in [x for x in arts if x["rank"] <= 5]:
                if a["pmc_id"] in cache.index:
                    text = str(cache.loc[a["pmc_id"], "article_text"]).lower()
                    total += 1
                    if any(kw.lower() in text for kw in keywords):
                        hits += 1
        rate = (hits / total * 100) if total > 0 else 0
        rates.append(f"{rate:.0f}%")
    lines.append(f"| {group_name} | {rates[0]} | {rates[1]} |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## Interpretation")
lines.append("")

output = ROOT / "experiments" / "analysis" / "e11_rag_wins_analysis.md"
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Written to {output}")
print(f"RAG wins: {len(wins)}, RAG losses: {len(losses)}")
