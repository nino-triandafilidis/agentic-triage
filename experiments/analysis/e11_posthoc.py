"""Post-hoc analysis of E11 distance-gated retrieval results.

Reconstructs E11's per-row gating decision by:
  1. Computing FAISS top-1 distances for 150 dev_tune rows (embedding API only).
  2. Merging with existing E07 RAG predictions (always-on RAG, same 150 rows).
  3. Merging with existing E00.5 LLM-only predictions (same 150 rows).
  4. Simulating gating at threshold=0.25 and analyzing subsets.

Analyses:
  a) kappa / accuracy on RAG-used vs LLM-only fallback subsets
  b) ESI level distribution by subset
  c) Distance distribution of RAG-used cases
  d) Error patterns — over-triage vs under-triage by subset

Cost: ~150 embedding API calls (text-embedding-005), negligible cost (~$0.001).

Usage:
    .venv/bin/python experiments/analysis/e11_posthoc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Resolve the main repo root (worktrees may not have data/runs)
# Walk up from ROOT to find the actual repo root with data/splits
MAIN_ROOT = ROOT
if not (MAIN_ROOT / "data" / "splits").exists():
    # We're likely in a worktree — find the main repo
    _candidate = MAIN_ROOT
    while _candidate != _candidate.parent:
        _main = _candidate.parent
        # git worktrees are typically under .claude/worktrees/
        if (_main / "data" / "splits").exists():
            MAIN_ROOT = _main
            break
        # Also check if we're in .claude/worktrees/<name>/
        for _p in [_candidate.parent, _candidate.parent.parent, _candidate.parent.parent.parent]:
            if (_p / "data" / "splits").exists():
                MAIN_ROOT = _p
                break
        if MAIN_ROOT != ROOT:
            break
        _candidate = _candidate.parent

print(f"Script ROOT: {ROOT}")
print(f"Data ROOT:   {MAIN_ROOT}")

# ── Paths ──────────────────────────────────────────────────────────────
DEV_TUNE = MAIN_ROOT / "data" / "splits" / "dev_tune.csv"

# E07 always-on RAG (8k + strip, top_k=5) — same 150 rows, same model
# Note: the E07 top_k=5 run was saved under the E04_8k_snippet_cleaning prefix
E07_RAG = MAIN_ROOT / "data" / "runs" / "E04_8k_snippet_cleaning" / "E04_8k_snippet_cleaning_strip_20260305_041145.csv"

# E00.5 LLM-only — same 150 rows, same model (gemini-2.5-flash)
LLM_ONLY = MAIN_ROOT / "data" / "runs" / "dev_tune_llm_triage_results" / "dev_tune_llm_triage_results_20260303_230439.csv"

THRESHOLD = 0.25
N_ROWS = 150


def compute_distances(df: pd.DataFrame) -> list[dict]:
    """Compute FAISS top-1 and top-5 mean distances for each row."""
    import src.config as config
    from src.rag.retrieval import search_pubmed_articles
    from src.rag.query_agents import _concat_fields

    config.setup_clients()

    results = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        case = row.to_dict()
        query_text = _concat_fields(case)

        articles = search_pubmed_articles(query_text, top_k=5)
        # Column may be 'score' or 'distance' depending on code version
        dist_col = "distance" if "distance" in articles.columns else "score"
        top1_dist = float(articles[dist_col].iloc[0]) if len(articles) > 0 else 999.0
        mean_dist = float(articles[dist_col].mean()) if len(articles) > 0 else 999.0

        results.append({
            "stay_id": row["stay_id"],
            "top1_distance": round(top1_dist, 6),
            "mean_top5_distance": round(mean_dist, 6),
        })

        if (idx + 1) % 25 == 0:
            print(f"  Computed distances for {idx + 1}/{len(df)} rows")

    return results


def evaluate_subset(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict:
    """Compute triage metrics for a subset."""
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    mask = y_pred_num.notna() & y_true.notna()
    n_total = len(y_true)
    n_valid = int(mask.sum())

    if n_valid == 0:
        return {"label": label, "n": 0}

    yt = y_true[mask].astype(int)
    yp = y_pred_num[mask].astype(int)

    exact_acc = float((yt == yp).mean())
    diff = yt - yp
    range_acc = float(((diff >= 0) & (diff <= 1)).mean())
    mae = float(np.abs(yt - yp).mean())
    adj1_acc = float((np.abs(yt - yp) <= 1).mean())
    under_triage = float((yp > yt).mean())
    over_triage = float((yp < yt).mean())

    try:
        kappa = float(cohen_kappa_score(yt, yp, weights="quadratic"))
    except ValueError:
        kappa = float("nan")

    return {
        "label": label,
        "n": n_valid,
        "exact_accuracy": round(exact_acc, 4),
        "adj1_accuracy": round(adj1_acc, 4),
        "range_accuracy": round(range_acc, 4),
        "mae": round(mae, 4),
        "quadratic_kappa": round(kappa, 4),
        "under_triage_rate": round(under_triage, 4),
        "over_triage_rate": round(over_triage, 4),
    }


def main():
    print("=" * 70)
    print("E11 POST-HOC ANALYSIS: Distance-gated retrieval (threshold=0.25)")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    df_dev = pd.read_csv(DEV_TUNE, nrows=N_ROWS)
    df_rag = pd.read_csv(E07_RAG)
    df_llm = pd.read_csv(LLM_ONLY)

    assert len(df_dev) == N_ROWS, f"Expected {N_ROWS} rows, got {len(df_dev)}"
    assert len(df_rag) == N_ROWS, f"E07 RAG has {len(df_rag)} rows, expected {N_ROWS}"
    assert len(df_llm) == N_ROWS, f"LLM-only has {len(df_llm)} rows, expected {N_ROWS}"

    # Verify row alignment
    assert list(df_dev["stay_id"]) == list(df_rag["stay_id"]), "E07 RAG rows not aligned with dev_tune"
    assert list(df_dev["stay_id"]) == list(df_llm["stay_id"]), "LLM-only rows not aligned with dev_tune"

    print(f"\nLoaded {N_ROWS} rows from dev_tune, E07 RAG, and LLM-only results")

    # ── Step 1: Compute distances ──────────────────────────────────────
    print("\nStep 1: Computing FAISS top-1 distances for all 150 rows...")
    print("  (Embedding API calls only — no LLM calls, negligible cost)")
    dist_records = compute_distances(df_dev)
    dist_df = pd.DataFrame(dist_records)

    # ── Step 2: Merge into analysis DataFrame ──────────────────────────
    analysis = df_dev[["stay_id", "triage"]].copy()
    analysis = analysis.merge(dist_df, on="stay_id", how="left")
    analysis["triage_RAG"] = df_rag["triage_RAG"].values
    analysis["triage_LLM"] = df_llm["triage_LLM"].values

    analysis["used_rag"] = analysis["top1_distance"] <= THRESHOLD
    analysis["triage_gated"] = np.where(
        analysis["used_rag"],
        analysis["triage_RAG"],
        analysis["triage_LLM"],
    )

    n_rag = analysis["used_rag"].sum()
    n_llm = (~analysis["used_rag"]).sum()
    print(f"\nGating results (threshold={THRESHOLD}):")
    print(f"  RAG used:       {n_rag} cases ({n_rag/N_ROWS:.1%})")
    print(f"  LLM-only:       {n_llm} cases ({n_llm/N_ROWS:.1%})")

    rag_subset = analysis[analysis["used_rag"]]
    llm_subset = analysis[~analysis["used_rag"]]

    # ── Analysis A: kappa / accuracy by subset ─────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS A: Metrics by subset")
    print("=" * 70)

    metrics_overall = evaluate_subset(
        analysis["triage"], analysis["triage_gated"], "E11_overall (gated)"
    )
    metrics_rag = evaluate_subset(
        rag_subset["triage"], rag_subset["triage_RAG"], "RAG-used subset (E07 RAG predictions)"
    )
    metrics_llm = evaluate_subset(
        llm_subset["triage"], llm_subset["triage_LLM"], "LLM-only fallback subset"
    )
    # What if RAG-used subset had used LLM-only instead?
    metrics_rag_counterfactual = evaluate_subset(
        rag_subset["triage"], rag_subset["triage_LLM"],
        "COUNTERFACTUAL: RAG-used subset with LLM-only predictions"
    )
    # What if LLM-only subset had used RAG instead?
    metrics_llm_counterfactual = evaluate_subset(
        llm_subset["triage"], llm_subset["triage_RAG"],
        "COUNTERFACTUAL: LLM-only subset with RAG predictions"
    )

    print("\n── Summary Table ──")
    summary_rows = [metrics_overall, metrics_rag, metrics_llm,
                    metrics_rag_counterfactual, metrics_llm_counterfactual]
    summary_df = pd.DataFrame(summary_rows)
    cols = ["label", "n", "quadratic_kappa", "exact_accuracy", "range_accuracy",
            "mae", "under_triage_rate", "over_triage_rate"]
    print(summary_df[cols].to_string(index=False))

    # ── Analysis B: ESI distribution by subset ─────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS B: Ground-truth ESI distribution by subset")
    print("=" * 70)

    gt_rag = rag_subset["triage"].value_counts().sort_index()
    gt_llm = llm_subset["triage"].value_counts().sort_index()
    gt_all = analysis["triage"].value_counts().sort_index()

    esi_dist = pd.DataFrame({
        "ESI": range(1, 6),
        "All (n)": [gt_all.get(i, 0) for i in range(1, 6)],
        "RAG-used (n)": [gt_rag.get(i, 0) for i in range(1, 6)],
        "LLM-only (n)": [gt_llm.get(i, 0) for i in range(1, 6)],
    })
    esi_dist["RAG-used (%)"] = (esi_dist["RAG-used (n)"] / n_rag * 100).round(1)
    esi_dist["LLM-only (%)"] = (esi_dist["LLM-only (n)"] / n_llm * 100).round(1)
    print(esi_dist.to_string(index=False))

    # Chief complaint distribution (if available)
    if "chiefcomplaint" in analysis.columns:
        print("\n── Top chief complaints by subset ──")
        cc_rag = rag_subset["chiefcomplaint"].value_counts().head(10)
        cc_llm = llm_subset["chiefcomplaint"].value_counts().head(10)
        print("\nRAG-used (top 10 chief complaints):")
        for cc, count in cc_rag.items():
            print(f"  {count:3d}  {cc}")
        print("\nLLM-only fallback (top 10 chief complaints):")
        for cc, count in cc_llm.items():
            print(f"  {count:3d}  {cc}")

    # ── Analysis C: Distance distribution ──────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS C: Distance distribution")
    print("=" * 70)

    all_dists = analysis["top1_distance"]
    rag_dists = rag_subset["top1_distance"]
    llm_dists = llm_subset["top1_distance"]

    print(f"\nAll 150 rows:")
    print(f"  Mean:   {all_dists.mean():.4f}")
    print(f"  Median: {all_dists.median():.4f}")
    print(f"  Std:    {all_dists.std():.4f}")
    print(f"  Min:    {all_dists.min():.4f}")
    print(f"  Max:    {all_dists.max():.4f}")

    print(f"\nRAG-used cases (top1 <= {THRESHOLD}):")
    print(f"  Mean:   {rag_dists.mean():.4f}")
    print(f"  Median: {rag_dists.median():.4f}")
    print(f"  Std:    {rag_dists.std():.4f}")
    print(f"  Min:    {rag_dists.min():.4f}")
    print(f"  Max:    {rag_dists.max():.4f}")

    print(f"\nLLM-only fallback cases (top1 > {THRESHOLD}):")
    print(f"  Mean:   {llm_dists.mean():.4f}")
    print(f"  Median: {llm_dists.median():.4f}")
    print(f"  Std:    {llm_dists.std():.4f}")
    print(f"  Min:    {llm_dists.min():.4f}")
    print(f"  Max:    {llm_dists.max():.4f}")

    # Histogram of distances
    print("\n── Distance histogram (all 150 rows) ──")
    bins = [0.15, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.35, 0.40, 0.50]
    hist, _ = np.histogram(all_dists, bins=bins)
    for i in range(len(bins) - 1):
        bar = "#" * hist[i]
        marker = " <-- gate" if bins[i + 1] == 0.25 or bins[i] == 0.25 else ""
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f})  {hist[i]:3d}  {bar}{marker}")

    # ── Analysis D: Error patterns ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS D: Error patterns by subset")
    print("=" * 70)

    for subset_name, subset, pred_col in [
        ("RAG-used", rag_subset, "triage_RAG"),
        ("LLM-only fallback", llm_subset, "triage_LLM"),
    ]:
        y_true = pd.to_numeric(subset["triage"], errors="coerce").dropna().astype(int)
        y_pred = pd.to_numeric(subset[pred_col], errors="coerce")
        mask = y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask].astype(int)

        diff = y_true - y_pred  # positive = over-triage, negative = under-triage
        exact = (diff == 0).sum()
        over1 = (diff == 1).sum()
        over2 = (diff >= 2).sum()
        under1 = (diff == -1).sum()
        under2 = (diff <= -2).sum()

        print(f"\n{subset_name} (n={len(y_true)}):")
        print(f"  Exact match:        {exact:3d} ({exact/len(y_true):.1%})")
        print(f"  Over-triage by 1:   {over1:3d} ({over1/len(y_true):.1%})")
        print(f"  Over-triage by 2+:  {over2:3d} ({over2/len(y_true):.1%})")
        print(f"  Under-triage by 1:  {under1:3d} ({under1/len(y_true):.1%})")
        print(f"  Under-triage by 2+: {under2:3d} ({under2/len(y_true):.1%})")

        # Per-ESI error pattern
        print(f"\n  Per-ESI error breakdown:")
        for esi in sorted(y_true.unique()):
            mask_esi = y_true == esi
            yt_esi = y_true[mask_esi]
            yp_esi = y_pred[mask_esi]
            diff_esi = yt_esi - yp_esi
            n_esi = len(yt_esi)
            if n_esi == 0:
                continue
            exact_esi = (diff_esi == 0).sum()
            over_esi = (diff_esi > 0).sum()
            under_esi = (diff_esi < 0).sum()
            pred_dist = yp_esi.value_counts().sort_index().to_dict()
            print(f"    ESI {esi} (n={n_esi}): exact={exact_esi} over={over_esi} under={under_esi}  preds={pred_dist}")

    # ── Analysis E: Confusion matrices ─────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS E: Confusion matrices")
    print("=" * 70)

    for subset_name, subset, pred_col in [
        ("RAG-used", rag_subset, "triage_RAG"),
        ("LLM-only fallback", llm_subset, "triage_LLM"),
    ]:
        y_true = pd.to_numeric(subset["triage"], errors="coerce").dropna().astype(int)
        y_pred = pd.to_numeric(subset[pred_col], errors="coerce")
        mask = y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask].astype(int)

        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        print(f"\n{subset_name} confusion matrix (rows=true, cols=predicted):")
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                             columns=[f"pred_{l}" for l in labels])
        print(cm_df.to_string())

    # ── Analysis F: Does RAG help or hurt the 48 close-retrieval cases? ──
    print("\n" + "=" * 70)
    print("ANALYSIS F: RAG vs LLM-only for the close-retrieval cases")
    print("=" * 70)

    # For the RAG-used cases, compare RAG prediction vs LLM-only prediction
    rag_correct = (pd.to_numeric(rag_subset["triage_RAG"], errors="coerce") ==
                   rag_subset["triage"]).sum()
    llm_correct = (pd.to_numeric(rag_subset["triage_LLM"], errors="coerce") ==
                   rag_subset["triage"]).sum()

    print(f"\nAmong the {n_rag} RAG-used cases (top1 <= {THRESHOLD}):")
    print(f"  RAG exact matches:      {rag_correct} ({rag_correct/n_rag:.1%})")
    print(f"  LLM-only exact matches: {llm_correct} ({llm_correct/n_rag:.1%})")

    # Per-row comparison
    rag_pred = pd.to_numeric(rag_subset["triage_RAG"], errors="coerce")
    llm_pred = pd.to_numeric(rag_subset["triage_LLM"], errors="coerce")
    gt = rag_subset["triage"].values

    rag_right_llm_wrong = ((rag_pred == gt) & (llm_pred != gt)).sum()
    rag_wrong_llm_right = ((rag_pred != gt) & (llm_pred == gt)).sum()
    both_right = ((rag_pred == gt) & (llm_pred == gt)).sum()
    both_wrong = ((rag_pred != gt) & (llm_pred != gt)).sum()

    print(f"\n  Both correct:           {both_right}")
    print(f"  RAG correct, LLM wrong: {rag_right_llm_wrong}")
    print(f"  LLM correct, RAG wrong: {rag_wrong_llm_right}")
    print(f"  Both wrong:             {both_wrong}")

    # Same for the LLM-only fallback cases
    rag_correct_fb = (pd.to_numeric(llm_subset["triage_RAG"], errors="coerce") ==
                      llm_subset["triage"]).sum()
    llm_correct_fb = (pd.to_numeric(llm_subset["triage_LLM"], errors="coerce") ==
                      llm_subset["triage"]).sum()

    print(f"\nAmong the {n_llm} LLM-only fallback cases (top1 > {THRESHOLD}):")
    print(f"  RAG exact matches:      {rag_correct_fb} ({rag_correct_fb/n_llm:.1%})")
    print(f"  LLM-only exact matches: {llm_correct_fb} ({llm_correct_fb/n_llm:.1%})")

    rag_pred_fb = pd.to_numeric(llm_subset["triage_RAG"], errors="coerce")
    llm_pred_fb = pd.to_numeric(llm_subset["triage_LLM"], errors="coerce")
    gt_fb = llm_subset["triage"].values

    rag_right_llm_wrong_fb = ((rag_pred_fb == gt_fb) & (llm_pred_fb != gt_fb)).sum()
    rag_wrong_llm_right_fb = ((rag_pred_fb != gt_fb) & (llm_pred_fb == gt_fb)).sum()
    both_right_fb = ((rag_pred_fb == gt_fb) & (llm_pred_fb == gt_fb)).sum()
    both_wrong_fb = ((rag_pred_fb != gt_fb) & (llm_pred_fb != gt_fb)).sum()

    print(f"\n  Both correct:           {both_right_fb}")
    print(f"  RAG correct, LLM wrong: {rag_right_llm_wrong_fb}")
    print(f"  LLM correct, RAG wrong: {rag_wrong_llm_right_fb}")
    print(f"  Both wrong:             {both_wrong_fb}")

    # ── Save analysis DataFrame ────────────────────────────────────────
    out_dir = ROOT / "experiments" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e11_posthoc_analysis.csv"
    analysis.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"Analysis DataFrame saved to: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
