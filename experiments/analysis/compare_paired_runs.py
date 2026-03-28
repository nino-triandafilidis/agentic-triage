"""Compare two paired triage run CSVs row-by-row to measure marginal value.

Takes a base run and comparison run (matched by stay_id) and classifies
each row as helped, hurt, or neutral.  Produces summary statistics
overall, stratified by ESI level, and optionally by distance bucket.

Reuses eval_triage.evaluate() for per-subset metric computation.

Usage:
    .venv/bin/python experiments/analysis/compare_paired_runs.py \
        --base data/runs/E00.5_llm/E00.5_llm_20260303.csv \
        --comparison data/runs/E07_rag/E07_rag_20260305.csv

    # With distance buckets from e11 posthoc CSV:
    .venv/bin/python experiments/analysis/compare_paired_runs.py \
        --base data/runs/E00.5_llm/E00.5_llm_20260303.csv \
        --comparison data/runs/E07_rag/E07_rag_20260305.csv \
        --distance-csv experiments/analysis/e11_posthoc_analysis.csv

    # Specify prediction column names explicitly:
    .venv/bin/python experiments/analysis/compare_paired_runs.py \
        --base data/runs/run_a.csv --base-pred triage_LLM \
        --comparison data/runs/run_b.csv --comp-pred triage_RAG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.eval_triage import evaluate, find_gt_column


# ── Helpers ────────────────────────────────────────────────────────────

def _find_pred_column(df: pd.DataFrame, hint: str | None) -> str:
    """Find a prediction column in the DataFrame."""
    if hint:
        if hint not in df.columns:
            raise ValueError(f"Prediction column '{hint}' not in CSV.")
        return hint
    # Auto-detect: prefer triage_RAG, then triage_LLM, then triage_gated
    for name in ["triage_RAG", "triage_LLM", "triage_gated"]:
        if name in df.columns:
            return name
    raise ValueError(
        "Could not auto-detect prediction column. "
        "Pass --base-pred / --comp-pred explicitly."
    )


def _classify_rows(
    gt: pd.Series,
    base_pred: pd.Series,
    comp_pred: pd.Series,
) -> pd.Series:
    """Classify each row: helped, hurt, neutral_both_correct,
    neutral_both_wrong, neutral_same_pred.

    Returns a categorical Series aligned with the input index.
    """
    gt_n = pd.to_numeric(gt, errors="coerce")
    base_n = pd.to_numeric(base_pred, errors="coerce")
    comp_n = pd.to_numeric(comp_pred, errors="coerce")

    base_correct = base_n == gt_n
    comp_correct = comp_n == gt_n
    same_pred = base_n == comp_n

    categories = pd.Series("neutral_same_pred", index=gt.index)
    categories[~same_pred & comp_correct & ~base_correct] = "helped"
    categories[~same_pred & ~comp_correct & base_correct] = "hurt"
    categories[~same_pred & comp_correct & base_correct] = "neutral_both_correct"
    categories[~same_pred & ~comp_correct & ~base_correct] = "neutral_both_wrong"
    # Rows where either base or comp is NaN (parse failure)
    any_nan = base_n.isna() | comp_n.isna() | gt_n.isna()
    categories[any_nan] = "parse_failure"

    return categories


def _subset_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: str,
) -> dict:
    """Compute kappa + accuracy for a subset. Minimal wrapper around evaluate()."""
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    mask = y_pred_num.notna() & y_true.notna()
    n_valid = int(mask.sum())
    if n_valid < 2:
        return {"label": label, "n": n_valid}

    yt = y_true[mask].astype(int)
    yp = y_pred_num[mask].astype(int)

    exact_acc = float((yt == yp).mean())
    try:
        kappa = float(cohen_kappa_score(yt, yp, weights="quadratic"))
    except ValueError:
        kappa = float("nan")

    return {
        "label": label,
        "n": n_valid,
        "exact_accuracy": round(exact_acc, 4),
        "quadratic_kappa": round(kappa, 4),
    }


def _print_classification_summary(
    categories: pd.Series,
    label: str = "Overall",
) -> dict:
    """Print and return summary counts for a classification series."""
    n = len(categories)
    counts = categories.value_counts()

    helped = int(counts.get("helped", 0))
    hurt = int(counts.get("hurt", 0))
    neutral_same = int(counts.get("neutral_same_pred", 0))
    neutral_bc = int(counts.get("neutral_both_correct", 0))
    neutral_bw = int(counts.get("neutral_both_wrong", 0))
    parse_fail = int(counts.get("parse_failure", 0))
    changed = helped + hurt + neutral_bc + neutral_bw

    print(f"\n  {label} (n={n}):")
    print(f"    Changed predictions:  {changed:4d} ({changed/n:.1%})")
    print(f"      Helped (comp correct, base wrong):  {helped:4d} ({helped/n:.1%})")
    print(f"      Hurt   (comp wrong, base correct):  {hurt:4d} ({hurt/n:.1%})")
    print(f"      Both correct (different pred):       {neutral_bc:4d}")
    print(f"      Both wrong   (different pred):       {neutral_bw:4d}")
    print(f"    Same prediction:      {neutral_same:4d} ({neutral_same/n:.1%})")
    if parse_fail > 0:
        print(f"    Parse failures:       {parse_fail:4d}")

    return {
        "label": label,
        "n": n,
        "changed": changed,
        "helped": helped,
        "hurt": hurt,
        "neutral_both_correct": neutral_bc,
        "neutral_both_wrong": neutral_bw,
        "neutral_same_pred": neutral_same,
        "parse_failure": parse_fail,
        "net_lift": helped - hurt,
    }


# ── Main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two paired triage run CSVs row-by-row."
    )
    p.add_argument("--base", type=Path, required=True,
                   help="Base run CSV (the 'before' or control)")
    p.add_argument("--comparison", type=Path, required=True,
                   help="Comparison run CSV (the 'after' or treatment)")
    p.add_argument("--base-pred", type=str, default=None,
                   help="Prediction column in base CSV (auto-detected if omitted)")
    p.add_argument("--comp-pred", type=str, default=None,
                   help="Prediction column in comparison CSV (auto-detected if omitted)")
    p.add_argument("--gt-col", type=str, default=None,
                   help="Ground-truth column (auto-detected if omitted)")
    p.add_argument("--distance-csv", type=Path, default=None,
                   help="CSV with stay_id and top1_distance columns for distance "
                   "bucket stratification (e.g. e11_posthoc_analysis.csv)")
    p.add_argument("--distance-col", type=str, default="top1_distance",
                   help="Column name for distance in --distance-csv (default: top1_distance)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV with per-row classification (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    df_base = pd.read_csv(args.base)
    df_comp = pd.read_csv(args.comparison)

    print(f"Base run:       {args.base.name} ({len(df_base)} rows)")
    print(f"Comparison run: {args.comparison.name} ({len(df_comp)} rows)")

    # Find columns
    gt_col = find_gt_column(df_base, args.gt_col)
    base_pred_col = _find_pred_column(df_base, args.base_pred)
    comp_pred_col = _find_pred_column(df_comp, args.comp_pred)

    print(f"Ground-truth:   '{gt_col}'")
    print(f"Base pred col:  '{base_pred_col}'")
    print(f"Comp pred col:  '{comp_pred_col}'")

    # Align by stay_id
    if "stay_id" not in df_base.columns or "stay_id" not in df_comp.columns:
        print("WARNING: No stay_id column — assuming rows are aligned by position.")
        if len(df_base) != len(df_comp):
            raise ValueError(
                f"Row count mismatch ({len(df_base)} vs {len(df_comp)}) "
                "and no stay_id column to merge on."
            )
        merged = pd.DataFrame({
            "gt": df_base[gt_col].values,
            "base_pred": df_base[base_pred_col].values,
            "comp_pred": df_comp[comp_pred_col].values,
        })
        if gt_col in df_base.columns:
            # Bring ESI level through for stratification
            merged["esi_gt"] = pd.to_numeric(df_base[gt_col], errors="coerce")
        if "stay_id" in df_base.columns:
            merged["stay_id"] = df_base["stay_id"].values
    else:
        merged = df_base[["stay_id", gt_col]].rename(columns={gt_col: "gt"}).copy()
        merged["base_pred"] = df_base[base_pred_col].values
        comp_cols = df_comp[["stay_id", comp_pred_col]].rename(
            columns={comp_pred_col: "comp_pred"}
        )
        merged = merged.merge(comp_cols, on="stay_id", how="inner")
        if len(merged) < len(df_base):
            print(f"WARNING: Inner join on stay_id reduced rows from "
                  f"{len(df_base)} to {len(merged)}")
        merged["esi_gt"] = pd.to_numeric(merged["gt"], errors="coerce")

    # Classify rows
    merged["category"] = _classify_rows(
        merged["gt"], merged["base_pred"], merged["comp_pred"]
    )

    # ── Overall summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROW-LEVEL COMPARISON: base vs comparison")
    print("=" * 70)

    overall = _print_classification_summary(merged["category"], "Overall")

    # ── Per-subset metrics ────────────────────────────────────────────
    print("\n── Per-run metrics (all rows) ──")
    base_metrics = evaluate(
        pd.to_numeric(merged["gt"], errors="coerce"),
        pd.to_numeric(merged["base_pred"], errors="coerce"),
        label="Base run",
    )
    comp_metrics = evaluate(
        pd.to_numeric(merged["gt"], errors="coerce"),
        pd.to_numeric(merged["comp_pred"], errors="coerce"),
        label="Comparison run",
    )

    # ── Stratify by ESI level ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STRATIFICATION BY ESI LEVEL")
    print("=" * 70)

    esi_summaries = []
    for esi in sorted(merged["esi_gt"].dropna().unique()):
        esi = int(esi)
        mask = merged["esi_gt"] == esi
        subset = merged[mask]
        if len(subset) == 0:
            continue

        summary = _print_classification_summary(
            subset["category"], f"ESI {esi}"
        )
        esi_summaries.append(summary)

        # Per-ESI metrics for both runs
        base_m = _subset_metrics(
            pd.to_numeric(subset["gt"], errors="coerce"),
            pd.to_numeric(subset["base_pred"], errors="coerce"),
            f"ESI {esi} base",
        )
        comp_m = _subset_metrics(
            pd.to_numeric(subset["gt"], errors="coerce"),
            pd.to_numeric(subset["comp_pred"], errors="coerce"),
            f"ESI {esi} comp",
        )
        if base_m.get("exact_accuracy") is not None:
            print(f"    Base:  acc={base_m['exact_accuracy']:.3f}  "
                  f"kappa={base_m.get('quadratic_kappa', 'N/A')}")
        if comp_m.get("exact_accuracy") is not None:
            print(f"    Comp:  acc={comp_m['exact_accuracy']:.3f}  "
                  f"kappa={comp_m.get('quadratic_kappa', 'N/A')}")

    # ESI summary table
    if esi_summaries:
        print("\n── ESI summary table ──")
        esi_df = pd.DataFrame(esi_summaries)
        cols = ["label", "n", "helped", "hurt", "net_lift",
                "changed", "neutral_same_pred"]
        print(esi_df[[c for c in cols if c in esi_df.columns]].to_string(index=False))

    # ── Stratify by distance bucket (optional) ────────────────────────
    if args.distance_csv is not None:
        print("\n" + "=" * 70)
        print("STRATIFICATION BY DISTANCE BUCKET")
        print("=" * 70)

        dist_df = pd.read_csv(args.distance_csv)
        dist_col = args.distance_col

        if dist_col not in dist_df.columns:
            print(f"WARNING: Column '{dist_col}' not found in {args.distance_csv.name}")
        elif "stay_id" not in dist_df.columns:
            print(f"WARNING: No stay_id column in {args.distance_csv.name}")
        else:
            merged_dist = merged.merge(
                dist_df[["stay_id", dist_col]], on="stay_id", how="left"
            )
            n_with_dist = merged_dist[dist_col].notna().sum()
            print(f"Matched {n_with_dist}/{len(merged)} rows with distance data")

            if n_with_dist > 0:
                # Define distance bins
                bins = [0.0, 0.20, 0.22, 0.24, 0.25, 0.27, 0.30, 0.35, 1.0]
                labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})"
                          for i in range(len(bins) - 1)]
                merged_dist["dist_bucket"] = pd.cut(
                    merged_dist[dist_col], bins=bins, labels=labels,
                    right=False,
                )

                dist_summaries = []
                for bucket in labels:
                    mask = merged_dist["dist_bucket"] == bucket
                    subset = merged_dist[mask]
                    if len(subset) == 0:
                        continue

                    summary = _print_classification_summary(
                        subset["category"], f"Dist {bucket}"
                    )
                    dist_summaries.append(summary)

                    base_m = _subset_metrics(
                        pd.to_numeric(subset["gt"], errors="coerce"),
                        pd.to_numeric(subset["base_pred"], errors="coerce"),
                        f"{bucket} base",
                    )
                    comp_m = _subset_metrics(
                        pd.to_numeric(subset["gt"], errors="coerce"),
                        pd.to_numeric(subset["comp_pred"], errors="coerce"),
                        f"{bucket} comp",
                    )
                    if base_m.get("exact_accuracy") is not None:
                        print(f"    Base:  acc={base_m['exact_accuracy']:.3f}  "
                              f"kappa={base_m.get('quadratic_kappa', 'N/A')}")
                    if comp_m.get("exact_accuracy") is not None:
                        print(f"    Comp:  acc={comp_m['exact_accuracy']:.3f}  "
                              f"kappa={comp_m.get('quadratic_kappa', 'N/A')}")

                if dist_summaries:
                    print("\n── Distance bucket summary table ──")
                    dist_summary_df = pd.DataFrame(dist_summaries)
                    cols = ["label", "n", "helped", "hurt", "net_lift",
                            "changed", "neutral_same_pred"]
                    print(dist_summary_df[
                        [c for c in cols if c in dist_summary_df.columns]
                    ].to_string(index=False))

    # ── Save per-row output ───────────────────────────────────────────
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output, index=False)
        print(f"\nPer-row classification saved to: {args.output}")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Helped (comparison better): {overall['helped']}")
    print(f"  Hurt   (comparison worse):  {overall['hurt']}")
    print(f"  Net lift:                   {overall['net_lift']:+d}")
    if overall['n'] > 0:
        print(f"  Net lift rate:              {overall['net_lift']/overall['n']:+.1%}")


if __name__ == "__main__":
    main()
