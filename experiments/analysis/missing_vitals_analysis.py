"""Analyse missing-vitals prevalence and ESI distribution across splits.

Produces the statistics cited in docs/paper_notes.md §1 (Split structure
and vitals-missing rows).  Re-run to verify or update those claims.

Usage:
    .venv/bin/python experiments/analysis/missing_vitals_analysis.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_SRC = ROOT / "third_part_code" / "medLLMbenchmark" / "MIMIC-IV-Ext-Triage-Specialty-Diagnosis-Decision-Support.csv"
DEV_TUNE = ROOT / "data" / "splits" / "dev_tune.csv"


def analyse_split(path: Path, label: str) -> None:
    df = pd.read_csv(path)
    n = len(df)
    na_mask = df["initial_vitals"].isna()
    n_na = int(na_mask.sum())
    pct_na = n_na / n * 100

    print(f"\n{'=' * 60}")
    print(f" {label}  (n={n:,})")
    print(f"{'=' * 60}")
    print(f"  Missing initial_vitals: {n_na} / {n} ({pct_na:.1f}%)")

    if n_na == 0:
        print("  No missing-vitals rows — nothing else to report.")
        return

    triage_col = "triage" if "triage" in df.columns else "acuity"

    print(f"\n  ESI distribution — missing vitals only:")
    na_dist = df.loc[na_mask, triage_col].value_counts().sort_index()
    for esi, count in na_dist.items():
        pct_of_na = count / n_na * 100
        print(f"    ESI {esi}: {count:>4d}  ({pct_of_na:.0f}% of missing-vitals rows)")

    print(f"\n  Impact on ESI-1:")
    total_esi1 = int((df[triage_col] == 1).sum())
    na_esi1 = int(na_dist.get(1, 0))
    if total_esi1 > 0:
        pct_esi1_missing = na_esi1 / total_esi1 * 100
        print(f"    {na_esi1} of {total_esi1} ESI-1 cases have missing vitals ({pct_esi1_missing:.0f}%)")
    else:
        print("    No ESI-1 cases in this split.")

    print(f"\n  Full ESI distribution — all rows:")
    full_dist = df[triage_col].value_counts().sort_index()
    for esi, count in full_dist.items():
        print(f"    ESI {esi}: {count:>4d}  ({count / n * 100:.1f}%)")


def main() -> None:
    for path, label in [(TEST_SRC, "Test set (2,200)"), (DEV_TUNE, "dev_tune")]:
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        analyse_split(path, label)
    print()


if __name__ == "__main__":
    main()
