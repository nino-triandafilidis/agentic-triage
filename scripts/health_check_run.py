#!/usr/bin/env python3
"""Post-merge health check for a test-set evaluation run.

Validates that a merged run artifact is internally consistent, matches its
sidecar metadata, agrees with the experiment_log.csv entry, and that all
source shards are accounted for.

Usage:
    python scripts/health_check_run.py \
        --merged data/runs/test_W1_ref/test_W1_ref_merged_2200.csv \
        --shard-dir-pattern "data/runs/test_W1_ref_shard*" \
        --pred triage_LLM \
        --expect-rows 2200 \
        --expect-shards 10
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "experiments" / "results" / "experiment_log.csv"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _print(label: str, ok: bool, detail: str = ""):
    icon = "PASS" if ok else "FAIL"
    msg = f"  [{icon}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Health-check a merged test run")
    parser.add_argument("--merged", type=Path, required=True, help="Path to merged CSV")
    parser.add_argument("--shard-dir-pattern", type=str, default=None,
                        help="Glob pattern for shard directories (e.g. 'data/runs/test_W1_ref_shard*')")
    parser.add_argument("--pred", type=str, default="triage_LLM", help="Prediction column name")
    parser.add_argument("--target", type=str, default="triage", help="Ground-truth column name")
    parser.add_argument("--expect-rows", type=int, default=2200)
    parser.add_argument("--expect-shards", type=int, default=10)
    args = parser.parse_args()

    merged_path = ROOT / args.merged if not args.merged.is_absolute() else args.merged
    sidecar_path = merged_path.with_suffix(".json")

    print("=" * 60)
    print(f"Health check: {merged_path.name}")
    print("=" * 60)

    all_ok = True

    # ── 1. Merged CSV basics ──────────────────────────────────────
    print("\n1. Merged CSV integrity")
    if not merged_path.exists():
        _print("File exists", False, str(merged_path))
        sys.exit(1)

    df = pd.read_csv(merged_path)
    all_ok &= _print("Row count", len(df) == args.expect_rows,
                      f"{len(df)} (expected {args.expect_rows})")

    has_pred = args.pred in df.columns
    all_ok &= _print(f"Prediction column '{args.pred}' exists", has_pred)

    has_target = args.target in df.columns
    all_ok &= _print(f"Ground-truth column '{args.target}' exists", has_target)

    has_stay_id = "stay_id" in df.columns
    all_ok &= _print("stay_id column exists", has_stay_id)

    if has_stay_id:
        n_unique_ids = df["stay_id"].nunique()
        all_ok &= _print("No duplicate stay_ids", n_unique_ids == len(df),
                          f"{n_unique_ids} unique / {len(df)} rows")

    # ── 2. Prediction quality checks ─────────────────────────────
    print("\n2. Prediction quality")
    if has_pred:
        pred_vals = pd.to_numeric(df[args.pred], errors="coerce")
        n_parseable = pred_vals.notna().sum()
        parse_fail_rate = 1.0 - n_parseable / len(df)
        all_ok &= _print("Parse fail rate < 1%", parse_fail_rate < 0.01,
                          f"{parse_fail_rate:.3%} ({len(df) - n_parseable} failures)")

        valid_preds = pred_vals.dropna().astype(int)
        n_unique = valid_preds.nunique()
        all_ok &= _print("Unique predicted >= 3", n_unique >= 3, f"{n_unique} unique values")

        in_range = valid_preds.between(1, 5).all()
        all_ok &= _print("All predictions in [1, 5]", in_range,
                          f"range: [{valid_preds.min()}, {valid_preds.max()}]")

        # Distribution check — flag if any single class > 80% (near-total collapse)
        dist = valid_preds.value_counts(normalize=True)
        max_class_pct = dist.max()
        all_ok &= _print("No class > 80% of predictions", max_class_pct <= 0.8,
                          f"max: ESI-{dist.idxmax()} at {max_class_pct:.1%}")

    # ── 3. Sidecar metadata ──────────────────────────────────────
    print("\n3. Sidecar metadata")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        _print("Sidecar found", True, sidecar_path.name)

        sc_model = sidecar.get("model", "")
        _print("Model recorded", bool(sc_model), sc_model)

        sc_input = sidecar.get("input_file", "")
        all_ok &= _print("Input file is test.csv", sc_input == "test.csv",
                          f"got '{sc_input}'")

        sc_rows = sidecar.get("n_rows")
        if sc_rows is not None:
            all_ok &= _print("Sidecar n_rows matches CSV",
                              sc_rows == len(df) or sc_rows == args.expect_rows // args.expect_shards,
                              f"sidecar={sc_rows}, csv={len(df)}")

        sc_hash = sidecar.get("input_sha256", "")
        if sc_hash:
            actual_hash = _sha256(TEST_CSV)
            all_ok &= _print("Input SHA256 matches test.csv",
                              sc_hash == actual_hash,
                              f"{'match' if sc_hash == actual_hash else 'MISMATCH'}")
    else:
        all_ok &= _print("Sidecar found", False, "missing")

    # ── 4. Shard completeness ────────────────────────────────────
    print("\n4. Shard completeness")
    if args.shard_dir_pattern:
        pattern = str(ROOT / args.shard_dir_pattern) if not Path(args.shard_dir_pattern).is_absolute() else args.shard_dir_pattern
        shard_dirs = sorted(d for d in glob.glob(pattern) if Path(d).is_dir())
        all_ok &= _print("Shard directory count", len(shard_dirs) == args.expect_shards,
                          f"{len(shard_dirs)} (expected {args.expect_shards})")

        total_shard_rows = 0
        for sd in shard_dirs:
            csvs = list(Path(sd).glob("*.csv"))
            if csvs:
                shard_df = pd.read_csv(csvs[0])
                total_shard_rows += len(shard_df)

        if total_shard_rows > 0:
            all_ok &= _print("Shard rows sum to expected",
                              total_shard_rows == args.expect_rows,
                              f"{total_shard_rows} (expected {args.expect_rows})")

        # Check shard logs for error indicators
        log_pattern = pattern.rstrip("*") + "*.log" if not pattern.endswith(".log") else pattern
        log_files = sorted(glob.glob(pattern + ".log") if not pattern.endswith("*") else
                           glob.glob(pattern.rstrip("*") + "*.log"))
        # Try simpler pattern
        base = args.shard_dir_pattern.rstrip("*")
        log_files = sorted(glob.glob(str(ROOT / f"{base}*.log")))
        errors_found = 0
        for lf in log_files:
            with open(lf) as f:
                content = f.read()
                if "Traceback" in content or "Error" in content.split("\n")[-5:]:
                    errors_found += 1
        _print("No tracebacks in shard logs", errors_found == 0,
               f"{errors_found} logs with errors" if errors_found else f"checked {len(log_files)} logs")
    else:
        _print("Shard check", True, "skipped (no --shard-dir-pattern)")

    # ── 5. Experiment log cross-check ────────────────────────────
    print("\n5. Experiment log cross-check")
    if LOG_PATH.exists():
        log_df = pd.read_csv(LOG_PATH)
        # Find rows matching this merged file
        matches = log_df[log_df["input_file"].astype(str).str.contains(merged_path.stem, na=False)]
        all_ok &= _print("Entry found in experiment_log.csv", len(matches) > 0,
                          f"{len(matches)} matching row(s)")

        if len(matches) > 0:
            entry = matches.iloc[-1]
            log_n = int(entry.get("n_evaluated", 0)) + int(entry.get("n_skipped", 0))
            all_ok &= _print("Log n matches expected", log_n == args.expect_rows,
                              f"log n_eval+n_skip={log_n}")

            log_split = entry.get("split", "")
            all_ok &= _print("Log split is 'test'", log_split == "test",
                              f"got '{log_split}'")

            log_exact = entry.get("exact_accuracy")
            if has_pred and has_target and log_exact is not None:
                pred_vals = pd.to_numeric(df[args.pred], errors="coerce")
                gt_vals = pd.to_numeric(df[args.target], errors="coerce")
                mask = pred_vals.notna() & gt_vals.notna()
                recomputed_exact = (pred_vals[mask] == gt_vals[mask]).mean()
                delta = abs(float(log_exact) - recomputed_exact)
                all_ok &= _print("Log exact_accuracy matches recompute",
                                  delta < 0.001,
                                  f"log={float(log_exact):.4f}, recomputed={recomputed_exact:.4f}, delta={delta:.6f}")
    else:
        _print("Experiment log", False, "not found")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED — investigate before proceeding")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
