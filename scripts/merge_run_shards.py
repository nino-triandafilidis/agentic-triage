"""Merge sharded experiment outputs into a single validated run artifact.

When query_strategy_sweep.py is run multiple times with non-overlapping
--skip-rows / --n-rows ranges, this script merges the resulting CSVs and
sidecar JSONs into one combined output.

Usage:
    python scripts/merge_run_shards.py \
        --inputs data/runs/W1_rag_shard0/shard0.csv data/runs/W1_rag_shard1/shard1.csv \
        --output data/runs/W1_rag/W1_rag_combined_1000.csv \
        --expect-rows 1000

    python scripts/merge_run_shards.py \
        --inputs shard*.csv --output merged.csv --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Sidecar fields
# ---------------------------------------------------------------------------

# These must match across shards (always fatal on mismatch)
CRITICAL_INVARIANTS = ("input_sha256", "model", "mode")

# These should match (fatal only in --strict mode, warn otherwise)
SOFT_INVARIANTS = (
    "prompt_template",
    "top_k",
    "retrieval_backend",
    "input_file",
    "git_hash",
    "code_fingerprint",
    "context_chars",
    "temperature",
)

# These are summed across shards
SUMMABLE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "thinking_tokens",
    "generation_cost_usd",
    "rewrite_cost_usd",
    "cost_usd",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sidecar(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _check_invariants(
    sidecars: list[dict],
    paths: list[Path],
    fields: tuple[str, ...],
    strict: bool,
    fatal: bool,
) -> list[str]:
    """Check that fields match across all sidecars. Return list of warnings."""
    warnings = []
    ref = sidecars[0]
    for field in fields:
        ref_val = ref.get(field)
        for i, sc in enumerate(sidecars[1:], 1):
            val = sc.get(field)
            if val != ref_val:
                msg = (
                    f"Sidecar mismatch: {field!r} = {ref_val!r} in "
                    f"{paths[0].name} but {val!r} in {paths[i].name}"
                )
                if fatal or strict:
                    print(f"FATAL: {msg}", file=sys.stderr)
                    sys.exit(1)
                else:
                    warnings.append(msg)
    return warnings


def _compute_ranges(sidecars: list[dict]) -> list[tuple[int, int]] | None:
    """Extract (skip_rows, skip_rows + n_rows) ranges from sidecars."""
    ranges = []
    for sc in sidecars:
        skip = sc.get("skip_rows")
        n = sc.get("n_rows")
        if skip is None or n is None:
            return None
        ranges.append((int(skip), int(skip) + int(n)))
    return ranges


def _check_range_overlap(ranges: list[tuple[int, int]], paths: list[Path]) -> None:
    """Fail if any shard ranges overlap."""
    indexed = sorted(zip(ranges, paths), key=lambda x: x[0][0])
    for i in range(len(indexed) - 1):
        (_, end_a), path_a = indexed[i]
        (start_b, _), path_b = indexed[i + 1]
        if end_a > start_b:
            print(
                f"FATAL: Overlapping row ranges: {path_a.name} ends at row "
                f"{end_a} but {path_b.name} starts at row {start_b}",
                file=sys.stderr,
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge sharded sweep outputs into one validated artifact."
    )
    p.add_argument(
        "--inputs", nargs="+", type=Path, required=True,
        help="Shard CSV files to merge",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Path for the merged CSV",
    )
    p.add_argument(
        "--expect-rows", type=int, default=None,
        help="Expected total row count (fail if mismatch)",
    )
    p.add_argument(
        "--id-col", default="stay_id",
        help="Column used for duplicate checking (default: stay_id)",
    )
    p.add_argument(
        "--sidecars", nargs="*", default=None,
        help="Sidecar JSON files (default: auto-detect from CSV paths)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Fail on any sidecar metadata mismatch",
    )
    p.add_argument(
        "--sort-by", choices=["skip_rows", "stay_id"], default="skip_rows",
        help="How to order merged rows (default: skip_rows)",
    )
    args = p.parse_args()

    csv_paths = args.inputs
    warnings_list: list[str] = []

    # --- Load CSVs -----------------------------------------------------------

    if len(csv_paths) < 2:
        print("FATAL: Need at least 2 shard CSVs to merge.", file=sys.stderr)
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        if not path.exists():
            print(f"FATAL: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(path)
        if len(df) == 0:
            print(f"FATAL: Empty shard: {path}", file=sys.stderr)
            sys.exit(1)
        frames.append(df)

    # --- Schema check --------------------------------------------------------

    ref_cols = list(frames[0].columns)
    for i, df in enumerate(frames[1:], 1):
        if list(df.columns) != ref_cols:
            extra = set(df.columns) - set(ref_cols)
            missing = set(ref_cols) - set(df.columns)
            print(
                f"FATAL: Schema mismatch in {csv_paths[i].name}. "
                f"Extra: {extra or 'none'}, Missing: {missing or 'none'}",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- ID column check -----------------------------------------------------

    id_col = args.id_col
    if id_col not in ref_cols:
        print(
            f"FATAL: ID column {id_col!r} not found in CSVs. "
            f"Available: {ref_cols[:5]}...",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Duplicate ID check --------------------------------------------------

    all_ids = pd.concat([df[id_col] for df in frames], ignore_index=True)
    dupes = all_ids[all_ids.duplicated()]
    if len(dupes) > 0:
        print(
            f"FATAL: {len(dupes)} duplicate {id_col} values across shards: "
            f"{dupes.head(5).tolist()}...",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Sidecar loading -----------------------------------------------------

    sidecar_paths: list[Path]
    if args.sidecars is not None:
        sidecar_paths = [Path(s) for s in args.sidecars]
        if len(sidecar_paths) != len(csv_paths):
            print(
                f"FATAL: {len(sidecar_paths)} sidecars provided for "
                f"{len(csv_paths)} CSVs.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        sidecar_paths = [p.with_suffix(".json") for p in csv_paths]

    sidecars: list[dict | None] = []
    for sp in sidecar_paths:
        sc = _load_sidecar(sp)
        if sc is None:
            warnings_list.append(f"Sidecar not found: {sp}")
        sidecars.append(sc)

    valid_sidecars = [sc for sc in sidecars if sc is not None]
    valid_sidecar_paths = [sp for sp, sc in zip(sidecar_paths, sidecars) if sc is not None]
    has_sidecars = len(valid_sidecars) == len(csv_paths)

    # --- Sidecar validation --------------------------------------------------

    if has_sidecars:
        # Critical invariants: always fatal
        _check_invariants(
            valid_sidecars, valid_sidecar_paths,
            CRITICAL_INVARIANTS, strict=True, fatal=True,
        )
        # Soft invariants: fatal only in strict mode
        ws = _check_invariants(
            valid_sidecars, valid_sidecar_paths,
            SOFT_INVARIANTS, strict=args.strict, fatal=False,
        )
        warnings_list.extend(ws)

        # Range overlap check
        ranges = _compute_ranges(valid_sidecars)
        if ranges is not None:
            _check_range_overlap(ranges, csv_paths)

    # --- Ordering and merge --------------------------------------------------

    sort_by = args.sort_by
    if sort_by == "skip_rows" and has_sidecars:
        ranges = _compute_ranges(sidecars)
        if ranges is not None:
            order = sorted(range(len(frames)), key=lambda i: ranges[i][0])
            frames = [frames[i] for i in order]
            csv_paths = [csv_paths[i] for i in order]
            sidecar_paths = [sidecar_paths[i] for i in order]
            sidecars = [sidecars[i] for i in order]
            valid_sidecars = [sc for sc in sidecars if sc is not None]
            valid_sidecar_paths = [sp for sp, sc in zip(sidecar_paths, sidecars) if sc is not None]
        else:
            warnings_list.append(
                "skip_rows/n_rows not in sidecars; falling back to stay_id order"
            )
            sort_by = "stay_id"
    elif sort_by == "skip_rows" and not has_sidecars:
        warnings_list.append(
            "Missing sidecar(s): cannot sort by skip_rows; falling back to stay_id order"
        )
        sort_by = "stay_id"

    merged = pd.concat(frames, ignore_index=True)

    if sort_by == "stay_id":
        merged = merged.sort_values(id_col).reset_index(drop=True)

    # --- Row count check -----------------------------------------------------

    if args.expect_rows is not None and len(merged) != args.expect_rows:
        print(
            f"FATAL: Expected {args.expect_rows} rows but got {len(merged)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Write merged CSV ----------------------------------------------------

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    # --- Write merged sidecar ------------------------------------------------

    merged_sidecar_path = args.output.with_suffix(".json")
    merged_sidecar: dict = {
        "manifest_version": "1.2",
        "merge_type": "sharded_run_merge",
        "merged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_shard_count": len(csv_paths),
        "source_csvs": [str(p) for p in csv_paths],
        "source_sidecars": [str(p) for p in valid_sidecar_paths],
        "source_run_ids": [sc.get("run_id") for sc in valid_sidecars],
        "n_rows": len(merged),
        "skip_rows": None,
        "output_file": args.output.name,
        "output_path": str(args.output.resolve()),
    }

    # Inherit invariant fields from first sidecar
    if has_sidecars:
        ref_sc = valid_sidecars[0]
        for field in CRITICAL_INVARIANTS + SOFT_INVARIANTS:
            if field in ref_sc:
                merged_sidecar[field] = ref_sc[field]

        # Sum numeric fields
        for field in SUMMABLE_FIELDS:
            vals = [sc.get(field) for sc in valid_sidecars]
            if all(v is not None for v in vals):
                merged_sidecar[field] = round(sum(vals), 6)
            else:
                merged_sidecar[field] = None

        # Cost per row
        total_cost = merged_sidecar.get("cost_usd")
        if total_cost is not None and len(merged) > 0:
            merged_sidecar["cost_per_row_usd"] = round(total_cost / len(merged), 6)
        else:
            merged_sidecar["cost_per_row_usd"] = None

        # Pricing source
        merged_sidecar["pricing_source"] = ref_sc.get("pricing_source")

    # Metrics must be recomputed on the merged set -- do not average
    merged_sidecar["metrics"] = None

    merged_sidecar_path.write_text(
        json.dumps(merged_sidecar, indent=2), encoding="utf-8"
    )

    # --- Error count ---------------------------------------------------------

    error_count = 0
    if "error" in merged.columns:
        error_count = int(merged["error"].notna().sum())

    # --- Summary -------------------------------------------------------------

    print("=" * 60)
    print("Merge summary")
    print("=" * 60)
    print(f"  Shards:           {len(csv_paths)}")
    print(f"  Merged rows:      {len(merged)}")
    if args.expect_rows:
        print(f"  Expected rows:    {args.expect_rows} -- OK")
    print(f"  ID column:        {id_col}")
    print(f"  Duplicate check:  PASSED")
    print(f"  Schema check:     PASSED")
    if has_sidecars:
        print(f"  Metadata check:   PASSED")
    else:
        missing_count = len(csv_paths) - len(valid_sidecars)
        print(f"  Metadata check:   {missing_count} sidecar(s) missing")
    if error_count > 0:
        print(f"  Rows with errors: {error_count}")
    print(f"  Output CSV:       {args.output}")
    print(f"  Output sidecar:   {merged_sidecar_path}")
    if warnings_list:
        print()
        print("Warnings:")
        for w in warnings_list:
            print(f"  - {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
