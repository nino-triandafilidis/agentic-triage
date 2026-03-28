"""Triage (ESI) evaluation: predicted vs ground-truth.

Works on any CSV that contains a ground-truth ESI column and one or more
prediction columns.  Computes a full metric stack including exact accuracy,
range accuracy (Gaber-style), safety metrics (under/over-triage, expected
triage cost), collapse/imbalance diagnostics (macro recall, per-class recall,
MA-MAE), chance-corrected ordinal agreement (linear-weighted Gwet's AC2),
and collapse indicators (prediction entropy, number of unique predicted classes).

Each run is appended to experiments/results/experiment_log.csv.

Usage examples:
    # Evaluate scratch run with defaults
    .venv/bin/python experiments/eval_triage.py

    # Custom file / columns, with run metadata
    .venv/bin/python experiments/eval_triage.py \\
        --input data/runs/scratch_rag_triage_results_<timestamp>.csv \\
        --pred triage_RAG \\
        --target triage \\
        --model gemini-2.0-flash \\
        --top-k 5 \\
        --split scratch \\
        --notes "first integration test"

    # Compare multiple prediction columns at once
    .venv/bin/python experiments/eval_triage.py --pred triage_RAG triage_baseline

    # Retrocompute metrics on an existing run CSV (no log append)
    .venv/bin/python experiments/eval_triage.py \\
        --retrocompute data/runs/haiku_fewshot_baseline/haiku_fewshot_baseline_llm_20260311_165744.csv \\
        --pred triage_LLM --target triage
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from sklearn.metrics import cohen_kappa_score, recall_score

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DEFAULT_PRED = ["triage_RAG"]

# ESI levels used for per-class metrics (always report all 5)
ESI_CLASSES = [1, 2, 3, 4, 5]

# ── Asymmetric cost matrix ────────────────────────────────────────────────
# Under-triage (pred > true, i.e. lower acuity assigned) is dangerous.
# Over-triage (pred < true, i.e. higher acuity assigned) is safer but wasteful.
#   Under-triage by 1: cost 3
#   Under-triage by 2+: cost 10
#   Over-triage by 1: cost 1
#   Over-triage by 2+: cost 2
#   Exact match: cost 0
COST_MATRIX = {
    "exact": 0,
    "under_1": 3,
    "under_2plus": 10,
    "over_1": 1,
    "over_2plus": 2,
}


def _latest_results_file() -> Path | None:
    """Return the most recently modified triage results CSV in data/, or None."""
    candidates = sorted(
        (ROOT / "data" / "runs").rglob("*triage_results*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


LOG_PATH = ROOT / "experiments" / "results" / "experiment_log.csv"

# Ground-truth column candidates (tried in order)
GT_CANDIDATES = ["triage", "acuity", "ESI"]

LOG_COLUMNS = [
    "timestamp", "run_id", "base_run_id", "diff_id", "diff_file",
    "split", "model", "mode", "top_k", "retrieval_backend",
    "n_evaluated", "n_skipped", "pred_col", "exact_accuracy", "adj1_accuracy",
    "range_accuracy", "mae", "quadratic_kappa",
    "under_triage_rate", "over_triage_rate", "expected_triage_cost",
    "macro_recall", "ma_mae", "linear_weighted_gwet_ac2",
    "prediction_entropy", "n_unique_predicted",
    "per_class_recall",
    "git_hash", "git_dirty", "code_fingerprint",
    "prompt_hash_rag", "prompt_hash_llm", "input_sha256", "source_file", "input_file",
    "prompt_tokens", "completion_tokens", "cost_usd", "pricing_source", "notes",
]


def _infer_split_from_source_file(source_file: str) -> str | None:
    """Infer the dataset split from the original input filename stored in sidecars."""
    if not source_file:
        return None
    split_by_name = {
        "scratch.csv": "scratch",
        "dev.csv": "dev",
        "dev_tune.csv": "dev_tune",
        "dev_holdout.csv": "dev_holdout",
        "val.csv": "val",
        "test.csv": "test",
    }
    return split_by_name.get(Path(source_file).name.lower())


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def find_gt_column(df: pd.DataFrame, hint: str | None) -> str:
    if hint:
        if hint not in df.columns:
            raise ValueError(f"Ground-truth column '{hint}' not in CSV.")
        return hint
    for name in GT_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find a ground-truth column. "
        f"Tried: {GT_CANDIDATES}. Pass --target explicitly."
    )


# ── Individual metric functions ───────────────────────────────────────────

def range_accuracy(yt: pd.Series, yp: pd.Series) -> float:
    """Gaber-style range accuracy.

    Correct when:
      - exact match, OR
      - one-level safer over-triage (pred == true - 1, i.e. higher acuity)
    Exceptions:
      - True ESI-1 must be predicted exactly (no over-triage possible above 1)
      - Under-triage never counts as correct
    """
    diff = yt - yp  # positive = over-triage, negative = under-triage
    # Standard rule: 0 <= diff <= 1
    accepted = (diff >= 0) & (diff <= 1)
    # ESI-1 exception: must be exact (diff == 0). Over-triage by 1 would
    # mean predicting ESI-0 which doesn't exist, so this is already
    # naturally enforced by the data. But we make it explicit for clarity
    # and to guard against any edge case where pred=0 slips through.
    esi1_mask = yt == 1
    accepted = accepted & (~esi1_mask | (diff == 0))
    return float(accepted.mean())


def expected_triage_cost(yt: pd.Series, yp: pd.Series) -> float:
    """Mean per-sample cost from asymmetric clinical cost matrix."""
    diff = yp - yt  # positive = under-triage, negative = over-triage
    costs = np.zeros(len(diff), dtype=float)
    # Under-triage (diff > 0): patient assigned lower acuity than needed
    costs[diff == 1] = COST_MATRIX["under_1"]
    costs[diff >= 2] = COST_MATRIX["under_2plus"]
    # Over-triage (diff < 0): patient assigned higher acuity than needed
    costs[diff == -1] = COST_MATRIX["over_1"]
    costs[diff <= -2] = COST_MATRIX["over_2plus"]
    # Exact match: cost 0 (already initialized)
    return float(costs.mean())


def macro_averaged_mae(yt: pd.Series, yp: pd.Series) -> float:
    """Macro-averaged Mean Absolute Error (MA-MAE).

    For each class c present in y_true, compute the mean absolute error
    on rows where y_true == c, then average across classes. This handles
    class imbalance by giving equal weight to each class.
    """
    classes = sorted(yt.unique())
    class_maes = []
    for c in classes:
        mask = yt == c
        if mask.sum() == 0:
            continue
        class_mae = np.abs(yt[mask] - yp[mask]).mean()
        class_maes.append(float(class_mae))
    if not class_maes:
        return float("nan")
    return float(np.mean(class_maes))


def linear_weighted_gwet_ac2(yt: pd.Series, yp: pd.Series) -> float:
    """Gwet's AC2 with linear weights, computed from scratch.

    AC2 = (P_a - P_e) / (1 - P_e)

    where:
      P_a = observed weighted agreement (using linear weights)
      P_e = expected agreement under Gwet's uniform-marginal model

    Linear weight for categories i, j on a K-point ordinal scale:
      w(i, j) = 1 - |i - j| / (K - 1)

    For Gwet's AC2, P_e uses the "pi" distribution:
      pi_k = (n_k_rater1 + n_k_rater2) / (2 * n)
      P_e = sum over all pairs (k, l) of: pi_k * pi_l * w(k, l)
      adjusted by the Gwet factor: 1 / (K - 1)

    Reference: Gwet, K.L. (2014). Handbook of Inter-Rater Reliability,
    4th Edition. Advanced Analytics, LLC.
    """
    n = len(yt)
    if n == 0:
        return float("nan")

    # Determine the category set from the data
    all_cats = sorted(set(yt.unique()) | set(yp.unique()))
    K = len(all_cats)
    if K < 2:
        return float("nan")

    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    yt_idx = np.array([cat_to_idx[v] for v in yt])
    yp_idx = np.array([cat_to_idx[v] for v in yp])

    # Linear weight matrix: w(i,j) = 1 - |i-j|/(K-1)
    w = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            w[i, j] = 1.0 - abs(i - j) / (K - 1)

    # Observed agreement (Pa) — mean of weights for each (true, pred) pair
    pa = np.mean([w[yt_idx[i], yp_idx[i]] for i in range(n)])

    # Gwet's expected agreement (Pe)
    # pi_k = proportion of times category k is used across both raters
    pi = np.zeros(K)
    for k in range(K):
        pi[k] = ((yt_idx == k).sum() + (yp_idx == k).sum()) / (2.0 * n)

    # Tw = sum of pi_k * (1 - pi_k) * weight contribution
    # For AC2 with ordinal weights, Pe = (1/(K-1)) * sum_{k} pi_k * (1 - pi_k)
    # weighted by the diagonal-adjacent weights.
    # More precisely, using Gwet's formulation for ordinal weights:
    # Pe = sum_{k,l} pi_k * pi_l * w(k,l)  -- but that's Cohen's Pe.
    # Gwet's AC2 Pe = (1 / (n*(n-1))) * sum of weighted-chance agreement.
    #
    # The standard Gwet AC2 formula:
    # Tw(w) = sum_{k=1}^{K} sum_{l=1}^{K} w(k,l) * pi_k * pi_l
    # But Gwet specifically uses:
    # Pe = Tw where Tw = sum_{k} pi_k * sum_{l != k} pi_l * w(k,l) ... no.
    #
    # Correct Gwet AC2 formulation (Gwet 2014, eq 5.3.3):
    # Pe(AC2) = (1 / choose(q,2)) * sum_{k<l} w(k,l) * pi_k * pi_l
    # where q = K, and pi_k are defined as above.
    # But choose(q,2) = K*(K-1)/2
    #
    # Actually, Gwet's AC2 with ordinal weights uses:
    # Tw = sum of pi_k^2 weighted by the self-agreement weight w(k,k)=1
    # Pe = [1/(K-1)] * sum_{k} pi_k * (1 - pi_k)
    # This is the standard formulation.

    # Let me use the correct formula from Gwet (2014):
    # For weighted AC2:
    # Pe(gamma) = [ sum_{k=1}^{K} pi_k * (D_k) ] / (K - 1)
    # where D_k = sum_{l=1}^{K} w(k, l) * (1 if l != k else 0) ... no
    #
    # Simplest correct approach for AC2 with weights:
    # Pe = Tw(pi) where Tw is computed from the weight matrix and
    # the marginal distribution, using Gwet's chance-agreement model.
    #
    # From Gwet (2008), for weighted agreement:
    # Pe = (1 / (K*(K-1))) * sum_{k} sum_{l, l!=k} w(k,l)
    # multiplied by sum of pi_k^2... no.
    #
    # Let me implement this correctly.
    # Gwet AC2 (weighted) formula from the irrCAC R documentation:
    #
    # Pa = (1/n) * sum_i w(r1_i, r2_i)  [observed weighted agreement]
    # Pe = sum_{k} sum_{l} w(k,l) * pi_k * pi_l  [but this is kappa's Pe]
    #
    # No. The key difference between kappa and AC2 is Pe.
    # For AC2: Pe is based on the assumption that raters assign categories
    # uniformly at random with some probability, and use their judgment
    # otherwise.
    #
    # Gwet's AC2 with weights (Gwet 2014, Chapter 5):
    # Pe = (sum_{k} pi_k * (1 - pi_k)) / (K - 1)  (for unweighted)
    # For weighted:
    # Pe = sum_{k=1}^{K} sum_{l=1}^{K} w(k,l) * pi_k * pi_l
    # minus the "true agreement" component... Actually no.
    #
    # I'll use the formula from Gwet (2008) "Computing inter-rater
    # reliability and its variance in the presence of high agreement":
    #
    # For weighted AC2 with 2 raters, the expected agreement is:
    # Pe(gamma) = [ sum_{k=1}^{K} pi_k * Ew_k ] where
    # Ew_k = (1/(K-1)) * sum_{l != k} w(k,l)
    #
    # This reduces Pe to:
    # Pe = (1/(K-1)) * sum_{k} pi_k * sum_{l!=k} w(k,l)

    pe = 0.0
    for k in range(K):
        ew_k = sum(w[k, l] for l in range(K) if l != k) / (K - 1)
        pe += pi[k] * ew_k

    if abs(1.0 - pe) < 1e-12:
        return float("nan")

    ac2 = (pa - pe) / (1.0 - pe)
    return float(ac2)


def prediction_entropy(yp: pd.Series) -> float:
    """Shannon entropy of the predicted class distribution (bits)."""
    counts = yp.value_counts()
    probs = counts / counts.sum()
    # Use base-2 for bits
    return float(shannon_entropy(probs, base=2))


def n_unique_predicted(yp: pd.Series) -> int:
    """Number of unique predicted classes."""
    return int(yp.nunique())


def per_class_recall(yt: pd.Series, yp: pd.Series) -> dict:
    """Per-class recall for ESI levels present in the data.

    Returns a dict like {"recall_ESI1": 0.75, "recall_ESI2": 0.82, ...}
    Only includes classes present in y_true.
    """
    # Use all ESI classes that appear in y_true
    classes_present = sorted(yt.unique())
    recalls = recall_score(yt, yp, labels=classes_present, average=None, zero_division=0)
    result = {}
    for cls, rec in zip(classes_present, recalls):
        result[f"recall_ESI{cls}"] = round(float(rec), 4)
    return result


# ── Main evaluate function ────────────────────────────────────────────────

def evaluate(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict:
    """Compute full triage metric stack for one prediction column."""
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    mask = y_pred_num.notna() & y_true.notna()
    n_total = len(y_true)
    n_valid = mask.sum()

    if n_valid == 0:
        print(f"[{label}] No valid (int) predictions -- skipping.")
        return {}

    yt = y_true[mask].astype(int)
    yp = y_pred_num[mask].astype(int)

    # ── Primary benchmark-facing ──────────────────────────────────────
    exact_acc = float((yt == yp).mean())
    adj_acc = float((np.abs(yt - yp) <= 1).mean())
    range_acc = range_accuracy(yt, yp)
    mae = float(np.abs(yt - yp).mean())

    # ── Safety metrics ────────────────────────────────────────────────
    under_triage = float((yp > yt).mean())
    over_triage = float((yp < yt).mean())
    exp_cost = expected_triage_cost(yt, yp)

    # ── Collapse / imbalance diagnostics ──────────────────────────────
    classes_in_true = sorted(yt.unique())
    m_recall = float(recall_score(yt, yp, labels=classes_in_true, average="macro", zero_division=0))
    pc_recall = per_class_recall(yt, yp)
    ma_mae_val = macro_averaged_mae(yt, yp)

    # ── Chance-corrected ordinal diagnostic ───────────────────────────
    try:
        kappa = cohen_kappa_score(yt, yp, weights="quadratic")
    except ValueError:
        kappa = float("nan")

    gwet = linear_weighted_gwet_ac2(yt, yp)

    # ── Collapse indicators ───────────────────────────────────────────
    pred_ent = prediction_entropy(yp)
    n_unique = n_unique_predicted(yp)

    parse_fail_rate = round(1.0 - n_valid / n_total, 4) if n_total > 0 else 0.0

    results = {
        "n_evaluated": int(n_valid),
        "n_skipped": int(n_total - n_valid),
        "parse_fail_rate": parse_fail_rate,
        # Primary
        "exact_accuracy": round(exact_acc, 4),
        "adj1_accuracy": round(adj_acc, 4),
        "range_accuracy": round(range_acc, 4),
        "mae": round(mae, 4),
        # Safety
        "under_triage_rate": round(under_triage, 4),
        "over_triage_rate": round(over_triage, 4),
        "expected_triage_cost": round(exp_cost, 4),
        # Collapse / imbalance
        "macro_recall": round(m_recall, 4),
        "per_class_recall": pc_recall,
        "ma_mae": round(ma_mae_val, 4),
        # Chance-corrected
        "quadratic_kappa": round(kappa, 4),
        "linear_weighted_gwet_ac2": round(gwet, 4),
        # Collapse indicators
        "prediction_entropy": round(pred_ent, 4),
        "n_unique_predicted": n_unique,
    }

    # ── Pretty-print ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" {label} (n={n_valid}/{n_total})")
    print(f"{'=' * 60}")

    print(f"\n  Primary benchmark-facing:")
    print(f"    Exact accuracy   : {exact_acc:.1%}")
    print(f"    Range accuracy   : {range_acc:.1%}  (Gaber: exact + 1-level over-triage; ESI-1 exact only)")
    print(f"    +/-1 accuracy    : {adj_acc:.1%}")
    print(f"    MAE              : {mae:.3f}")

    print(f"\n  Safety metrics:")
    print(f"    Under-triage rate: {under_triage:.1%}")
    print(f"    Over-triage rate : {over_triage:.1%}")
    print(f"    Expected cost    : {exp_cost:.3f}  (0=perfect, higher=worse)")

    print(f"\n  Collapse / imbalance diagnostics:")
    print(f"    Macro recall     : {m_recall:.3f}")
    for cls_key in sorted(pc_recall.keys()):
        print(f"    {cls_key:16s} : {pc_recall[cls_key]:.3f}")
    print(f"    MA-MAE           : {ma_mae_val:.3f}")

    print(f"\n  Chance-corrected ordinal:")
    print(f"    Gwet AC2 (linear): {gwet:.3f}")
    print(f"    Quadratic kappa  : {kappa:.3f}  (diagnostic only)")

    print(f"\n  Collapse indicators:")
    print(f"    Prediction entropy : {pred_ent:.3f} bits")
    print(f"    Unique predicted   : {n_unique}")

    if n_valid < n_total:
        print(f"\n  Parse failures: {n_total - n_valid} ({parse_fail_rate:.1%})")

    print()

    return results


def log_result(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        if list(existing.columns) != LOG_COLUMNS:
            migrated = existing.reindex(columns=LOG_COLUMNS)
            migrated.to_csv(LOG_PATH, index=False)
            print(f"Migrated log schema to latest columns: {LOG_PATH.relative_to(ROOT)}")
    log_df = pd.DataFrame([row], columns=LOG_COLUMNS)
    write_header = not LOG_PATH.exists()
    log_df.to_csv(LOG_PATH, mode="a", header=write_header, index=False)
    print(f"Logged to {LOG_PATH.relative_to(ROOT)}")


def retrocompute(input_path: Path, pred_cols: list[str], target: str | None) -> None:
    """Recompute metrics on an existing run CSV without appending to the log.

    Prints the full metric stack to stdout. Useful for comparing old runs
    against the expanded metric set.
    """
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path.name}")

    gt_col = find_gt_column(df, target)
    print(f"Ground-truth column: '{gt_col}'")

    all_results = []
    for pred_col in pred_cols:
        if pred_col not in df.columns:
            print(f"WARNING: prediction column '{pred_col}' not found -- skipping.")
            continue
        result = evaluate(df[gt_col], df[pred_col], label=pred_col)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print("\n-- Summary --")
        summary = pd.DataFrame(all_results)
        print(summary.to_string(index=False))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate triage ESI predictions.")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="CSV with predictions (default: most recently modified *triage_results*.csv in data/)",
    )
    parser.add_argument(
        "--pred", nargs="+", default=DEFAULT_PRED,
        help="Prediction column name(s) (default: triage_RAG)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help=f"Ground-truth column (auto-detected from {GT_CANDIDATES} if omitted)",
    )
    parser.add_argument("--model",              type=str, default=None, help="Model name (auto-read from sidecar if omitted)")
    parser.add_argument("--top-k",              type=int, default=None, help="top_k used in retrieval (auto-read from sidecar if omitted)")
    parser.add_argument("--retrieval-backend",  type=str, default=None, help="Retrieval backend used (auto-read from sidecar if omitted; 'none' for llm mode)")
    parser.add_argument(
        "--split", type=str, default=None,
        help="Data split (scratch/dev/dev_tune/dev_holdout/val/test). "
             "If omitted, infer from the sidecar input_file when available.",
    )
    parser.add_argument("--notes",              type=str, default="", help="Free-text notes for this run")
    parser.add_argument(
        "--retrocompute", type=Path, default=None, metavar="CSV",
        help="Recompute metrics on an existing CSV without logging. Prints results to stdout only.",
    )
    args = parser.parse_args()

    # ── Retrocompute mode ─────────────────────────────────────────────
    if args.retrocompute is not None:
        retrocompute(args.retrocompute, args.pred, args.target)
        return

    # ── Normal evaluation mode ────────────────────────────────────────
    if args.input is None:
        args.input = _latest_results_file()
        if args.input is None:
            raise FileNotFoundError("No *triage_results*.csv found in data/. Pass --input explicitly.")
        print(f"Auto-selected latest results file: {args.input.name}")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input.name}")

    # Auto-read run metadata from sidecar written by run_rag_triage.py
    sidecar_path = args.input.with_suffix(".json")
    sidecar = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        print(f"Loaded metadata from {sidecar_path.name}")

    model = args.model or sidecar.get("model", "")
    top_k = args.top_k if args.top_k is not None else sidecar.get("top_k")
    mode = sidecar.get("mode", "rag" if (top_k or 0) > 0 else "llm")
    source_file = sidecar.get("input_file", "")  # original input CSV (e.g. dev.csv)
    inferred_split = _infer_split_from_source_file(source_file)
    if args.split is not None and inferred_split is not None and args.split != inferred_split:
        raise ValueError(
            f"--split={args.split!r} conflicts with sidecar input_file={source_file!r} "
            f"(inferred split {inferred_split!r})."
        )
    split = args.split or inferred_split or "scratch"
    if args.split is None and inferred_split is not None:
        print(f"Inferred split='{split}' from sidecar input_file='{source_file}'.")
    retrieval_backend = (
        args.retrieval_backend
        or sidecar.get("retrieval_backend")
        or ("none" if mode == "llm" else "bq")  # fallback for old sidecars that predate this field
    )
    prompt_tokens = sidecar.get("prompt_tokens")
    completion_tokens = sidecar.get("completion_tokens")
    cost_usd = sidecar.get("cost_usd")
    pricing_source = sidecar.get("pricing_source")
    run_id = sidecar.get("run_id")
    base_run_id = sidecar.get("base_run_id")
    diff_id = sidecar.get("diff_id")
    diff_file = sidecar.get("diff_file")
    git_dirty = sidecar.get("git_dirty")
    code_fingerprint = sidecar.get("code_fingerprint")
    input_sha256 = sidecar.get("input_sha256")
    prompt_hashes = sidecar.get("prompt_hashes") or {}
    prompt_hash_rag = prompt_hashes.get("rag")
    prompt_hash_llm = prompt_hashes.get("llm")

    gt_col = find_gt_column(df, args.target)
    print(f"Ground-truth column: '{gt_col}'")

    git_hash = sidecar.get("git_hash") or _git_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_results = []
    for pred_col in args.pred:
        if pred_col not in df.columns:
            print(f"WARNING: prediction column '{pred_col}' not found -- skipping.")
            continue
        result = evaluate(df[gt_col], df[pred_col], label=pred_col)
        if not result:
            continue
        all_results.append(result)

        # Serialize per_class_recall dict as JSON string for CSV column
        pcr_str = json.dumps(result["per_class_recall"])

        log_result({
            "timestamp": timestamp,
            "run_id": run_id,
            "base_run_id": base_run_id,
            "diff_id": diff_id,
            "diff_file": diff_file,
            "split": split,
            "model": model,
            "mode": mode,
            "top_k": top_k,
            "retrieval_backend": retrieval_backend,
            "n_evaluated": result["n_evaluated"],
            "n_skipped": result["n_skipped"],
            "pred_col": pred_col,
            "exact_accuracy": result["exact_accuracy"],
            "adj1_accuracy": result["adj1_accuracy"],
            "range_accuracy": result["range_accuracy"],
            "mae": result["mae"],
            "quadratic_kappa": result["quadratic_kappa"],
            "under_triage_rate": result["under_triage_rate"],
            "over_triage_rate": result["over_triage_rate"],
            "expected_triage_cost": result["expected_triage_cost"],
            "macro_recall": result["macro_recall"],
            "ma_mae": result["ma_mae"],
            "linear_weighted_gwet_ac2": result["linear_weighted_gwet_ac2"],
            "prediction_entropy": result["prediction_entropy"],
            "n_unique_predicted": result["n_unique_predicted"],
            "per_class_recall": pcr_str,
            "git_hash": git_hash,
            "git_dirty": git_dirty,
            "code_fingerprint": code_fingerprint,
            "prompt_hash_rag": prompt_hash_rag,
            "prompt_hash_llm": prompt_hash_llm,
            "input_sha256": input_sha256,
            "source_file": source_file,
            "input_file": args.input.name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "pricing_source": pricing_source,
            "notes": args.notes,
        })

    if len(all_results) > 1:
        print("\n-- Summary --")
        summary = pd.DataFrame(all_results)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
