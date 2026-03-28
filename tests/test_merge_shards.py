"""Tests for scripts/merge_run_shards.py."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "merge_run_shards.py"
PYTHON = sys.executable


def _make_sidecar(path: Path, **overrides) -> None:
    """Write a minimal sidecar JSON."""
    sc = {
        "manifest_version": "1.2",
        "run_id": f"test_{path.stem}",
        "model": "gemini-2.5-flash",
        "mode": "llm",
        "input_sha256": "abc123",
        "prompt_template": "fewshot",
        "top_k": 0,
        "retrieval_backend": "none",
        "input_file": "dev_tune.csv",
        "git_hash": "deadbeef",
        "code_fingerprint": "fp000",
        "context_chars": 8000,
        "temperature": 0.0,
        "skip_rows": 0,
        "n_rows": 5,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "thinking_tokens": 200,
        "generation_cost_usd": 0.01,
        "rewrite_cost_usd": 0.0,
        "cost_usd": 0.01,
        "pricing_source": "fallback_static",
        "metrics": {"exact_accuracy": 0.5, "quadratic_kappa": 0.3},
    }
    sc.update(overrides)
    path.write_text(json.dumps(sc, indent=2))


def _make_shard_csv(path: Path, stay_ids: list[int], pred_col: str = "triage_LLM") -> None:
    """Write a minimal shard CSV."""
    df = pd.DataFrame({
        "stay_id": stay_ids,
        "text": [f"patient_{i}" for i in stay_ids],
        f"{pred_col}_raw": [f"ESI {(i % 5) + 1}" for i in stay_ids],
        pred_col: [(i % 5) + 1 for i in stay_ids],
        "error": [None] * len(stay_ids),
    })
    df.to_csv(path, index=False)


def _run_merge(args: list[str], expect_fail: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [PYTHON, str(SCRIPT)] + args,
        capture_output=True, text=True,
    )
    if expect_fail:
        assert result.returncode != 0, f"Expected failure but got:\n{result.stdout}"
    else:
        assert result.returncode == 0, f"Merge failed:\n{result.stderr}"
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path(tmp_path):
    """2 shards, valid sidecars, --expect-rows."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3, 4, 5])
    _make_shard_csv(shard1, [6, 7, 8, 9, 10])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=0, n_rows=5,
                  prompt_tokens=500, completion_tokens=50, cost_usd=0.005)
    _make_sidecar(shard1.with_suffix(".json"), skip_rows=5, n_rows=5,
                  prompt_tokens=600, completion_tokens=60, cost_usd=0.006)

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
        "--expect-rows", "10",
    ])

    assert out.exists()
    merged = pd.read_csv(out)
    assert len(merged) == 10
    assert list(merged["stay_id"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Check sidecar sums
    sc = json.loads(out.with_suffix(".json").read_text())
    assert sc["merge_type"] == "sharded_run_merge"
    assert sc["n_rows"] == 10
    assert sc["prompt_tokens"] == 1100
    assert sc["completion_tokens"] == 110
    assert sc["cost_usd"] == pytest.approx(0.011)
    assert sc["metrics"] is None
    assert "PASSED" in result.stdout


def test_overlapping_ranges(tmp_path):
    """Overlapping skip_rows ranges should fail."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3, 4, 5])
    _make_shard_csv(shard1, [6, 7, 8, 9, 10])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=0, n_rows=6)
    _make_sidecar(shard1.with_suffix(".json"), skip_rows=5, n_rows=5)

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ], expect_fail=True)
    assert "Overlapping" in result.stderr


def test_duplicate_ids(tmp_path):
    """Duplicate stay_id across shards should fail."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3])
    _make_shard_csv(shard1, [3, 4, 5])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=0, n_rows=3)
    _make_sidecar(shard1.with_suffix(".json"), skip_rows=3, n_rows=3)

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ], expect_fail=True)
    assert "duplicate" in result.stderr.lower()


def test_schema_mismatch(tmp_path):
    """Different columns across shards should fail."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3])
    # Write shard1 with extra column
    df = pd.DataFrame({
        "stay_id": [4, 5, 6],
        "text": ["a", "b", "c"],
        "triage_LLM_raw": ["ESI 1", "ESI 2", "ESI 3"],
        "triage_LLM": [1, 2, 3],
        "error": [None, None, None],
        "extra_col": [0, 0, 0],
    })
    df.to_csv(shard1, index=False)

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ], expect_fail=True)
    assert "Schema mismatch" in result.stderr


def test_missing_sidecar_warns(tmp_path):
    """Missing sidecar with auto-detect should warn but succeed."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3])
    _make_shard_csv(shard1, [4, 5, 6])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=0, n_rows=3)
    # No sidecar for shard1

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
        "--sort-by", "stay_id",
    ])
    assert out.exists()
    assert "missing" in result.stdout.lower()


def test_model_mismatch_fails(tmp_path):
    """Different model values should fail even without --strict."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3])
    _make_shard_csv(shard1, [4, 5, 6])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=0, n_rows=3,
                  model="gemini-2.5-flash")
    _make_sidecar(shard1.with_suffix(".json"), skip_rows=3, n_rows=3,
                  model="claude-haiku-4-5")

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ], expect_fail=True)
    assert "model" in result.stderr.lower()


def test_expect_rows_wrong(tmp_path):
    """Wrong --expect-rows should fail."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3])
    _make_shard_csv(shard1, [4, 5, 6])

    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
        "--expect-rows", "7",
        "--sort-by", "stay_id",
    ], expect_fail=True)
    assert "Expected 7" in result.stderr


def test_shard_ordering_by_skip_rows(tmp_path):
    """Shards provided in reverse order should be sorted by skip_rows."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    # shard0 has later rows, shard1 has earlier rows
    _make_shard_csv(shard0, [6, 7, 8, 9, 10])
    _make_shard_csv(shard1, [1, 2, 3, 4, 5])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=5, n_rows=5,
                  run_id="run_shard0")
    _make_sidecar(shard1.with_suffix(".json"), skip_rows=0, n_rows=5,
                  run_id="run_shard1")

    # Pass in "wrong" order: shard0 (skip=5) first, shard1 (skip=0) second
    _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ])

    merged = pd.read_csv(out)
    # Should be reordered by skip_rows: shard1 first, then shard0
    assert list(merged["stay_id"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # P2 fix: sidecar provenance must also be reordered
    sc = json.loads(out.with_suffix(".json").read_text())
    # source_csvs should be [shard1, shard0] (sorted by skip_rows)
    assert sc["source_csvs"][0].endswith("shard1.csv")
    assert sc["source_csvs"][1].endswith("shard0.csv")
    # source_sidecars must match the same order
    assert sc["source_sidecars"][0].endswith("shard1.json")
    assert sc["source_sidecars"][1].endswith("shard0.json")
    # source_run_ids must match
    assert sc["source_run_ids"] == ["run_shard1", "run_shard0"]


def test_missing_sidecar_falls_back_to_stay_id(tmp_path):
    """Missing sidecar with --sort-by skip_rows should fall back to stay_id."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    # shard0 has later IDs, shard1 has earlier IDs
    _make_shard_csv(shard0, [6, 7, 8])
    _make_shard_csv(shard1, [1, 2, 3])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=3, n_rows=3)
    # No sidecar for shard1 — missing

    # Default sort is skip_rows, but missing sidecar should force stay_id
    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ])

    merged = pd.read_csv(out)
    # Should be sorted by stay_id, not input order
    assert list(merged["stay_id"]) == [1, 2, 3, 6, 7, 8]
    assert "falling back" in result.stdout.lower()


def test_strict_mode_fails_on_soft_mismatch(tmp_path):
    """--strict should fail on soft invariant mismatches like git_hash."""
    shard0 = tmp_path / "shard0.csv"
    shard1 = tmp_path / "shard1.csv"
    out = tmp_path / "merged.csv"

    _make_shard_csv(shard0, [1, 2, 3])
    _make_shard_csv(shard1, [4, 5, 6])
    _make_sidecar(shard0.with_suffix(".json"), skip_rows=0, n_rows=3,
                  git_hash="aaa")
    _make_sidecar(shard1.with_suffix(".json"), skip_rows=3, n_rows=3,
                  git_hash="bbb")

    # Without --strict: should warn but succeed
    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
    ])
    assert "git_hash" in result.stdout.lower() or "warning" in result.stdout.lower()

    # With --strict: should fail
    result = _run_merge([
        "--inputs", str(shard0), str(shard1),
        "--output", str(out),
        "--strict",
    ], expect_fail=True)
    assert "git_hash" in result.stderr.lower()
