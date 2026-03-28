"""Run manifest and run-diff helpers for experiment analytics tracking.

This module is intentionally lightweight and local-only:
- no cloud/API calls
- deterministic fingerprints for prompts/code/config/input
- run-to-run diffs that can be linked to evaluation metrics
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TRACKED_FILES = [
    "experiments/run_rag_triage.py",
    "experiments/eval_triage.py",
    "experiments/query_strategy_sweep.py",
    "src/config.py",
    "src/rag/retrieval.py",
    "src/rag/pipeline.py",
    "src/rag/generation.py",
    "src/rag/query_agents.py",
    "src/rag/agentic_pipeline.py",
    "src/rag/triage_core.py",
]

MANIFEST_EPHEMERAL_KEYS = {
    "timestamp",
    "timestamp_start_utc",
    "timestamp_end_utc",
    "run_id",
    "base_run_id",
    "diff_id",
    "diff_file",
    "base_manifest_file",
    "output_file",
    "output_path",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(root: Path, args: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="replace")
    except Exception:
        return default


def collect_git_state(root: Path) -> dict[str, Any]:
    """Collect a deterministic fingerprint of current git state."""
    head = _run_git(root, ["rev-parse", "--short", "HEAD"], default="unknown").strip()
    status = _run_git(root, ["status", "--porcelain"], default="")
    unstaged = _run_git(root, ["diff", "--no-color"], default="")
    staged = _run_git(root, ["diff", "--cached", "--no-color"], default="")
    untracked = _run_git(root, ["ls-files", "--others", "--exclude-standard"], default="")

    dirty_paths = []
    for line in status.splitlines():
        if len(line) >= 4:
            dirty_paths.append(line[3:])

    worktree_material = "\n".join(
        [
            "==STAGED==",
            staged,
            "==UNSTAGED==",
            unstaged,
            "==UNTRACKED==",
            untracked,
        ]
    )

    return {
        "git_hash": head,
        "git_dirty": bool(status.strip()),
        "git_dirty_paths": sorted(dirty_paths),
        "git_status_sha256": sha256_text(status),
        "git_worktree_diff_sha256": sha256_text(worktree_material),
    }


def compute_file_hashes(root: Path, rel_paths: list[str]) -> tuple[dict[str, str], list[str]]:
    file_hashes: dict[str, str] = {}
    missing: list[str] = []
    for rel in rel_paths:
        abs_path = root / rel
        if abs_path.exists():
            file_hashes[rel] = sha256_file(abs_path)
        else:
            missing.append(rel)
    return file_hashes, missing


def generate_run_id(prefix: str = "triage") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    nonce = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{nonce}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_manifest(runs_dir: Path) -> Path | None:
    """Return the latest triage sidecar JSON if present."""
    candidates = sorted(
        runs_dir.glob("*_triage_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = load_json(path)
            if isinstance(payload, dict):
                return path
        except Exception:
            continue
    return None


def build_manifest_diff(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic run-to-run diff object."""
    scalar_changes = []
    all_scalar_keys = sorted(set(base.keys()) | set(new.keys()))
    for key in all_scalar_keys:
        if key in MANIFEST_EPHEMERAL_KEYS:
            continue
        base_value = base.get(key)
        new_value = new.get(key)
        # Structured fields are compared in dedicated sections below.
        if key in {"prompt_hashes", "code_files", "prompt_templates"}:
            continue
        if base_value != new_value:
            scalar_changes.append({
                "field": key,
                "before": base_value,
                "after": new_value,
            })

    prompt_changes = []
    base_prompts = base.get("prompt_hashes", {}) or {}
    new_prompts = new.get("prompt_hashes", {}) or {}
    for key in sorted(set(base_prompts.keys()) | set(new_prompts.keys())):
        b = base_prompts.get(key)
        n = new_prompts.get(key)
        if b != n:
            prompt_changes.append({"prompt": key, "before": b, "after": n})

    code_changes = []
    base_code = base.get("code_files", {}) or {}
    new_code = new.get("code_files", {}) or {}
    for key in sorted(set(base_code.keys()) | set(new_code.keys())):
        b = base_code.get(key)
        n = new_code.get(key)
        if b != n:
            code_changes.append({"file": key, "before": b, "after": n})

    categories = []
    if prompt_changes:
        categories.append("prompt_change")
    if code_changes:
        categories.append("code_change")
    if any(c["field"] in {"input_file", "input_sha256"} for c in scalar_changes):
        categories.append("dataset_change")
    if any(c["field"] in {"model", "top_k", "mode", "retrieval_backend", "context_chars"} for c in scalar_changes):
        categories.append("config_change")
    if not categories and scalar_changes:
        categories.append("metadata_change")

    diff_core = {
        "base_run_id": base.get("run_id"),
        "new_run_id": new.get("run_id"),
        "scalar_changes": scalar_changes,
        "prompt_changes": prompt_changes,
        "code_changes": code_changes,
        "change_categories": categories,
    }

    diff_id = sha256_json(diff_core)[:16]
    return {
        "diff_version": "1.0",
        "diff_id": diff_id,
        "base_run_id": base.get("run_id"),
        "new_run_id": new.get("run_id"),
        "has_changes": bool(scalar_changes or prompt_changes or code_changes),
        "change_categories": categories,
        "scalar_changes": scalar_changes,
        "prompt_changes": prompt_changes,
        "code_changes": code_changes,
        "change_count": len(scalar_changes) + len(prompt_changes) + len(code_changes),
        "created_at_utc": utc_now_iso(),
    }

