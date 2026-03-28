"""Compare two run manifests and write a deterministic diff artifact.

Usage:
  .venv/bin/python experiments/diff_runs.py \
      --base data/runs/<base>.json \
      --new  data/runs/<new>.json

Optional semantic summary (1 additional model call):
  .venv/bin/python experiments/diff_runs.py \
      --base data/runs/<base>.json \
      --new  data/runs/<new>.json \
      --semantic
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments import tracking


def _semantic_summary(diff_payload: dict, model_id: str) -> dict:
    from src.llm import generate as llm_generate
    from src.llm.types import GenerationConfig

    prompt = (
        "You are comparing ML experiment runs. "
        "Summarize only behaviorally meaningful differences from this deterministic diff. "
        "Return 3 lines: summary, expected impact on quality/cost, and confidence.\n\n"
        f"{json.dumps(diff_payload, indent=2)}"
    )
    resp = llm_generate(
        prompt,
        model_id=model_id,
        config=GenerationConfig(temperature=0.0),
    )
    return {"model": model_id, "summary": (resp.text or "").strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a run-to-run diff JSON from two manifests.")
    parser.add_argument("--base", type=Path, required=True, help="Base manifest JSON path")
    parser.add_argument("--new", type=Path, required=True, help="New manifest JSON path")
    parser.add_argument("--output", type=Path, default=None, help="Output diff JSON path")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use one additional model call for semantic diff summary.",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default=None,
        help="Model for semantic summary (default: src.config.MODEL_ID).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base.exists():
        raise FileNotFoundError(f"--base not found: {args.base}")
    if not args.new.exists():
        raise FileNotFoundError(f"--new not found: {args.new}")

    base_manifest = tracking.load_json(args.base)
    new_manifest = tracking.load_json(args.new)
    diff_payload = tracking.build_manifest_diff(base_manifest, new_manifest)
    diff_payload["base_manifest_file"] = args.base.name
    diff_payload["new_manifest_file"] = args.new.name

    if args.semantic:
        import src.config as config

        model_id = args.semantic_model or config.MODEL_ID
        diff_payload["semantic_summary"] = _semantic_summary(diff_payload, model_id)
        diff_payload["semantic_diff_enabled"] = True
    else:
        diff_payload["semantic_diff_enabled"] = False

    output_path = args.output
    if output_path is None:
        out_name = f"{args.new.stem}__vs__{args.base.stem}.diff.json"
        output_path = args.new.with_name(out_name)

    output_path.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")
    print(f"Diff written to {output_path}")
    print(f"has_changes={diff_payload['has_changes']}  change_count={diff_payload['change_count']}")


if __name__ == "__main__":
    main()
