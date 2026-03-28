"""Extract ESI Handbook PDF to a single plaintext file for use as a system-level context prefix.

Run with:
    .venv/bin/python scripts/extract_esi_handbook_text.py [--pdf PATH] [--out PATH]

Uses the same page-filtering logic as build_esi_handbook_index.py (cover, TOC, near-empty,
non-content pages dropped). Output includes page-number markers (--- Page N ---) so the LLM
can cite specific pages. No embedding or FAISS — plain text only.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse filtering from build_esi_handbook_index (no code duplication)
_spec = importlib.util.spec_from_file_location(
    "build_esi_handbook_index",
    ROOT / "scripts" / "build_esi_handbook_index.py",
)
_build_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_module)

_extract_text_by_page = _build_module._extract_text_by_page
_filter_pages = _build_module._filter_pages

DEFAULT_PDF = Path(
    "/Users/ninotriandafilidis/Library/CloudStorage/"
    "GoogleDrive-ninot@stanford.edu/Shared drives/CS224n/Emergency_Severity_Index_Handbook.pdf"
)
DEFAULT_OUT = Path(
    "/Users/ninotriandafilidis/Library/CloudStorage/"
    "GoogleDrive-ninot@stanford.edu/Shared drives/CS224n/esi_handbook_text.txt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ESI Handbook PDF to plaintext for LLM context prefix"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Path to ESI Handbook PDF")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output plaintext file path")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    print("Extracting text by page …")
    pages = _extract_text_by_page(args.pdf)
    print(f"  Total pages: {len(pages)}")
    kept = _filter_pages(pages)
    print(f"  Pages kept: {len(kept)}")

    parts = []
    for num, text in kept:
        parts.append(f"\n--- Page {num} ---\n{text.strip()}")
    out_text = "\n".join(parts).lstrip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text, encoding="utf-8")
    print(f"Wrote {args.out} ({len(out_text):,} chars)")
    print("Done.")


if __name__ == "__main__":
    main()
