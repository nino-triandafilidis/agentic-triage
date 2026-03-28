"""Build a standalone FAISS index from the ESI (Emergency Severity Index) Handbook PDF.

Run with:
    .venv/bin/python scripts/build_esi_handbook_index.py [--pdf PATH] [--out DIR]

Steps:
  1. Parse the PDF, drop pages not useful for ESI prediction (cover, TOC, blank, acknowledgments).
  2. Chunk retained text for retrieval (section-aware, ~500 tokens with overlap).
  3. Embed chunks with Vertex AI text-embedding-005 (same as main FAISS corpus).
  4. Build a standalone FAISS IndexFlatIP and write index + metadata to --out.

Output (in --out directory):
  - index.faiss       — FAISS index (cosine / inner product)
  - chunks.json       — chunk_id, text, page_start, page_end
  - manifest.json     — checksums, embedding_model, num_chunks

Cost: Vertex AI embedding only; ~1–5k chunks ≈ a few hundred embed calls (low cost).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── PDF parsing ───────────────────────────────────────────────────────

def _extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (1-based page number, text)."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.append((i + 1, text))
    return out


# Pages we typically want to drop for ESI decision-support (no clinical content)
_DROP_PAGE_PATTERNS = re.compile(
    r"^(?:\s*(?:acknowledgments?|about the authors?|copyright|table of contents|"
    r"foreword|preface|introduction\s*$|appendix\s+[a-z]\s*$|references\s*$|"
    r"index\s*$|blank\s*$))",
    re.IGNORECASE | re.MULTILINE,
)
_MIN_PAGE_CHARS = 150  # skip nearly empty pages (TOC, section dividers)


def _should_drop_page(page_num: int, text: str, total_pages: int) -> bool:
    """Return True if this page should be excluded from indexing."""
    stripped = text.strip()
    if len(stripped) < _MIN_PAGE_CHARS:
        return True
    # First/last few pages often boilerplate
    if page_num <= 2:
        return True
    if page_num >= total_pages - 1:
        return True
    # Title-only or known non-content
    first_line = stripped.split("\n")[0].strip() if stripped else ""
    if _DROP_PAGE_PATTERNS.match(first_line) or len(stripped) < 200 and "ESI" not in stripped:
        if "level 1" not in stripped.lower() and "level 2" not in stripped.lower():
            return True
    return False


def _filter_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop pages not useful for ESI prediction."""
    n = len(pages)
    kept = []
    for num, text in pages:
        if _should_drop_page(num, text, n):
            continue
        kept.append((num, text))
    return kept


# ── Chunking ───────────────────────────────────────────────────────────
# Target ~500 tokens (~2000 chars), overlap ~100 tokens (~400 chars).
# text-embedding-005 supports up to 2048 tokens per input.
_CHUNK_CHARS = 2000
_OVERLAP_CHARS = 400


def _chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Produce chunks with page range and text. Each chunk has page_start, page_end, text."""
    combined = []
    for num, text in pages:
        combined.append((num, text.replace("\r\n", "\n")))
    full_text = "\n\n".join(t for _, t in combined)
    page_boundaries = []  # (char_start, page_num) for each page start
    pos = 0
    for num, text in combined:
        page_boundaries.append((pos, num))
        pos += len(text) + 2  # +2 for \n\n
    page_boundaries.append((len(full_text), pages[-1][0] + 1 if pages else 0))

    chunks = []
    start = 0
    chunk_id = 0
    while start < len(full_text):
        end = min(start + _CHUNK_CHARS, len(full_text))
        if end < len(full_text):
            # Break at sentence or paragraph
            break_at = full_text.rfind("\n\n", start, end + 1)
            if break_at > start + 500:
                end = break_at + 2
            else:
                break_at = full_text.rfind(". ", start, end + 1)
                if break_at > start + 300:
                    end = break_at + 2
        text_slice = full_text[start:end].strip()
        if not text_slice:
            start = end
            continue
        # Resolve page range for this chunk (page containing start/end)
        page_start = next((p for pos, p in reversed(page_boundaries) if pos <= start), 1)
        page_end = next((p for pos, p in reversed(page_boundaries) if pos <= end), page_start)
        chunks.append({
            "chunk_id": chunk_id,
            "text": text_slice,
            "page_start": page_start,
            "page_end": page_end,
        })
        chunk_id += 1
        start = end - _OVERLAP_CHARS
        if start >= len(full_text):
            break
    return chunks


# ── Embedding ──────────────────────────────────────────────────────────

def _embed_chunks(chunks: list[dict], embed_fn) -> np.ndarray:
    """Return (N, 768) float32 L2-normalised vectors."""
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _embed_one(text: str):
        return embed_fn(text)

    vectors = []
    for i, c in enumerate(chunks):
        vec = _embed_one(c["text"])
        vectors.append(vec.ravel())
        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} …")
    return np.array(vectors, dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from ESI Handbook PDF")
    default_pdf = Path(
        "/Users/ninotriandafilidis/Library/CloudStorage/"
        "GoogleDrive-ninot@stanford.edu/Shared drives/CS224n/Emergency_Severity_Index_Handbook.pdf"
    )
    default_out = Path(
        "/Users/ninotriandafilidis/Library/CloudStorage/"
        "GoogleDrive-ninot@stanford.edu/Shared drives/CS224n/esi_handbook_index"
    )
    parser.add_argument("--pdf", type=Path, default=default_pdf, help="Path to ESI Handbook PDF")
    parser.add_argument("--out", type=Path, default=default_out, help="Output directory for index and metadata")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)
    args.out.mkdir(parents=True, exist_ok=True)

    import src.config as config
    config.setup_clients(force_bq=False)
    from src.rag.retrieval import _embed_query

    print("Parsing PDF …")
    pages = _extract_text_by_page(args.pdf)
    print(f"  Total pages: {len(pages)}")
    kept = _filter_pages(pages)
    print(f"  Pages kept for indexing: {len(kept)}")
    chunks = _chunk_pages(kept)
    print(f"  Chunks: {len(chunks)}")

    print("\nEmbedding chunks (Vertex AI text-embedding-005) …")
    vectors = _embed_chunks(chunks, _embed_query)
    import faiss
    faiss.normalize_L2(vectors)

    print("\nBuilding FAISS index …")
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    index_path = args.out / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"  Wrote {index_path} ({index_path.stat().st_size / (1024**2):.1f} MB)")

    chunks_path = args.out / "chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"  Wrote {chunks_path}")

    manifest = {
        "source_pdf": str(args.pdf),
        "embedding_model": config.EMBEDDING_MODEL_ID,
        "embedding_dim": int(d),
        "num_chunks": len(chunks),
        "checksums": {
            "index.faiss": _sha256(index_path),
            "chunks.json": _sha256(chunks_path),
        },
    }
    manifest_path = args.out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path}")

    print("\n[DONE] Standalone ESI handbook index ready.")
    print(f"  To search: load {index_path} and {chunks_path}; embed query with same model; index.search(query_vec, k).")


if __name__ == "__main__":
    main()
