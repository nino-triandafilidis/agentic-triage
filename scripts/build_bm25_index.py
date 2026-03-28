"""Build a BM25S sparse index from the local PubMed SQLite article store.

Reads from ~/medllm/faiss/pmc_articles.db (same store used by FAISS),
strips front-matter using the full E04/E07 section-marker set, truncates
body text, and builds a BM25S index.  Articles without a detectable
section marker are excluded from the index entirely (see E04 skip results).

All operations are local — zero cloud cost.

Usage:
    .venv/bin/python scripts/build_bm25_index.py [--body-chars 2000] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config
from src.rag.text_cleaning import extract_body

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_corpus(
    db_path: Path, body_chars: int
) -> tuple[list[str], list[str], int]:
    """Stream articles from SQLite and return (corpus, pmc_ids, n_skipped).

    Articles without a detectable section marker are excluded.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.execute("SELECT COUNT(*) FROM articles")
    total = cursor.fetchone()[0]
    logger.info("Total articles in DB: %d", total)

    corpus: list[str] = []
    pmc_ids: list[str] = []
    n_skipped = 0
    batch_size = 50_000
    offset = 0

    while offset < total:
        rows = conn.execute(
            "SELECT pmc_id, article_text FROM articles LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break

        for pmc_id, text in rows:
            body = extract_body(text, body_chars)
            if body is None:
                n_skipped += 1
                continue
            corpus.append(body)
            pmc_ids.append(pmc_id)

        offset += batch_size
        logger.info(
            "Extracted %d / %d articles (%.1f%%), skipped %d (no marker)",
            len(corpus), total, 100.0 * (len(corpus) + n_skipped) / total,
            n_skipped,
        )

    conn.close()
    return corpus, pmc_ids, n_skipped


def early_abort_check(
    corpus: list[str],
    pmc_ids: list[str],
    sample_size: int,
    output_dir: Path,
    free_disk_bytes: int,
) -> bool:
    """Build a trial index on a sample and check extrapolated disk usage.

    Returns True if safe to proceed, False if the extrapolated index
    would exceed 80% of free disk.
    """
    import bm25s
    import Stemmer

    n_total = len(corpus)
    n_sample = min(sample_size, n_total)
    logger.info("Early-abort check: building trial index on %d articles...", n_sample)

    stemmer = Stemmer.Stemmer("english")
    sample_tokens = bm25s.tokenize(corpus[:n_sample], stemmer=stemmer, return_ids=True)

    retriever = bm25s.BM25()
    retriever.index(sample_tokens)

    trial_dir = output_dir / "_trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(trial_dir))

    trial_bytes = sum(f.stat().st_size for f in trial_dir.rglob("*") if f.is_file())
    shutil.rmtree(trial_dir)

    extrap_bytes = int(trial_bytes * (n_total / n_sample))
    extrap_gb = extrap_bytes / (1024 ** 3)
    free_gb = free_disk_bytes / (1024 ** 3)
    threshold_gb = 0.8 * free_gb

    logger.info(
        "Trial index: %d articles → %.1f MB on disk",
        n_sample, trial_bytes / (1024 ** 2),
    )
    logger.info(
        "Extrapolated full index: %.1f GB (free disk: %.1f GB, threshold: %.1f GB)",
        extrap_gb, free_gb, threshold_gb,
    )

    if extrap_gb > threshold_gb:
        logger.error(
            "ABORTING: extrapolated index (%.1f GB) exceeds 80%% of free disk (%.1f GB). "
            "Consider reducing --body-chars or freeing disk space.",
            extrap_gb, free_gb,
        )
        return False

    return True


def get_free_disk_bytes(path: Path) -> int:
    """Return free bytes on the filesystem containing *path*."""
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def main():
    parser = argparse.ArgumentParser(description="Build BM25S index from local PubMed SQLite.")
    parser.add_argument("--body-chars", type=int, default=2000,
                        help="Max chars of body text to index per article (default: 2000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run early-abort check only, don't build full index")
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip the early-abort disk check")
    args = parser.parse_args()

    import bm25s
    import Stemmer

    db_path = Path(config.FAISS_LOCAL_DIR) / "pmc_articles.db"
    output_dir = Path(config.BM25_LOCAL_DIR)

    if not db_path.exists():
        logger.error("SQLite DB not found: %s", db_path)
        sys.exit(1)

    logger.info("Source DB: %s", db_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Body chars: %d", args.body_chars)

    # 1. Extract corpus (articles without a section marker are excluded)
    t0 = time.time()
    corpus, pmc_ids, n_skipped = load_corpus(db_path, args.body_chars)
    logger.info(
        "Extraction complete: %d articles kept, %d skipped (no marker) in %.1fs",
        len(corpus), n_skipped, time.time() - t0,
    )

    # 2. Early-abort check
    if not args.skip_check:
        output_dir.mkdir(parents=True, exist_ok=True)
        free_bytes = get_free_disk_bytes(output_dir)
        safe = early_abort_check(corpus, pmc_ids, 100_000, output_dir, free_bytes)
        if not safe:
            sys.exit(1)
        if args.dry_run:
            logger.info("Dry run complete. Exiting without building full index.")
            return

    # 3. Tokenize full corpus
    logger.info("Tokenizing %d articles with Snowball English stemmer...", len(corpus))
    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer, return_ids=True)
    logger.info("Tokenization complete in %.1fs", time.time() - t0)

    # 4. Build index
    logger.info("Building BM25S index...")
    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    logger.info("Index built in %.1fs", time.time() - t0)

    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving index to %s ...", output_dir)
    retriever.save(str(output_dir))

    pmc_ids_path = output_dir / "pmc_ids.json"
    with open(pmc_ids_path, "w") as f:
        json.dump(pmc_ids, f)

    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(
        "Done. Total output: %.1f GB (%d files)",
        total_bytes / (1024 ** 3),
        sum(1 for _ in output_dir.rglob("*") if _.is_file()),
    )


if __name__ == "__main__":
    main()
