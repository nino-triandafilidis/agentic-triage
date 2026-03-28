"""One-time BQ export: download embeddings + articles for local FAISS index.

Run with:
    .venv/bin/python scripts/export_faiss_data.py

Exports:
  - pmc_embeddings.npy  (~6.5 GB) — float32 vectors, shape (N, 768)
  - pmc_ids.parquet     (~50 MB)  — pmc_id + pmid for each row (same order)
  - pmc_articles.db     (current 8k sidecar is ~17 GB; size varies) —
                        SQLite database with article metadata

Uses chunked reads via FARM_FINGERPRINT to avoid holding the full dataset in
RAM at once.  Each step prints estimated cost and waits for confirmation.

Writes manifest.json with SHA-256 checksums and row counts.
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402

N_CHUNKS = 10  # split export into 10 FARM_FINGERPRINT buckets


def _confirm(prompt: str) -> None:
    answer = input(f"\n{prompt}\nProceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_normalised(vectors: np.ndarray, sample_size: int = 1000) -> bool:
    """Check whether vectors are already L2-normalised (norms ≈ 1.0)."""
    idx = np.random.default_rng(42).choice(len(vectors), min(sample_size, len(vectors)), replace=False)
    norms = np.linalg.norm(vectors[idx], axis=1)
    return bool(np.allclose(norms, 1.0, atol=1e-3))


# ── Step 1: Export embeddings ─────────────────────────────────────────

def export_embeddings(out_dir: Path) -> tuple[int, bool]:
    """Export embedding vectors and IDs from BQ in chunks.

    Returns (total_rows, is_normalised).
    """
    vectors_path = out_dir / "pmc_embeddings.npy"
    ids_path = out_dir / "pmc_ids.parquet"

    if vectors_path.exists() and ids_path.exists():
        print(f"[SKIP] {vectors_path.name} and {ids_path.name} already exist.")
        existing = np.load(vectors_path, mmap_mode="r")
        is_norm = _check_normalised(existing)
        ids_df = pd.read_parquet(ids_path)
        return len(ids_df), is_norm

    _confirm(
        "STEP 1 — Export pmc_embeddings (vectors + IDs)\n"
        f"  Scans embedding column of {config.PMC_EMBEDDINGS_TABLE}\n"
        f"  Each of {N_CHUNKS} chunk queries scans the full table (~13.4 GB each)\n"
        "  Estimated one-time BQ cost: ~$0.67  (10 × 13.4 GB × $5/TB)\n"
        f"  Output: {vectors_path} (~6.5 GB) + {ids_path} (~50 MB)"
    )

    # --- Pass 1: count total rows so we can pre-allocate the npy file ---
    print("  Counting rows per chunk …", end=" ", flush=True)
    chunk_counts = []
    for i in range(N_CHUNKS):
        sql = f"""
        SELECT COUNT(*) AS cnt
        FROM `{config.PMC_EMBEDDINGS_TABLE}`
        WHERE MOD(ABS(FARM_FINGERPRINT(pmc_id)), {N_CHUNKS}) = {i}
        """
        cnt = int(config.bq_client.query(sql).to_dataframe()["cnt"].iloc[0])
        chunk_counts.append(cnt)
    total_rows = sum(chunk_counts)
    print(f"{total_rows:,} total")

    # --- Pre-allocate memory-mapped npy file (avoids doubling RAM) ---
    # np.lib.format: write header, then mmap the data region
    dim = 768
    fp = np.lib.format.open_memmap(
        str(vectors_path), mode="w+", dtype=np.float32, shape=(total_rows, dim)
    )

    all_ids = []
    write_offset = 0

    for i in range(N_CHUNKS):
        print(f"  Chunk {i + 1}/{N_CHUNKS} …", end=" ", flush=True)
        sql = f"""
        SELECT
            pmc_id,
            pmid,
            ml_generate_embedding_result
        FROM `{config.PMC_EMBEDDINGS_TABLE}`
        WHERE MOD(ABS(FARM_FINGERPRINT(pmc_id)), {N_CHUNKS}) = {i}
        """
        df = config.bq_client.query(sql).to_dataframe()
        n = len(df)
        print(f"{n:,} rows")

        # Extract float32 vectors and write directly into mmap
        vecs = np.array(df["ml_generate_embedding_result"].tolist(), dtype=np.float32)
        fp[write_offset:write_offset + n] = vecs
        write_offset += n

        all_ids.append(df[["pmc_id", "pmid"]])
        del df, vecs  # free chunk memory

    fp.flush()
    print(f"\n  Total rows exported: {total_rows:,}")

    is_norm = _check_normalised(fp)
    print(f"  Vectors L2-normalised: {is_norm}")
    del fp  # close mmap

    print(f"  Writing {ids_path.name} …")
    ids_df = pd.concat(all_ids, ignore_index=True)
    del all_ids
    ids_df.to_parquet(ids_path, index=False)

    return total_rows, is_norm


# ── Step 2: Export articles ───────────────────────────────────────────

def export_articles(out_dir: Path) -> int:
    """Export article metadata from BQ in chunks to a SQLite database.

    Returns total_rows.
    """
    articles_path = out_dir / "pmc_articles.db"

    if articles_path.exists():
        print(f"[SKIP] {articles_path.name} already exists.")
        conn = sqlite3.connect(articles_path)
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        return total

    _confirm(
        "STEP 2 — Export pmc_articles (metadata)\n"
        f"  Scans {config.PMC_ARTICLES_TABLE}\n"
        f"  Each of {N_CHUNKS} chunk queries scans the full table (~4.5 GB each)\n"
        "  Estimated one-time BQ cost: ~$0.22  (10 × 4.5 GB × $5/TB)\n"
        f"  Output: {articles_path} (~2-3 GB)"
    )

    conn = sqlite3.connect(articles_path)
    conn.execute("""
        CREATE TABLE articles (
            pmc_id TEXT PRIMARY KEY,
            pmid TEXT,
            article_text TEXT,
            article_citation TEXT
        )
    """)
    total_rows = 0

    for i in range(N_CHUNKS):
        print(f"  Chunk {i + 1}/{N_CHUNKS} …", end=" ", flush=True)
        sql = f"""
        SELECT pmc_id, pmid, article_text, article_citation
        FROM `{config.PMC_ARTICLES_TABLE}`
        WHERE MOD(ABS(FARM_FINGERPRINT(pmc_id)), {N_CHUNKS}) = {i}
        """
        df = config.bq_client.query(sql).to_dataframe()
        n = len(df)
        total_rows += n
        print(f"{n:,} rows")

        conn.executemany(
            "INSERT INTO articles (pmc_id, pmid, article_text, article_citation) VALUES (?, ?, ?, ?)",
            df[["pmc_id", "pmid", "article_text", "article_citation"]].values.tolist(),
        )
        conn.commit()
        del df

    conn.close()

    print(f"\n  Total rows exported: {total_rows:,}")
    return total_rows


# ── Manifest ──────────────────────────────────────────────────────────

def write_manifest(out_dir: Path, emb_rows: int, art_rows: int, is_normalised: bool) -> None:
    """Write manifest.json with checksums and metadata."""
    manifest = {
        "embedding_rows": emb_rows,
        "article_rows": art_rows,
        "vectors_normalised": is_normalised,
        "embedding_dim": 768,
        "checksums": {},
    }
    for fname in ["pmc_embeddings.npy", "pmc_ids.parquet", "pmc_articles.db"]:
        path = out_dir / fname
        if path.exists():
            print(f"  Computing SHA-256 for {fname} …")
            manifest["checksums"][fname] = _sha256(path)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[OK] Manifest written to {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path(config.FAISS_STORE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAISS Data Export")
    print(f"Project  : {config.PROJECT_ID}")
    print(f"Output   : {out_dir}")
    print("=" * 60)
    print(
        "\nTotal estimated one-time cost: ~$0.90"
        "\n  (each chunk query re-scans the full table; 10 chunks × ~18 GB)"
        "\nThis exports BQ data locally — zero cost for future searches."
    )

    config.setup_clients(bq_guardrail=False, force_bq=True)

    emb_rows, is_norm = export_embeddings(out_dir)
    art_rows = export_articles(out_dir)
    write_manifest(out_dir, emb_rows, art_rows, is_norm)

    print("\n[DONE] Export complete.")
    print("Next: run scripts/build_faiss_index.py to build the FAISS index.")


if __name__ == "__main__":
    main()
