"""PubMed article retrieval — multiple backends.

Backends (selected by config.RETRIEVAL_BACKEND):
  "faiss"  — Dense: memory-mapped IVF index + Vertex AI embedding
  "bm25"   — Sparse: local BM25S index, zero API cost
  "hybrid" — FAISS + BM25 fused via Reciprocal Rank Fusion
  "bq"     — BigQuery VECTOR_SEARCH (~$0.10/call, avoid)

All backends return a DataFrame with columns:
  pmc_id, pmid, rank, score, score_type, article_text, article_citation,
  context_text (cleaned body text when available, else None).

Score semantics vary by backend — do NOT compare scores across backends:
  faiss/bq: score = cosine distance (lower = more similar)
  bm25:     score = BM25 score (higher = more relevant)
  hybrid:   score = RRF score (higher = more relevant)

Setup — first-time FAISS file download:
  Files are downloaded from Google Drive via the Drive API (resumable, handles
  large files reliably).  Requires a one-time auth that includes Drive scope:

      gcloud auth application-default login \\
          --scopes=openid,https://www.googleapis.com/auth/userinfo.email,\\
    https://www.googleapis.com/auth/cloud-platform,\\
    https://www.googleapis.com/auth/drive.readonly

  If Drive auth is unavailable, falls back to FUSE mount copy, then suggests
  BQ re-export as a last resort.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import src.config as config
from src.rag.text_cleaning import prepare_article_excerpt
logger = logging.getLogger(__name__)

# Google Drive file IDs for FAISS artefacts on the CS224n shared drive.
# Retrieved via xattr com.google.drivefs.item-id#S on the FUSE-mounted files.
GDRIVE_FILE_IDS = {
    "index.faiss": "1iWIK_r9181huSoH32dtEzoGbKN4GcfY8",
    "pmc_ids.parquet": "1oByUSKom4-GV7Yd9Ca7Dks2xH07xgcD4",
    "pmc_articles.db": "1efqNE9Y4G5HqLhvOPo1ohAXAcuC5_GhD",
}

# ── Shared output columns ─────────────────────────────────────────────
RESULT_COLS = [
    "pmc_id", "pmid", "rank", "score", "score_type",
    "article_text", "article_citation", "context_text",
]

# ── Lazy-loaded FAISS singletons ──────────────────────────────────────
_faiss_index = None
_pmc_ids_df: pd.DataFrame | None = None
_articles_db: sqlite3.Connection | None = None

# ── Lazy-loaded BM25S singletons ─────────────────────────────────────
_bm25_retriever = None
_bm25_pmc_ids: list[str] | None = None
_bm25_stemmer = None


# ── FAISS helpers ─────────────────────────────────────────────────────

def _verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Return True if SHA-256 of *file_path* matches *expected_sha256*."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def _download_from_gdrive(file_id: str, dest: Path) -> None:
    """Download a file from Google Drive using the Drive API v3.

    Uses resumable media download (100 MB chunks) so that large files
    (17+ GB) transfer reliably without the ETIMEDOUT failures seen with
    the CloudStorage FUSE mount.

    Automatically retries on transient errors with exponential backoff.
    Writes to a .tmp file and renames atomically on success.
    """
    from google.auth import default as auth_default
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds, _ = auth_default(scopes=SCOPES)
    creds.refresh(Request())

    service = build("drive", "v3", credentials=creds)
    request = service.files().get_media(fileId=file_id)

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    tmp_dest.unlink(missing_ok=True)  # clean up any prior partial download

    chunk_size = 100 * 1024 * 1024  # 100 MB chunks
    max_retries = 20

    with open(tmp_dest, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=chunk_size)
        done = False
        retries = 0
        while not done:
            try:
                status, done = downloader.next_chunk()
                retries = 0  # reset on success
                if status:
                    pct = int(status.progress() * 100)
                    mb = f.tell() >> 20
                    logger.info("  %s: %d%% (%d MB)", dest.name, pct, mb)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    tmp_dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Google Drive download failed after {max_retries} retries: {e}"
                    ) from e
                wait = min(2 ** retries, 120)
                logger.warning(
                    "  Download error (retry %d/%d): %s — waiting %ds",
                    retries, max_retries, e, wait,
                )
                time.sleep(wait)

    os.rename(tmp_dest, dest)
    logger.info("Downloaded %s (%d MB)", dest.name, dest.stat().st_size >> 20)


def _copy_from_fuse(src_file: Path, dest: Path) -> None:
    """Copy a file from the FUSE-mounted shared drive with chunked retry."""
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    max_retries = 200
    for attempt in range(1, max_retries + 1):
        copied = tmp_dest.stat().st_size if tmp_dest.exists() else 0
        try:
            with open(src_file, "rb") as fsrc, open(tmp_dest, "ab") as fdst:
                fsrc.seek(copied)
                while True:
                    buf = fsrc.read(1 << 22)  # 4 MB chunks
                    if not buf:
                        break
                    fdst.write(buf)
            break  # success
        except (TimeoutError, OSError) as e:
            if attempt == max_retries:
                raise
            wait = min(30 * attempt, 120)
            logger.warning(
                "FUSE copy interrupted at %d MB (attempt %d/%d): %s — waiting %ds",
                copied >> 20, attempt, max_retries, e, wait,
            )
            time.sleep(wait)

    os.rename(tmp_dest, dest)
    logger.info("Copied %s (%d MB)", dest.name, dest.stat().st_size >> 20)


def _ensure_local_copy() -> Path:
    """Ensure FAISS artefacts are present locally, downloading if needed.

    Download strategy (tries in order):
      1. Google Drive API — resumable 100 MB chunks, most reliable
      2. FUSE mount copy — fast when it works, fails on large files
      3. Error with instructions to re-export from BQ

    Each file is written to a .tmp suffix first, then os.rename()'d to the
    final name (atomic on POSIX).  If any download is interrupted, no
    corrupted partial files are left behind.

    Returns the local directory path.
    """
    store = Path(config.FAISS_STORE_DIR)
    local = Path(config.FAISS_LOCAL_DIR)

    # Load manifest — try local copy first, then shared drive
    manifest = None
    for mpath in [local / "manifest.json", store / "manifest.json"]:
        if mpath.exists():
            with open(mpath) as f:
                manifest = json.load(f)
            break

    if manifest is None:
        raise FileNotFoundError(
            f"manifest.json not found in {local} or {store}. "
            "Run scripts/export_faiss_data.py and scripts/build_faiss_index.py first."
        )

    local.mkdir(parents=True, exist_ok=True)

    # Copy manifest locally if only on shared drive
    local_manifest = local / "manifest.json"
    if not local_manifest.exists():
        with open(local_manifest, "w") as f:
            json.dump(manifest, f, indent=2)

    files_to_copy = ["index.faiss", "pmc_ids.parquet", "pmc_articles.db"]
    for fname in files_to_copy:
        dest = local / fname
        expected = manifest["checksums"].get(fname)

        # Skip if already present and valid
        if dest.exists() and expected and _verify_checksum(dest, expected):
            logger.debug("Local copy OK: %s", fname)
            continue

        logger.info("Need to download %s to %s", fname, local)

        # --- Strategy 1: Google Drive API (resumable, reliable) ---
        gdrive_id = GDRIVE_FILE_IDS.get(fname)
        if gdrive_id:
            try:
                logger.info("Trying Google Drive API download for %s …", fname)
                _download_from_gdrive(gdrive_id, dest)
                if expected and not _verify_checksum(dest, expected):
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(f"Checksum mismatch after Drive download of {fname}")
                continue  # success — next file
            except Exception as e:
                logger.warning("Drive API download failed: %s", e)
                dest.unlink(missing_ok=True)
                dest.with_suffix(dest.suffix + ".tmp").unlink(missing_ok=True)

        # --- Strategy 2: FUSE mount copy ---
        src_file = store / fname
        if src_file.exists():
            try:
                logger.info("Trying FUSE mount copy for %s …", fname)
                _copy_from_fuse(src_file, dest)
                if expected and not _verify_checksum(dest, expected):
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(f"Checksum mismatch after FUSE copy of {fname}")
                continue  # success — next file
            except Exception as e:
                logger.warning("FUSE copy failed: %s", e)
                dest.unlink(missing_ok=True)
                dest.with_suffix(dest.suffix + ".tmp").unlink(missing_ok=True)

        # --- All strategies failed ---
        raise RuntimeError(
            f"Could not obtain {fname}. Options to fix:\n"
            "  1. Set up Drive API auth (one-time):\n"
            "       gcloud auth application-default login \\\n"
            "         --scopes=openid,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/drive.readonly\n"
            "  2. Download manually from Google Drive web UI to:\n"
            f"       {dest}\n"
            "  3. Re-export from BigQuery (~$0.22):\n"
            "       .venv/bin/python scripts/export_faiss_data.py"
        )

    return local


def _load_faiss_index():
    """Load (or return cached) FAISS index and sidecar data.

    Articles are stored in a SQLite database with a B-tree index on pmc_id.
    Only the pages touched by each query are loaded (~40 KB per 10-row lookup),
    so steady-state RAM is essentially zero for the articles component.
    """
    global _faiss_index, _pmc_ids_df, _articles_db

    if _faiss_index is not None:
        return _faiss_index, _pmc_ids_df, _articles_db

    import faiss

    local = _ensure_local_copy()

    logger.info("Memory-mapping FAISS index from %s …", local)
    _faiss_index = faiss.read_index(
        str(local / "index.faiss"), faiss.IO_FLAG_MMAP
    )
    _faiss_index.nprobe = config.FAISS_NPROBE

    _pmc_ids_df = pd.read_parquet(local / "pmc_ids.parquet")

    # Open SQLite in read-only mode (no WAL, no locking overhead)
    db_path = local / "pmc_articles.db"
    _articles_db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    logger.info(
        "FAISS index loaded: %d vectors, nprobe=%d",
        _faiss_index.ntotal,
        config.FAISS_NPROBE,
    )
    return _faiss_index, _pmc_ids_df, _articles_db


def _embed_query(query_text: str) -> np.ndarray:
    """Embed a single query string, return L2-normalised float32 vector."""
    from src.llm import embed as llm_embed

    response = llm_embed(query_text, model_id=config.EMBEDDING_MODEL_ID)
    vec = np.array(response.values, dtype=np.float32).reshape(1, -1)
    # L2-normalise so inner-product == cosine similarity
    faiss_mod = __import__("faiss")
    faiss_mod.normalize_L2(vec)
    return vec


# ── Backend implementations ───────────────────────────────────────────

def _search_faiss(query_text: str, top_k: int = 10) -> pd.DataFrame:
    """Search the local FAISS index and return results with cosine distance scores.

    The IVF index with nprobe>1 can return the same vector from multiple
    probed cells.  We over-fetch by a margin, deduplicate by pmc_id (keeping
    the best similarity), and return exactly top_k unique articles.
    """
    _DEDUP_MARGIN = 10  # extra results to compensate for duplicates

    index, pmc_ids_df, articles_db = _load_faiss_index()
    query_vec = _embed_query(query_text)

    fetch_k = top_k + _DEDUP_MARGIN
    similarities, indices = index.search(query_vec, fetch_k)

    # Deduplicate: keep best (highest similarity) per pmc_id
    best: dict[str, tuple[str, float]] = {}  # pmc_id -> (pmc_id, cosine_dist)
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        pmc_id = pmc_ids_df.iloc[idx]["pmc_id"]
        cosine_dist = 1.0 - float(sim)
        if pmc_id not in best or cosine_dist < best[pmc_id][1]:
            best[pmc_id] = (pmc_id, cosine_dist)

    # Sort by distance (ascending) and take top_k
    hits = sorted(best.values(), key=lambda x: x[1])[:top_k]

    if not hits:
        return pd.DataFrame(columns=RESULT_COLS)

    art_by_id = _lookup_articles(articles_db, [h[0] for h in hits])

    rows = []
    for rank_idx, (pmc_id, cosine_dist) in enumerate(hits):
        art = art_by_id.get(pmc_id)
        article_text = art[2] if art else None
        rows.append({
            "pmc_id": pmc_id,
            "pmid": art[1] if art else None,
            "rank": rank_idx,
            "score": cosine_dist,
            "score_type": "cosine_distance",
            "article_text": article_text,
            "article_citation": art[3] if art else None,
            "context_text": (
                prepare_article_excerpt(article_text, config.RETRIEVAL_CONTEXT_CHARS)
                if article_text
                else None
            ),
        })

    return pd.DataFrame(rows, columns=RESULT_COLS)


def _lookup_articles(db: sqlite3.Connection, pmc_ids: list[str]) -> dict:
    """Batch-lookup article metadata from SQLite. Returns {pmc_id: (pmc_id, pmid, text, citation)}."""
    placeholders = ",".join("?" * len(pmc_ids))
    cursor = db.execute(
        f"SELECT pmc_id, pmid, article_text, article_citation "
        f"FROM articles WHERE pmc_id IN ({placeholders})",
        pmc_ids,
    )
    return {row[0]: row for row in cursor}


# ── BM25 helpers ─────────────────────────────────────────────────────

def _load_bm25_index():
    """Load (or return cached) BM25S retriever, pmc_ids, and stemmer."""
    global _bm25_retriever, _bm25_pmc_ids, _bm25_stemmer

    if _bm25_retriever is not None:
        return _bm25_retriever, _bm25_pmc_ids, _bm25_stemmer

    import bm25s
    import Stemmer

    local = Path(config.BM25_LOCAL_DIR)

    if not (local / "params.index.json").exists():
        raise FileNotFoundError(
            f"BM25S index not found in {local}. "
            "Run: .venv/bin/python scripts/build_bm25_index.py"
        )

    logger.info("Loading BM25S index from %s (mmap=True) …", local)
    _bm25_retriever = bm25s.BM25.load(str(local), mmap=True)

    with open(local / "pmc_ids.json") as f:
        _bm25_pmc_ids = json.load(f)

    _bm25_stemmer = Stemmer.Stemmer("english")

    logger.info("BM25S index loaded: %d documents", len(_bm25_pmc_ids))
    return _bm25_retriever, _bm25_pmc_ids, _bm25_stemmer


def _search_bm25(query_text: str, top_k: int = 10) -> pd.DataFrame:
    """Search the local BM25S index. Zero API cost.

    context_text is derived on demand from article_text via the shared
    section-extraction helper (same logic used at index build time).
    """
    import bm25s

    retriever, pmc_ids, stemmer = _load_bm25_index()

    query_tokens = bm25s.tokenize(query_text, stemmer=stemmer)
    doc_indices, scores = retriever.retrieve(query_tokens, k=top_k)

    _, articles_db = _ensure_articles_db()

    hit_pmc_ids = []
    hit_scores = []
    for rank_idx in range(doc_indices.shape[1]):
        doc_idx = int(doc_indices[0, rank_idx])
        hit_pmc_ids.append(pmc_ids[doc_idx])
        hit_scores.append(float(scores[0, rank_idx]))

    art_by_id = _lookup_articles(articles_db, hit_pmc_ids)

    rows = []
    for rank_idx, (pmc_id, bm25_score) in enumerate(zip(hit_pmc_ids, hit_scores)):
        art = art_by_id.get(pmc_id)
        article_text = art[2] if art else None
        ctx = (
            prepare_article_excerpt(article_text, config.RETRIEVAL_CONTEXT_CHARS)
            if article_text
            else None
        )
        rows.append({
            "pmc_id": pmc_id,
            "pmid": art[1] if art else None,
            "rank": rank_idx,
            "score": bm25_score,
            "score_type": "bm25",
            "article_text": article_text,
            "article_citation": art[3] if art else None,
            "context_text": ctx,
        })

    return pd.DataFrame(rows, columns=RESULT_COLS)


def _ensure_articles_db() -> tuple[Path, sqlite3.Connection]:
    """Return (local_dir, articles_db), loading FAISS sidecar DB if needed."""
    global _articles_db
    if _articles_db is not None:
        return Path(config.FAISS_LOCAL_DIR), _articles_db

    local = _ensure_local_copy()
    db_path = local / "pmc_articles.db"
    _articles_db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    return local, _articles_db


def _search_hybrid(query_text: str, top_k: int = 10) -> pd.DataFrame:
    """Fuse FAISS (dense) and BM25 (sparse) via Reciprocal Rank Fusion."""
    over_k = top_k * 2

    dense_df = _search_faiss(query_text, over_k)
    sparse_df = _search_bm25(query_text, over_k)

    k = config.RRF_K
    w_dense = config.RRF_W_DENSE
    w_sparse = config.RRF_W_SPARSE

    rrf_scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for _, row in dense_df.iterrows():
        pid = row["pmc_id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_dense / (k + row["rank"])
        if pid not in meta:
            meta[pid] = row.to_dict()

    for _, row in sparse_df.iterrows():
        pid = row["pmc_id"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + w_sparse / (k + row["rank"])
        if pid not in meta:
            meta[pid] = row.to_dict()
        elif row.get("context_text") is not None:
            meta[pid]["context_text"] = row["context_text"]

    sorted_pids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    rows = []
    for rank_idx, pid in enumerate(sorted_pids):
        m = meta[pid]
        ctx = m.get("context_text")
        if ctx is None and m.get("article_text"):
            ctx = prepare_article_excerpt(
                m["article_text"],
                config.RETRIEVAL_CONTEXT_CHARS,
            )
        rows.append({
            "pmc_id": pid,
            "pmid": m.get("pmid"),
            "rank": rank_idx,
            "score": rrf_scores[pid],
            "score_type": "rrf",
            "article_text": m.get("article_text"),
            "article_citation": m.get("article_citation"),
            "context_text": ctx,
        })

    return pd.DataFrame(rows, columns=RESULT_COLS)


def _search_bq(query_text: str, top_k: int = 10) -> pd.DataFrame:
    """Search via BigQuery VECTOR_SEARCH (original implementation)."""
    from google.cloud import bigquery

    if config.bq_client is None:
        raise RuntimeError("Call config.setup_clients() before using retrieval.")

    sql = f"""
    SELECT
        base.pmc_id,
        base.pmid,
        distance,
        docs.article_text,
        docs.article_citation
    FROM VECTOR_SEARCH(
        TABLE `{config.PMC_EMBEDDINGS_TABLE}`,
        'ml_generate_embedding_result',
        (SELECT ml_generate_embedding_result
         FROM ML.GENERATE_EMBEDDING(
             MODEL `{config.EMBEDDING_MODEL}`,
             (SELECT @query AS content)
         )),
        top_k => {top_k},
        options => '{{"fraction_lists_to_search": 0.1}}'
    )
    JOIN `{config.PMC_ARTICLES_TABLE}` docs ON base.pmc_id = docs.pmc_id
    ORDER BY distance
    LIMIT {top_k}
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("query", "STRING", query_text)
        ]
    )
    return config.bq_client.query(sql, job_config=job_config).to_dataframe()


# ── Public API ────────────────────────────────────────────────────────

_BACKENDS = {
    "faiss": _search_faiss,
    "bm25": _search_bm25,
    "hybrid": _search_hybrid,
    "bq": _search_bq,
}


def search_pubmed_articles(query_text: str, top_k: int = 10) -> pd.DataFrame:
    """Return the top-k PubMed articles for *query_text*.

    Returns columns: pmc_id, pmid, rank, score, score_type,
                     article_text, article_citation, context_text.

    Score semantics depend on the backend (see module docstring).
    Backend is selected by config.RETRIEVAL_BACKEND.
    """
    backend = config.RETRIEVAL_BACKEND
    fn = _BACKENDS.get(backend)
    if fn is None:
        raise ValueError(
            f"Unknown RETRIEVAL_BACKEND: {backend!r}. "
            f"Choose from: {sorted(_BACKENDS)}."
        )
    return fn(query_text, top_k)


# ── Tool-use interface ────────────────────────────────────────────────

def search_pmc_articles(
    query: str,
    top_k: int = 5,
    *,
    context_chars: int | None = None,
) -> str:
    """Search PubMed Central articles by natural language query.

    Returns formatted article summaries suitable for insertion into
    tool-result messages (title, PMID, cleaned excerpt text).
    """
    articles = search_pubmed_articles(query, top_k=top_k)
    excerpt_chars = context_chars or config.RETRIEVAL_CONTEXT_CHARS

    if articles.empty:
        return "No relevant PubMed articles found for this query."

    parts = [f"Found {len(articles)} relevant article(s):\n"]
    for _, row in articles.iterrows():
        pmid = row.get("pmid", "N/A")
        citation = str(row.get("article_citation", ""))[:200]
        text = prepare_article_excerpt(
            row.get("article_text"),
            excerpt_chars,
            context_text=row.get("context_text"),
        )
        parts.append(
            f"--- PMID {pmid} ---\n"
            f"Citation: {citation}\n"
            f"Excerpt: {text}"
        )
    return "\n\n".join(parts)


PMC_SEARCH_TOOL_SCHEMA: dict = {
    "name": "search_pmc_articles",
    "description": (
        "Search a local index of ~2.3 million PubMed Central open-access "
        "articles for medical literature relevant to a clinical query. "
        "Returns article excerpts with PMIDs and citations. Use this tool "
        "to find evidence-based literature that can inform triage decisions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A natural language search query describing the "
                    "clinical scenario or medical topic to search for. "
                    "Example: 'acute chest pain troponin elevation "
                    "emergency department triage'"
                ),
            },
            "top_k": {
                "type": "integer",
                "description": (
                    "Number of articles to return (default 5, max 10)."
                ),
            },
        },
        "required": ["query"],
    },
}
