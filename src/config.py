"""Centralised GCP configuration and one-time project setup."""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.auth import default

# Load .env / .env.local (secrets / machine-specific overrides) if present.
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env", override=False)
load_dotenv(_project_root / ".env.local", override=False)
from google.cloud import bigquery
from google import genai

# ── Auto-detect credentials & project ──────────────────────────────────
if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user()


def _resolve_project() -> tuple:
    """Return (credentials, project_id) with ADC fallback."""
    credentials, project = default()
    if project:
        return credentials, project

    adc_path = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    if os.path.exists(adc_path):
        with open(adc_path) as f:
            adc = json.load(f)
        quota = adc.get("quota_project_id")
        if quota:
            return credentials, quota

    raise RuntimeError(
        "No GCP project found. Set GOOGLE_CLOUD_PROJECT or run:\n"
        "  gcloud config set project YOUR_PROJECT_ID"
    )


# Lazy-resolved credentials — _resolve_project() is NOT called at import time
# so that tests can import this module without GCP ADC.  Access CREDENTIALS,
# PROJECT_ID, EMBEDDING_MODEL, PMC_EMBEDDINGS_TABLE, or PMC_ARTICLES_TABLE
# to trigger resolution on first use.
_credentials = None
_project_id = None
_resolved = False


def _ensure_resolved() -> None:
    global _credentials, _project_id, _resolved
    if not _resolved:
        _credentials, _project_id = _resolve_project()
        _resolved = True


# ── Constants ──────────────────────────────────────────────────────────
BQ_LOCATION = "US"
LOCATION = "global"
MODEL_ID = "gemini-3-flash-preview"

PUBMED_DATASET = "bigquery-public-data.pmc_open_access_commercial"
PUBMED_TABLE = f"{PUBMED_DATASET}.articles"
USER_DATASET = "pubmed"

# ── Lazy attributes (resolved on first access via __getattr__) ────────
_LAZY_ATTRS = {
    "CREDENTIALS", "PROJECT_ID",
    "EMBEDDING_MODEL", "PMC_EMBEDDINGS_TABLE", "PMC_ARTICLES_TABLE",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        _ensure_resolved()
        if name == "CREDENTIALS":
            return _credentials
        if name == "PROJECT_ID":
            return _project_id
        if name == "EMBEDDING_MODEL":
            return f"{_project_id}.{USER_DATASET}.textembed"
        if name == "PMC_EMBEDDINGS_TABLE":
            return f"{_project_id}.{USER_DATASET}.pmc_embeddings"
        if name == "PMC_ARTICLES_TABLE":
            return f"{_project_id}.{USER_DATASET}.pmc_articles"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ── Retrieval config ────────────────────────────────────────────────────
RETRIEVAL_CONTEXT_CHARS = 8000  # chars per article passed to the prompt; tune freely

# ── Retrieval backend ──────────────────────────────────────────────────
# "faiss"  (default): local FAISS dense index — one Vertex embedding call per query
# "bm25":             local BM25S sparse index — zero API cost per query
# "hybrid":           FAISS + BM25 fused via Reciprocal Rank Fusion
# "bq":               BigQuery VECTOR_SEARCH — ~$0.10 per query (avoid)
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "faiss")

# ── FAISS local retrieval paths ────────────────────────────────────────
FAISS_STORE_DIR = os.environ.get(
    "FAISS_STORE_DIR",
    os.path.join(os.path.expanduser("~"), "medllm", "faiss"),
)
FAISS_LOCAL_DIR = os.environ.get(
    "FAISS_LOCAL_DIR",
    os.path.join(os.path.expanduser("~"), "medllm", "faiss"),
)
FAISS_NPROBE = int(os.environ.get("FAISS_NPROBE", "64"))
EMBEDDING_MODEL_ID = "text-embedding-005"

# ── BM25 local retrieval paths ────────────────────────────────────────
BM25_LOCAL_DIR = os.environ.get(
    "BM25_LOCAL_DIR",
    os.path.join(os.path.expanduser("~"), "medllm", "bm25s"),
)

# ── Hybrid (RRF) parameters ──────────────────────────────────────────
RRF_K = int(os.environ.get("RRF_K", "60"))
RRF_W_SPARSE = float(os.environ.get("RRF_W_SPARSE", "1.0"))
RRF_W_DENSE = float(os.environ.get("RRF_W_DENSE", "1.0"))

# ── BigQuery cost guardrail ─────────────────────────────────────────────
# Queries exceeding this limit are rejected by BQ before execution (no charge).
# Note: BQ uses a pre-execution *estimate* which can exceed actual bytes billed
# (e.g. IVF VECTOR_SEARCH estimates ~25 GB but actually bills ~12.57 GB).
# 30 GB covers the largest pre-execution estimate seen in this project while
# still blocking full article_text scans (~112 GB) and runaway analytics queries.
# At $6.25/TB, the hard cap per query is ~$0.19.
BQ_MAX_BYTES_BILLED = 30 * 1024 ** 3  # 30 GB

# ── Shared clients (initialised lazily via setup()) ────────────────────
genai_client: genai.Client | None = None
bq_client: bigquery.Client | None = None

REQUIRED_APIS = [
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "bigqueryconnection.googleapis.com",
]


def setup_clients(bq_guardrail: bool = True, *, force_bq: bool = False) -> None:
    """Initialise BQ and genai clients without calling gcloud.

    Use this instead of setup() when APIs are already enabled.

    Args:
        bq_guardrail: If True (default), cap each BQ query at BQ_MAX_BYTES_BILLED
            (~$0.12).  Pass False only in provisioning scripts that legitimately
            scan >20 GB (e.g. create_vector_store.py).
        force_bq: If True, always initialise the BQ client even when
            RETRIEVAL_BACKEND is "faiss".  Needed by export/validation scripts.
    """
    global genai_client, bq_client
    _ensure_resolved()

    genai_client = genai.Client(
        vertexai=True,
        project=_project_id,
        location=LOCATION,
        credentials=_credentials,
    )

    if RETRIEVAL_BACKEND == "bq" or force_bq:
        bq_client = bigquery.Client(
            project=_project_id,
            credentials=_credentials,
            default_query_job_config=bigquery.QueryJobConfig(
                **({} if not bq_guardrail else {"maximum_bytes_billed": BQ_MAX_BYTES_BILLED})
            ),
        )


def setup() -> None:
    """Enable APIs, create BQ dataset & embedding model, initialise clients.

    Safe to call repeatedly — every step is idempotent.
    """
    global genai_client, bq_client
    _ensure_resolved()

    # 1. Enable APIs
    for api in REQUIRED_APIS:
        subprocess.run(
            ["gcloud", "services", "enable", api, f"--project={_project_id}"],
            check=True,
            capture_output=True,
        )

    # 2. Initialise clients (setup always needs BQ for dataset/model creation)
    setup_clients(force_bq=True)

    # 3. Create BQ dataset if needed
    dataset_ref = bigquery.DatasetReference(_project_id, USER_DATASET)
    bq_client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)

    # 4. Create embedding model if needed
    embed_ddl = f"""
    CREATE MODEL IF NOT EXISTS `{_project_id}.{USER_DATASET}.textembed`
    REMOTE WITH CONNECTION DEFAULT
    OPTIONS(endpoint='text-embedding-005');
    """
    bq_client.query(embed_ddl).result()

    print(f"Setup complete for project {_project_id}")
