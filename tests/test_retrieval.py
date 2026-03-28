"""Tests for src/rag/retrieval.py.

Unit tests use mocks and never touch BigQuery or FAISS files.
Integration tests hit real backends and are skipped unless INTEGRATION=1 is set.

Run unit tests:
    .venv/bin/python -m pytest tests/test_retrieval.py -v

Run integration tests (costs ~$0.01 per BQ call with the IVF index):
    INTEGRATION=1 .venv/bin/python -m pytest tests/test_retrieval.py -v -m integration
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.config as config  # noqa: E402

INTEGRATION = os.environ.get("INTEGRATION", "0") == "1"
integration = pytest.mark.skipif(not INTEGRATION, reason="set INTEGRATION=1 to run")

EXPECTED_COLUMNS_BQ = {"pmc_id", "pmid", "distance", "article_text", "article_citation"}
EXPECTED_COLUMNS_UNIFIED = {
    "pmc_id", "pmid", "rank", "score", "score_type",
    "article_text", "article_citation", "context_text",
}

# ── Fixtures ────────────────────────────────────────────────────────────

FAKE_RESULT = pd.DataFrame({
    "pmc_id":           ["PMC1111111", "PMC2222222", "PMC3333333"],
    "pmid":             [11111111,     22222222,     33333333],
    "distance":         [0.20,         0.30,         0.40],
    "article_text":     ["Article A text", "Article B text", "Article C text"],
    "article_citation": ["J Med. 2022; 1:1", "J Med. 2023; 2:2", "J Med. 2024; 3:3"],
})


@pytest.fixture(autouse=True)
def inject_mock_bq_client():
    """Inject a mock BQ client into config for all unit tests."""
    mock_job = MagicMock()
    mock_job.to_dataframe.return_value = FAKE_RESULT

    mock_client = MagicMock()
    mock_client.query.return_value = mock_job

    with patch.object(config, "bq_client", mock_client):
        yield mock_client


# ── BQ backend unit tests ────────────────────────────────────────────

class TestBQBackend:
    """Tests for the BigQuery VECTOR_SEARCH backend."""

    @pytest.fixture(autouse=True)
    def set_bq_backend(self):
        with patch.object(config, "RETRIEVAL_BACKEND", "bq"):
            yield

    def test_returns_dataframe(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert EXPECTED_COLUMNS_BQ.issubset(set(result.columns)), \
            f"Missing columns: {EXPECTED_COLUMNS_BQ - set(result.columns)}"

    def test_raises_if_no_client(self):
        with patch.object(config, "bq_client", None):
            from src.rag.retrieval import search_pubmed_articles
            with pytest.raises(RuntimeError, match="setup_clients"):
                search_pubmed_articles("chest pain")

    def test_query_is_parameterized(self):
        """Query text must go through a QueryJobConfig parameter, not f-string interpolation."""
        from google.cloud import bigquery
        from src.rag.retrieval import search_pubmed_articles

        search_pubmed_articles("malicious'; DROP TABLE pmc_embeddings; --", top_k=1)

        call_kwargs = config.bq_client.query.call_args
        job_config = call_kwargs[1].get("job_config") or call_kwargs[0][1]
        assert isinstance(job_config, bigquery.QueryJobConfig)
        param_values = [p.value for p in job_config.query_parameters]
        assert any("DROP TABLE" in str(v) for v in param_values), \
            "Query text should be passed as a parameter, not interpolated into SQL"
        sql = call_kwargs[0][0]
        assert "DROP TABLE" not in sql, "Query text must not be interpolated into the SQL string"

    def test_top_k_in_sql(self):
        """top_k value must appear in the SQL (for LIMIT and top_k =>)."""
        from src.rag.retrieval import search_pubmed_articles

        search_pubmed_articles("stroke", top_k=7)
        sql = config.bq_client.query.call_args[0][0]
        assert "7" in sql

    def test_uses_owned_tables(self):
        """SQL must reference the owned tables, not the public source."""
        from src.rag.retrieval import search_pubmed_articles

        search_pubmed_articles("fever", top_k=3)
        sql = config.bq_client.query.call_args[0][0]
        assert config.PMC_EMBEDDINGS_TABLE in sql
        assert config.PMC_ARTICLES_TABLE in sql
        assert "pmc_open_access_commercial" not in sql, "Should not scan public table directly"


# ── FAISS backend unit tests ─────────────────────────────────────────

class TestFAISSBackend:
    """Tests for the local FAISS backend using mocks."""

    @pytest.fixture(autouse=True)
    def set_faiss_backend(self):
        with patch.object(config, "RETRIEVAL_BACKEND", "faiss"):
            yield

    @pytest.fixture(autouse=True)
    def mock_faiss_deps(self):
        """Mock the FAISS index, embedding call, and SQLite articles db."""
        import sqlite3
        import src.rag.retrieval as retrieval_mod

        # Reset cached singletons
        retrieval_mod._faiss_index = None
        retrieval_mod._pmc_ids_df = None
        retrieval_mod._articles_db = None

        # Build a tiny mock FAISS index
        n_vectors = 3
        dim = 768

        mock_index = MagicMock()
        mock_index.ntotal = n_vectors
        mock_index.nprobe = 64
        # Return similarities (inner product) and indices
        mock_index.search.return_value = (
            np.array([[0.80, 0.70, 0.60]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        pmc_ids_df = pd.DataFrame({
            "pmc_id": ["PMC1111111", "PMC2222222", "PMC3333333"],
            "pmid":   [11111111,     22222222,     33333333],
        })

        # Build an in-memory SQLite database to match production code
        articles_db = sqlite3.connect(":memory:")
        articles_db.execute("""
            CREATE TABLE articles (
                pmc_id TEXT PRIMARY KEY,
                pmid TEXT,
                article_text TEXT,
                article_citation TEXT
            )
        """)
        articles_db.executemany(
            "INSERT INTO articles VALUES (?, ?, ?, ?)",
            [
                ("PMC1111111", "11111111", "Article A text", "J Med. 2022; 1:1"),
                ("PMC2222222", "22222222", "Article B text", "J Med. 2023; 2:2"),
                ("PMC3333333", "33333333", "Article C text", "J Med. 2024; 3:3"),
            ],
        )
        articles_db.commit()

        # Mock _load_faiss_index to return our fake data (3-tuple)
        with patch.object(retrieval_mod, "_load_faiss_index",
                          return_value=(mock_index, pmc_ids_df, articles_db)):
            # Mock _embed_query to return a fake normalised vector
            fake_vec = np.random.default_rng(42).standard_normal((1, dim)).astype(np.float32)
            fake_vec /= np.linalg.norm(fake_vec)
            with patch.object(retrieval_mod, "_embed_query", return_value=fake_vec):
                yield mock_index

    def test_returns_dataframe(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert EXPECTED_COLUMNS_UNIFIED.issubset(set(result.columns)), \
            f"Missing columns: {EXPECTED_COLUMNS_UNIFIED - set(result.columns)}"

    def test_correct_row_count(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert len(result) == 3

    def test_score_is_cosine_distance(self):
        """Score should be 1 - similarity (cosine distance)."""
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        # Mock similarities are [0.80, 0.70, 0.60], so distances are [0.20, 0.30, 0.40]
        np.testing.assert_allclose(result["score"].values, [0.20, 0.30, 0.40], atol=1e-6)

    def test_score_type_is_cosine(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert (result["score_type"] == "cosine_distance").all()

    def test_sorted_by_score(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert result["score"].is_monotonic_increasing

    def test_rank_is_sequential(self):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=3)
        assert list(result["rank"]) == [0, 1, 2]

    def test_handles_empty_results(self, mock_faiss_deps):
        """Should handle case where FAISS returns -1 indices (no results)."""
        mock_faiss_deps.search.return_value = (
            np.array([[-1.0, -1.0, -1.0]], dtype=np.float32),
            np.array([[-1, -1, -1]], dtype=np.int64),
        )
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("nonexistent query", top_k=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert EXPECTED_COLUMNS_UNIFIED.issubset(set(result.columns))


# ── FAISS dedup tests ──────────────────────────────────────────────────

class TestFAISSDedup:
    """Tests that duplicate pmc_ids from IVF multi-probe are deduplicated."""

    @pytest.fixture(autouse=True)
    def set_faiss_backend(self):
        with patch.object(config, "RETRIEVAL_BACKEND", "faiss"):
            yield

    @pytest.fixture(autouse=True)
    def mock_faiss_with_dupes(self):
        """Mock FAISS returning duplicate pmc_ids (same vector from multiple cells)."""
        import sqlite3
        import src.rag.retrieval as retrieval_mod

        retrieval_mod._faiss_index = None
        retrieval_mod._pmc_ids_df = None
        retrieval_mod._articles_db = None

        dim = 768
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_index.nprobe = 64

        # Simulate IVF duplication: PMC1111111 appears at indices 0 and 3,
        # PMC2222222 appears at indices 1 and 4, PMC3333333 at index 2.
        # Index 3 has lower similarity (0.75) than index 0 (0.80) for PMC1111111.
        mock_index.search.return_value = (
            np.array([[0.80, 0.70, 0.60, 0.75, 0.65]], dtype=np.float32),
            np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
        )

        pmc_ids_df = pd.DataFrame({
            "pmc_id": ["PMC1111111", "PMC2222222", "PMC3333333", "PMC1111111", "PMC2222222"],
            "pmid":   [11111111,     22222222,     33333333,     11111111,     22222222],
        })

        articles_db = sqlite3.connect(":memory:")
        articles_db.execute("""
            CREATE TABLE articles (
                pmc_id TEXT PRIMARY KEY,
                pmid TEXT,
                article_text TEXT,
                article_citation TEXT
            )
        """)
        articles_db.executemany(
            "INSERT INTO articles VALUES (?, ?, ?, ?)",
            [
                ("PMC1111111", "11111111", "Article A text", "J Med. 2022; 1:1"),
                ("PMC2222222", "22222222", "Article B text", "J Med. 2023; 2:2"),
                ("PMC3333333", "33333333", "Article C text", "J Med. 2024; 3:3"),
            ],
        )
        articles_db.commit()

        with patch.object(retrieval_mod, "_load_faiss_index",
                          return_value=(mock_index, pmc_ids_df, articles_db)):
            fake_vec = np.random.default_rng(42).standard_normal((1, dim)).astype(np.float32)
            fake_vec /= np.linalg.norm(fake_vec)
            with patch.object(retrieval_mod, "_embed_query", return_value=fake_vec):
                yield mock_index

    def test_dedup_removes_duplicate_pmc_ids(self):
        """5 raw hits with 2 duplicates should yield 3 unique articles."""
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=5)
        assert len(result) == 3
        assert result["pmc_id"].nunique() == 3

    def test_dedup_keeps_best_score(self):
        """For duplicated PMC1111111 (sim 0.80 and 0.75), keep the better one (dist 0.20)."""
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=5)
        pmc1_score = result.loc[result["pmc_id"] == "PMC1111111", "score"].iloc[0]
        # Best similarity 0.80 → cosine distance 0.20
        assert pmc1_score == pytest.approx(0.20, abs=1e-6)

    def test_dedup_preserves_sort_order(self):
        """After dedup, results should still be sorted by score ascending."""
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=5)
        assert result["score"].is_monotonic_increasing

    def test_dedup_top_k_respected(self):
        """top_k=2 with duplicates should return exactly 2 unique articles."""
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain", top_k=2)
        assert len(result) == 2
        assert result["pmc_id"].nunique() == 2


# ── FAISS dependency & real-index tests ───────────────────────────────

class TestFAISSRealIndex:
    """Tests that exercise the real FAISS library (not mocked)."""

    def test_faiss_importable(self):
        """faiss-cpu must be installed for the FAISS backend to work."""
        import faiss  # noqa: F401

    def test_tiny_real_index(self):
        """Build a tiny FAISS index and verify search results are sane."""
        import faiss

        dim = 768
        n_vectors = 50
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
        faiss.normalize_L2(vectors)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 4, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = 4  # search all clusters for exact results

        query = vectors[0:1].copy()  # search for first vector
        sims, idxs = index.search(query, 5)

        assert idxs[0][0] == 0, "Nearest neighbour of vector 0 should be itself"
        assert sims[0][0] == pytest.approx(1.0, abs=1e-4), "Self-similarity should be ~1.0"


# ── Backend dispatch tests ────────────────────────────────────────────

class TestBackendDispatch:
    """Tests that search_pubmed_articles dispatches correctly."""

    def test_invalid_backend_raises(self):
        with patch.object(config, "RETRIEVAL_BACKEND", "invalid"):
            from src.rag.retrieval import search_pubmed_articles
            with pytest.raises(ValueError, match="Unknown RETRIEVAL_BACKEND"):
                search_pubmed_articles("test")


# ── Integration tests (INTEGRATION=1 required) ──────────────────────

@integration
def test_integration_bq_returns_rows():
    config.setup_clients(force_bq=True)
    with patch.object(config, "RETRIEVAL_BACKEND", "bq"):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("chest pain triage", top_k=3)
        assert len(result) == 3
        assert result["distance"].is_monotonic_increasing


@integration
def test_integration_bq_distance_range():
    config.setup_clients(force_bq=True)
    with patch.object(config, "RETRIEVAL_BACKEND", "bq"):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("sepsis fever", top_k=5)
        assert (result["distance"] >= 0).all()
        assert (result["distance"] <= 2).all(), "Cosine distance should be in [0, 2]"


@integration
def test_integration_bq_article_text_nonempty():
    config.setup_clients(force_bq=True)
    with patch.object(config, "RETRIEVAL_BACKEND", "bq"):
        from src.rag.retrieval import search_pubmed_articles
        result = search_pubmed_articles("stroke neurological", top_k=3)
        assert result["article_text"].str.len().gt(0).all(), "article_text should be non-empty"
