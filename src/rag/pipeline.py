"""End-to-end RAG pipeline: query -> retrieve -> generate.

Accepts either a free-text query or structured medLLMbenchmark fields
(HPI, patient_info, initial_vitals) and returns an LLM-generated answer
grounded in PubMed literature.
"""

from src.rag.retrieval import search_pubmed_articles
from src.rag.generation import generate_rag_response


def rag_pipeline(
    query: str | None = None,
    *,
    hpi: str | None = None,
    patient_info: str | None = None,
    initial_vitals: str | None = None,
    top_k: int = 10,
) -> str:
    """Run the full RAG pipeline.

    Two calling modes:
        # Manual query (for testing / ad-hoc use)
        rag_pipeline(query="What causes acute chest pain?")

        # From medLLMbenchmark fields (auto-builds query)
        rag_pipeline(hpi="chest pain...", patient_info="Male, 62")
    """
    if query is None:
        if hpi is None:
            raise ValueError("Provide either `query` or `hpi`.")
        parts = [hpi]
        if patient_info:
            parts.append(patient_info)
        if initial_vitals:
            parts.append(initial_vitals)
        query = " ".join(parts)

    articles = search_pubmed_articles(query, top_k=top_k)
    print(f"Retrieved {len(articles)} articles")
    return generate_rag_response(query, articles)
