"""LLM response generation grounded in retrieved PubMed articles.

Assembles a context block from retrieved articles, constructs a
grounded prompt, and calls the provider-agnostic LLM interface
to produce an answer.
"""

import pandas as pd
import src.config as config
from src.llm import generate as llm_generate
from src.llm.types import GenerationConfig


def generate_rag_response(
    query: str,
    context_articles: pd.DataFrame,
    article_len: int = 3000,
    model_id: str | None = None,
) -> str:
    """Build a context string from retrieved articles and call the LLM.

    Parameters:
    query:              dict with patient data (patient_info,
                          chiefcomplaint, pain, initial_vitals,
                          past_medical, tests, HPI)
    context_articles:   retrieved PubMed articles (from search_pubmed_articles)
    article_len:        length of article text to be included in prompt to llm
    model_id:           LLM model to use (default: config.MODEL_ID)
    """
    context_parts = []
    for i, row in context_articles.iterrows():
        text = row["article_text"][:article_len]
        context_parts.append(f"Article {i+1} (PMID {row['pmid']}): \n{text}")

    context_block = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful medical assistant. "
        "Using ONLY the provided context articles, answer the following query. "
        "If the context does not contain enough information, say so.\n\n"
        f"## Context\n{context_block}\n\n"
        f"## Query\n{query}\n\n"
        "## Answer\n"
    )

    response = llm_generate(
        prompt,
        model_id=model_id or config.MODEL_ID,
        config=GenerationConfig(temperature=0.2),
    )
    return response.text


