"""Shared front-matter stripping for PMC article text.

Used by:
  - scripts/build_bm25_index.py  (index-time preprocessing)
  - src/rag/retrieval.py          (query-time context_text generation)

Marker list and earliest-marker logic ported from
experiments/E04_snippet_cleaning.py, validated in E04/E07.
"""

from __future__ import annotations

import re

_SECTION_MARKERS = [
    "\n==== Body",
    "\nIntroduction\n",
    "\nINTRODUCTION\n",
    "\nBackground\n",
    "\nBACKGROUND\n",
    "\nAbstract\n",
    "\nABSTRACT\n",
    "\nMethods\n",
    "\nMETHODS\n",
    "\nObjective\n",
    "\nOBJECTIVE\n",
    "\nObjectives\n",
    "\nOBJECTIVES\n",
    "\nPurpose\n",
    "\nPURPOSE\n",
    "\nSimple Summary\n",
    "\nResults\n",
    "\nRESULTS\n",
]

_WS_RE = re.compile(r"\s+")


def extract_body(article_text: str, max_chars: int) -> str | None:
    """Extract the first *max_chars* of substantive body content.

    Scans *article_text* for the earliest section marker from the E04/E07
    validated set.  If found, returns up to *max_chars* of text starting
    immediately after the marker line, with whitespace normalised.

    Returns None if no marker is found (caller decides whether to skip
    or fall back).
    """
    best_pos = len(article_text)
    best_marker: str | None = None

    for marker in _SECTION_MARKERS:
        pos = article_text.find(marker)
        if 0 < pos < best_pos:
            best_pos = pos
            best_marker = marker

    if best_marker is None:
        return None

    start = best_pos + len(best_marker)
    raw = article_text[start : start + max_chars]
    return _WS_RE.sub(" ", raw).strip()


def prepare_article_excerpt(
    article_text: str | None,
    max_chars: int,
    *,
    context_text: str | None = None,
) -> str:
    """Return a cleaned, prompt-ready excerpt capped at *max_chars*.

    Preference order:
      1. Reuse ``context_text`` when a caller already prepared cleaned body text.
      2. Extract substantive body text from ``article_text`` using section markers.
      3. Fall back to whitespace-normalized raw text if no marker is found.
    """
    if context_text:
        return _WS_RE.sub(" ", context_text).strip()[:max_chars]
    if not article_text:
        return ""

    cleaned = extract_body(article_text, max_chars)
    if cleaned:
        return cleaned[:max_chars]

    return _WS_RE.sub(" ", article_text).strip()[:max_chars]
