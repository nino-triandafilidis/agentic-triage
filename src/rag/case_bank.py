"""ESI v4 case bank: parse, index, and search the 60 practice/competency cases.

Source: Gilboy N et al. *Emergency Severity Index, Version 4: Implementation
Handbook*. AHRQ Pub. No. 05-0046-2, May 2005.  Chapters 9 (practice) & 10
(competency), pp. 63-72.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CASE_BANK_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "corpus"
    / "esi_v4_practice_cases.md"
)


@dataclass(frozen=True)
class ESICase:
    """One parsed ESI practice/competency case."""

    chapter: str  # "practice" or "competency"
    case_number: int  # 1-30 within chapter
    vignette: str  # clinical vignette text
    esi_level: int  # 1-5
    rationale: str  # rationale text

    @property
    def case_id(self) -> str:
        return f"{self.chapter}_{self.case_number:02d}"

    def to_display(self) -> str:
        """Format for inclusion in LLM tool result."""
        return (
            f"[{self.chapter.title()} Case {self.case_number}]\n"
            f"Presentation: {self.vignette}\n"
            f"ESI Level: {self.esi_level}\n"
            f"Rationale: {self.rationale}"
        )


def parse_case_bank(path: Path | None = None) -> list[ESICase]:
    """Parse the markdown case bank into structured ESICase objects.

    Returns list of 60 ESICase objects (30 practice + 30 competency).
    """
    path = path or _CASE_BANK_PATH
    text = path.read_text(encoding="utf-8")

    cases: list[ESICase] = []
    chapter_map = {
        "Practice Cases": "practice",
        "Competency Cases": "competency",
    }

    # Split on chapter headings
    chapter_splits = re.split(r"^# Chapter \d+\.\s+", text, flags=re.MULTILINE)

    for section in chapter_splits:
        chapter_name = None
        for key, val in chapter_map.items():
            if section.startswith(key):
                chapter_name = val
                break
        if chapter_name is None:
            continue

        # Each case: **N.** vignette\n\n> **ESI level X: label.** body
        case_pattern = re.compile(
            r"\*\*(\d+)\.\*\*\s+"  # case number
            r"(.*?)\n"  # vignette (single paragraph)
            r"\n>\s*\*\*ESI level (\d):\s*"  # ESI level
            r"(.*?)\*\*\s*"  # rationale label (up to **)
            r"(.*?)(?=\n\n---|\Z)",  # rationale body
            re.DOTALL,
        )

        for match in case_pattern.finditer(section):
            case_num = int(match.group(1))
            vignette = match.group(2).strip()
            esi_level = int(match.group(3))
            rationale_label = match.group(4).strip().rstrip(".")
            rationale_body = match.group(5).strip()
            rationale = (
                f"{rationale_label}. {rationale_body}" if rationale_body else rationale_label
            )

            cases.append(
                ESICase(
                    chapter=chapter_name,
                    case_number=case_num,
                    vignette=vignette,
                    esi_level=esi_level,
                    rationale=rationale,
                )
            )

    return cases


# ── Lazy singleton ──────────────────────────────────────────────────────
_CASE_BANK: list[ESICase] | None = None


def get_case_bank() -> list[ESICase]:
    """Return the parsed case bank (lazy singleton)."""
    global _CASE_BANK
    if _CASE_BANK is None:
        _CASE_BANK = parse_case_bank()
    return _CASE_BANK


# ── Search ──────────────────────────────────────────────────────────────


def search_cases(
    *,
    esi_level: int | None = None,
    keywords: str | None = None,
    chapter: str | None = None,
    max_results: int = 10,
) -> list[ESICase]:
    """Search the case bank with optional filters.

    Args:
        esi_level: Filter to cases with this ESI level (1-5).
        keywords: Space-separated keywords; cases must contain ALL keywords
                  (case-insensitive) in vignette or rationale.
        chapter: Filter to "practice" or "competency".
        max_results: Maximum cases to return (default 10, hard cap 15).
    """
    results = list(get_case_bank())

    if esi_level is not None:
        results = [c for c in results if c.esi_level == esi_level]

    if chapter is not None:
        results = [c for c in results if c.chapter == chapter.lower()]

    if keywords is not None:
        kw_list = keywords.lower().split()
        results = [
            c
            for c in results
            if all(
                kw in f"{c.vignette} {c.rationale}".lower() for kw in kw_list
            )
        ]

    cap = min(max_results, 15)
    return results[:cap]


def format_tool_result(cases: list[ESICase]) -> str:
    """Format search results for insertion into tool-result message."""
    if not cases:
        return "No matching cases found in the ESI case bank."
    parts = [f"Found {len(cases)} matching case(s):\n"]
    for case in cases:
        parts.append(case.to_display())
    return "\n\n".join(parts)


# ── Tool definition ─────────────────────────────────────────────────────

CASE_BANK_TOOL_SCHEMA: dict = {
    "name": "search_esi_case_bank",
    "description": (
        "Search the ESI v4 case bank containing 60 expert-classified triage "
        "cases (30 practice + 30 competency) from the ESI Implementation "
        "Handbook. Each case includes a clinical vignette, the correct ESI "
        "level (1-5), and the clinical rationale. Use this tool to find "
        "similar cases that can inform your triage decision. You may call "
        "this tool multiple times with different filters."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "esi_level": {
                "type": "integer",
                "description": (
                    "Filter to cases assigned this ESI level (1-5). "
                    "Omit to search across all levels."
                ),
            },
            "keywords": {
                "type": "string",
                "description": (
                    "Space-separated keywords to search for in clinical "
                    "vignettes and rationales. All keywords must match "
                    "(case-insensitive). Examples: 'chest pain', "
                    "'diabetic vomiting', 'laceration pediatric'."
                ),
            },
            "chapter": {
                "type": "string",
                "enum": ["practice", "competency"],
                "description": "Filter to a specific chapter. Omit to search both.",
            },
        },
        "required": [],
    },
}
