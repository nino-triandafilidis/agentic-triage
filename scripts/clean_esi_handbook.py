#!/usr/bin/env python3
"""
clean_esi_handbook.py — Clean the PDF-extracted ESI Handbook v5 for use as
LLM context in ESI triage prediction.

══════════════════════════════════════════════════════════════════════════════
 SECTION-BY-SECTION DECISIONS
══════════════════════════════════════════════════════════════════════════════

KEEP
────
Chapter 1 – Introduction to the Emergency Severity Index
    Establishes what ESI is: a 5-level acuity scale, 94% adoption in US EDs,
    59% accuracy by trained nurses (strong baseline motivation). ACEP/ENA
    endorsement and reliability evidence across countries and age groups.

Chapter 2 – Overview of the ESI Algorithm
    The algorithmic core:
      • 4 sequential decision points (A = lifesaving?, B = high-risk?,
        C = how many resources?, D = high-risk vitals?)
      • Full algorithm with HR/RR thresholds by age group
      • Resource table (what counts as a resource vs. not: labs, ECG, CT,
        IV fluids, IV medications, specialty consult vs. H&P, oral meds,
        splints, tetanus)
      • Pediatric fever thresholds (ESI 2 if neonate fever >38°C, etc.)
      • Bias and stigma section: evidence for undertriage in minority,
        elderly, female, and behavioral-health populations.

Chapter 3 – Decision Point A: Lifesaving Intervention Required?
    Defines ESI level 1: unresponsive, active seizure, occluded airway,
    severe respiratory distress, profound hypotension/hypoglycemia.
    Full Table of Lifesaving Interventions (airway, electrical, procedures,
    hemodynamics, medications: epinephrine, naloxone, dextrose, etc.).
    Key nuances: method of arrival ≠ ESI 1; discharge ≠ non-ESI-1
    (e.g., resolved hypoglycemia or anaphylaxis can leave and still were ESI 1).

Chapter 4 – Decision Point B: High-Risk Presentation?
    Richest chapter. Defines ESI level 2 via three questions:
      • Is the situation high-risk / could deteriorate?
      • Is there new AMS (confusion / lethargy / disorientation)?
      • Is there severe pain (≥7/10) or severe distress?
    Then walks through red-flag criteria by organ system:
      Neurological   – thunderclap headache, post-ictal AMS, stroke signs,
                       headache + fever / nuchal rigidity
      Ocular         – sudden vision loss, intolerable eye pain, trauma
      ENT            – respiratory stridor, posterior epistaxis on anticoagulants,
                       button-battery ingestion (time-sensitive in <6-year-olds)
      Respiratory    – tachypnea + tachycardia in distress; peds: grunting,
                       retractions, belly breathing
      Cardiovascular – chest pain / ACS (including atypical presentation in
                       women), abnormal ECG, post-COVID cardiac risk;
                       hemodynamically unstable → ESI 1
      Abdominal/GI   – sepsis signs, hypoperfusion, ectopic pregnancy,
                       pregnancy complications; 31% undertriaged overall;
                       elderly most undertriaged (52%)
      OB/GYN         – SBP <90 or >150 in pregnant/postpartum → ESI 2 even
                       without other symptoms; heavy vaginal bleeding
      Genitourinary  – testicular torsion (time-sensitive), ovarian torsion,
                       urosepsis, severe flank pain
      Trauma         – falls ≥20 ft, ejection, penetrating head/neck/chest/abd,
                       high-risk ortho (pelvic/femur fracture, compartment
                       syndrome, neurovascular compromise); occult hypoperfusion
                       in age >55 with normal vitals
      Ingestions     – any toxic ingestion with AMS, respiratory changes, or
                       arrhythmia
      Transplant     – fever or infection in immunocompromised recipient → ESI 2
      Mental/Behavioral Health – suicidal, homicidal, psychotic, violent → ESI 2;
                       >12% of all ED visits

Chapter 5 – Decision Point C: How Many Resources?
    Defines ESI 3 (≥2 resources), ESI 4 (1 resource), ESI 5 (0 resources).
    Resource counting rules (type not quantity: CBC + electrolytes = 1 resource;
    CBC + CXR = 2 resources). Table 5-1 (resources vs. non-resources) and
    Table 5-2 (worked examples with patient presentations and ESI assignments).

Chapter 6 – Decision Point D: High-Risk Vital Signs?
    HR/RR age-stratified thresholds; SpO2 <92%.
    5 worked examples showing uptriage from ESI 3 → 2.
    Key nuance: vitals must be interpreted in clinical context (e.g., COPD
    baseline SpO2, immunosuppression blunting fever response).

REMOVE
──────
Front matter (pages 3–6)
    Table of contents, Dedication, Preface, Acknowledgements.
    The Preface's useful content (bias/undertriage, emphasis on vitals) is
    already covered more fully in Chapter 2.

Chapter navigation sidebars
    PDF layout artifact repeated at the top of every chapter:
    "Chapter 1 Introduction Chapter 2: Overview Chapter 3: Decision Point A…"

--- Page N --- markers and inline page-header noise
    PDF extraction artifacts: running headers, page numbers.

Drop-cap split characters
    PDF drop-cap letters extracted as isolated capital letters split from
    their word: "I\nn 2018…" → rejoin to "In 2018…"

All References sections → companion file
    Saved to esi_handbook_references.txt instead of discarded.
    # NOTE: these references (clinical-guideline and evidence-base citations)
    # may serve as an alternative retrieval corpus for the RAG pipeline —
    # e.g., fetching the full PubMed abstracts behind the cited studies.

Appendix A – Frequently Asked Questions
    Nursing-workflow Q&A (when to change ESI level, who can triage, etc.).
    Not useful for prediction from patient data.

Appendix B – ESI Triage Algorithm v5
    Confirmed duplicate of the algorithm already present in Chapter 2.
    Not present in the extracted text anyway.

══════════════════════════════════════════════════════════════════════════════
 PROCESSING NOTES
══════════════════════════════════════════════════════════════════════════════

Multi-page reference sections (Ch4 spans pages 22-24): cannot stop at the
first page marker. Reference extraction therefore runs AFTER page markers are
removed, using "CHAPTER\\n" (chapter heading) as the terminator instead.

Mixed pages (e.g., page 16 has Ch3 body + Ch3 refs; page 31 has Ch6 examples
+ Ch6 refs): handled correctly because we split on the "References" heading
itself, not on page boundaries.

══════════════════════════════════════════════════════════════════════════════

Usage:
    .venv/bin/python scripts/clean_esi_handbook.py [INPUT] [OUTPUT_CLEAN] [OUTPUT_REFS]

Defaults:
    INPUT:        ~/Library/CloudStorage/.../CS224n/esi_handbook_text.txt
    OUTPUT_CLEAN: same directory, esi_handbook_clean.txt
    OUTPUT_REFS:  same directory, esi_handbook_references.txt
"""

import re
import sys
from pathlib import Path

# ── Default paths ────────────────────────────────────────────────────────────
_GDRIVE = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-ninot@stanford.edu"
    / "Shared drives/CS224n"
)
_PROJECT = Path(__file__).parent.parent  # repo root

DEFAULT_INPUT       = str(_GDRIVE / "esi_handbook_text.txt")
DEFAULT_OUT_CLEAN   = str(_PROJECT / "data" / "corpus" / "esi_handbook_clean.txt")
DEFAULT_OUT_REFS    = str(_PROJECT / "data" / "corpus" / "esi_handbook_references.txt")


# ── Nav-sidebar pattern ───────────────────────────────────────────────────────
# This block is repeated verbatim at the top of every chapter (PDF layout bar).
# "Chapter 2:" may or may not have a trailing space before the newline.
_NAV_SIDEBAR = re.compile(
    r"CHAPTER\s*\n\s*\d+\s*\n"         # "CHAPTER\nN\n"
    r"(?:"                              # optionally followed by the full sidebar
    r"Chapter 1\s+Introduction\s+"
    r"Chapter 2:?\s+Overview\s+"
    r"Chapter 3:?\s+Decision Point A\s+"
    r"Chapter 4:?\s+Decision Point B\s+"
    r"Chapter 5:?\s+Decision Point C\s+"
    r"Chapter 6:?\s+Decision Point D\s*Appendix A\s*Appendix B\s*"
    r")?",
    re.MULTILINE | re.DOTALL,
)


def _extract_refs(text: str) -> tuple[str, list[str]]:
    """
    Extract every References section and return (cleaned_text, [ref_blocks]).

    Must be called AFTER page markers are removed, because Ch4 references span
    three pages — using page markers as terminators would split the block.
    Instead we use "CHAPTER\\n" (the chapter-heading token) or end-of-string
    as the section boundary.

    # NOTE: The extracted citation blocks are saved to a companion file and may
    # serve as an alternative retrieval database for the RAG pipeline (e.g.
    # fetching PubMed abstracts for the underlying clinical studies).
    """
    blocks: list[str] = []

    def _capture(m: re.Match) -> str:
        blocks.append(m.group(0).strip())
        return ""

    # "^References" → everything up to the next chapter cover ("CHAPTER\n")
    # or end of string.  re.DOTALL lets .* span newlines.
    pattern = re.compile(
        r"^References\s*\n.*?(?=CHAPTER\s*\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    text = pattern.sub(_capture, text)
    return text, blocks


def clean(input_path: str, out_clean: str, out_refs: str) -> None:
    src = Path(input_path).read_text(encoding="utf-8")

    # ── 1. Remove front matter ───────────────────────────────────────────────
    # Pages 3-6 (ToC, Dedication, Preface, Acknowledgements).
    # Chapter 1 body begins on page 7.
    src = re.sub(
        r"\A.*?(?=^--- Page 7 ---\s*\n)",
        "",
        src,
        flags=re.DOTALL | re.MULTILINE,
    )

    # ── 2. Remove Appendix A (and everything after) ──────────────────────────
    # Appendix A (FAQ/Q&A) starts on page 33.  Appendix B (algorithm figure)
    # would follow but was not extracted from the PDF — moot either way.
    src = re.sub(
        r"^--- Page 33 ---.*\Z",
        "",
        src,
        flags=re.DOTALL | re.MULTILINE,
    )

    # ── 3. Remove --- Page N --- markers ────────────────────────────────────
    # Must happen before reference extraction (step 4) so Ch4's three-page
    # reference block is not prematurely split by an intervening page marker.
    src = re.sub(r"^--- Page \d+ ---\s*\n", "", src, flags=re.MULTILINE)

    # ── 4. Extract and remove all References sections ────────────────────────
    src, ref_blocks = _extract_refs(src)

    # ── 5. Remove chapter cover-page artifacts (nav sidebars) ────────────────
    # "CHAPTER\nN\n" + the navigation sidebar block.
    src = _NAV_SIDEBAR.sub("", src)

    # ── 6. Remove inline page-header noise ───────────────────────────────────

    # "10\nEMERGENCY SEVERITY INDEX – V5" — page number + running header
    src = re.sub(
        r"^\d+\nEMERGENCY SEVERITY INDEX[^\n]*\n",
        "",
        src,
        flags=re.MULTILINE,
    )

    # "13CHAPTER 4 – Decision Point B: …" — page-number prefix on chapter title
    src = re.sub(r"^\d+CHAPTER \d+[^\n]*\n", "", src, flags=re.MULTILINE)

    # Standalone page numbers left over (a line containing only digits)
    # Safe to remove: all meaningful numeric content (vital-sign thresholds,
    # scores) always has surrounding text or "> " / "< " prefixes.
    src = re.sub(r"^\d+\s*$", "", src, flags=re.MULTILINE)

    # ── 7. Rejoin PDF drop-cap split characters ──────────────────────────────
    # Each chapter begins with a large drop-cap letter that the PDF extractor
    # puts on its own line, severed from the rest of the word.
    # Ch1: "I\nn 2018…"   Ch2: "A\nlgorithms…"   Ch3: "A\nt decision…"
    # Ch4: "I\nf it is…"  Ch5: "O\nnce an ESI…"  Ch6: "T\no reach…"
    src = re.sub(r"\n([A-Z])\n([a-z])", r"\n\1\2", src)

    # ── 8. Collapse excess blank lines ───────────────────────────────────────
    src = re.sub(r"\n{3,}", "\n\n", src)

    # ── Write outputs ─────────────────────────────────────────────────────────
    Path(out_clean).write_text(src.strip() + "\n", encoding="utf-8")

    # Separate reference blocks with a divider for readability
    refs_out = "\n\n---\n\n".join(ref_blocks)
    Path(out_refs).write_text(refs_out.strip() + "\n", encoding="utf-8")

    orig_lines  = len(Path(input_path).read_text(encoding="utf-8").splitlines())
    clean_lines = len(src.splitlines())
    refs_lines  = len(refs_out.splitlines())
    print(f"Clean:  {out_clean}")
    print(f"        {orig_lines} → {clean_lines} lines  "
          f"({100 * (orig_lines - clean_lines) // orig_lines}% reduction)")
    print(f"Refs:   {out_refs}")
    print(f"        {refs_lines} lines across {len(ref_blocks)} chapters")


if __name__ == "__main__":
    inp      = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_cln  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_CLEAN
    out_refs = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT_REFS
    clean(inp, out_cln, out_refs)
