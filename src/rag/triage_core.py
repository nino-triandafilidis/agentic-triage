"""Shared triage helpers: prompts, parsing, pricing, context formatting.

Extracted from experiments/run_rag_triage.py so that src/ modules never
import from experiments/.
"""

from __future__ import annotations

import pandas as pd

from src.rag.text_cleaning import prepare_article_excerpt
from src.schemas import TriagePrediction

# ── Generation config ────────────────────────────────────────────────────
GENERATION_TEMPERATURE = 0.0


def format_vitals_for_prompt(raw_value) -> str:
    """Render initial_vitals for prompt insertion.

    Maps None, NaN, pd.NA, empty/whitespace-only strings, and the literal
    ``"nan"``/``"<NA>"`` to ``"Missing/not recorded"`` so the model receives
    an explicit signal rather than a stringified Python sentinel.
    """
    try:
        if pd.isna(raw_value):
            return "Missing/not recorded"
    except (TypeError, ValueError):
        pass
    s = str(raw_value).strip()
    if s == "" or s.lower() in ("nan", "<na>"):
        return "Missing/not recorded"
    return s

# ── Triage prompts ───────────────────────────────────────────────────────
TRIAGE_PROMPT_WITH_RAG = (
    "You are a nurse with emergency and triage experience. "
    "The following PubMed literature may be relevant to this case:\n\n"
    "{context}\n\n"
    "Using the patient's history of present illness, their information and "
    "initial vitals, determine the triage level based on the Emergency Severity "
    "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
    "(lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_LLM_ONLY = (
    "You are a nurse with emergency and triage experience. "
    "Using the patient's history of present illness, their information and "
    "initial vitals, determine the triage level based on the Emergency Severity "
    "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
    "(lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

# ── Handbook-prefixed prompt variants ────────────────────────────────────
# These prepend the full ESI Handbook text as system-level context.
# The {esi_handbook} placeholder is filled by the caller with the loaded text.

TRIAGE_PROMPT_HANDBOOK_RAG = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "The following PubMed literature may also be relevant to this case:\n\n"
    "{context}\n\n"
    "Using the ESI algorithm from the handbook above, the patient's history "
    "of present illness, their information and initial vitals, determine the "
    "triage level (ESI 1-5).\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_HANDBOOK_ONLY = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "Using the ESI algorithm from the handbook above, the patient's history "
    "of present illness, their information and initial vitals, determine the "
    "triage level (ESI 1-5).\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

# ── Gaber et al. (2024) prompt variants ──────────────────────────────────
# More detailed ESI descriptions (with examples per level).
# Adapted from Claude_triage_ClinicalUser.py; output format changed from
# <acuity> tag to our JSON schema (handled at generation config level).

TRIAGE_PROMPT_GABER_RAG = (
    "You are a nurse with emergency and triage experience. "
    "The following PubMed literature may be relevant to this case:\n\n"
    "{context}\n\n"
    "Using the patient's history of present illness, his information and "
    "initial vitals, determine the triage level based on the Emergency "
    "Severity Index (ESI), ranging from ESI level 1 (highest acuity) to "
    "ESI level 5 (lowest acuity): "
    "1: Assign if the patient requires immediate lifesaving intervention. "
    "2: Assign if the patient is in a high-risk situation (e.g., confused, "
    "lethargic, disoriented, or experiencing severe pain/distress)  "
    "3: Assign if the patient requires two or more diagnostic or therapeutic "
    "interventions and their vital signs are within acceptable limits for "
    "non-urgent care. "
    "4: Assign if the patient requires one diagnostic or therapeutic "
    "intervention (e.g., lab test, imaging, or EKG). "
    "5: Assign if the patient does not require any diagnostic or therapeutic "
    "interventions beyond a physical exam (e.g., no labs, imaging, or "
    "wound care).\n"
    "History of present illness: {hpi}, patient info: {patient_info} "
    "and initial vitals: {initial_vitals}."
)

TRIAGE_PROMPT_GABER_LLM_ONLY = (
    "You are a nurse with emergency and triage experience. "
    "Using the patient's history of present illness, his information and "
    "initial vitals, determine the triage level based on the Emergency "
    "Severity Index (ESI), ranging from ESI level 1 (highest acuity) to "
    "ESI level 5 (lowest acuity): "
    "1: Assign if the patient requires immediate lifesaving intervention. "
    "2: Assign if the patient is in a high-risk situation (e.g., confused, "
    "lethargic, disoriented, or experiencing severe pain/distress)  "
    "3: Assign if the patient requires two or more diagnostic or therapeutic "
    "interventions and their vital signs are within acceptable limits for "
    "non-urgent care. "
    "4: Assign if the patient requires one diagnostic or therapeutic "
    "intervention (e.g., lab test, imaging, or EKG). "
    "5: Assign if the patient does not require any diagnostic or therapeutic "
    "interventions beyond a physical exam (e.g., no labs, imaging, or "
    "wound care).\n"
    "History of present illness: {hpi}, patient info: {patient_info} "
    "and initial vitals: {initial_vitals}."
)

TRIAGE_PROMPT_GABER_HANDBOOK_RAG = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "The following PubMed literature may also be relevant to this case:\n\n"
    "{context}\n\n"
    "Using the ESI algorithm from the handbook above, the patient's history "
    "of present illness, his information and initial vitals, determine the "
    "triage level based on the Emergency Severity Index (ESI), ranging from "
    "ESI level 1 (highest acuity) to ESI level 5 (lowest acuity): "
    "1: Assign if the patient requires immediate lifesaving intervention. "
    "2: Assign if the patient is in a high-risk situation (e.g., confused, "
    "lethargic, disoriented, or experiencing severe pain/distress)  "
    "3: Assign if the patient requires two or more diagnostic or therapeutic "
    "interventions and their vital signs are within acceptable limits for "
    "non-urgent care. "
    "4: Assign if the patient requires one diagnostic or therapeutic "
    "intervention (e.g., lab test, imaging, or EKG). "
    "5: Assign if the patient does not require any diagnostic or therapeutic "
    "interventions beyond a physical exam (e.g., no labs, imaging, or "
    "wound care).\n"
    "History of present illness: {hpi}, patient info: {patient_info} "
    "and initial vitals: {initial_vitals}."
)

TRIAGE_PROMPT_GABER_HANDBOOK_ONLY = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "Using the ESI algorithm from the handbook above, the patient's history "
    "of present illness, his information and initial vitals, determine the "
    "triage level based on the Emergency Severity Index (ESI), ranging from "
    "ESI level 1 (highest acuity) to ESI level 5 (lowest acuity): "
    "1: Assign if the patient requires immediate lifesaving intervention. "
    "2: Assign if the patient is in a high-risk situation (e.g., confused, "
    "lethargic, disoriented, or experiencing severe pain/distress)  "
    "3: Assign if the patient requires two or more diagnostic or therapeutic "
    "interventions and their vital signs are within acceptable limits for "
    "non-urgent care. "
    "4: Assign if the patient requires one diagnostic or therapeutic "
    "intervention (e.g., lab test, imaging, or EKG). "
    "5: Assign if the patient does not require any diagnostic or therapeutic "
    "interventions beyond a physical exam (e.g., no labs, imaging, or "
    "wound care).\n"
    "History of present illness: {hpi}, patient info: {patient_info} "
    "and initial vitals: {initial_vitals}."
)

# ── Few-shot demonstration block ─────────────────────────────────────────
# 5 gold-standard cases from ESI v4 Implementation Handbook (Chapter 9).
# Selection rationale: one per ESI level, chosen for boundary-teaching value.
# See data/corpus/esi_v4_fewshot_bank.md for provenance and full rationale.

FEWSHOT_EXAMPLES_BLOCK = (
    "Below are five reference triage cases from the ESI Handbook with their "
    "correct ESI levels and clinical rationales. Use them to calibrate your "
    "judgement.\n\n"
    "--- Example 1 ---\n"
    "Case: A 76-year-old male is brought to the ED because of severe abdominal "
    "pain. He tells you \"it feels like someone is ripping me apart.\" The pain "
    "began about 30 minutes prior to admission and he rates the intensity as "
    "20/10. He has hypertension for which he takes a diuretic. No allergies. "
    "The patient is sitting in a wheelchair moaning in pain. His skin is cool "
    "and diaphoretic. VS: HR 122, BP 88/68, RR 24, SpO2 94%.\n"
    "ESI: 1\n"
    "Rationale: Signs of shock — hypotensive, tachycardic, decreased peripheral "
    "perfusion. Needs immediate IV access, aggressive fluid resuscitation.\n\n"
    "--- Example 2 ---\n"
    "Case: A 44-year-old female is retching continuously into a large basin as "
    "her son wheels her into the triage area. Her son tells you that his "
    "diabetic mother has been vomiting for the past 5 hours and now it is "
    "\"just this yellow stuff.\" \"She hasn't eaten or taken her insulin.\" "
    "NKDA. VS: BP 148/70, P 126, RR 24.\n"
    "ESI: 2\n"
    "Rationale: High risk for diabetic ketoacidosis. Vital signs are concerning "
    "(tachycardic, tachypneic). Unsafe to wait.\n\n"
    "--- Example 3 ---\n"
    "Case: A 27-year-old female wants to be checked by a doctor. She has been "
    "experiencing low abdominal pain (6/10) for about 4 days. This morning she "
    "began spotting. She denies nausea, vomiting, diarrhea, or urinary "
    "symptoms. LMP 7 weeks ago. PMH: previous ectopic pregnancy. VS: T 98° F, "
    "HR 66, RR 14, BP 106/68.\n"
    "ESI: 3\n"
    "Rationale: Needs two or more resources (lab + ultrasound). Ectopic "
    "pregnancy on differential but hemodynamically stable.\n\n"
    "--- Example 4 ---\n"
    "Case: \"I have a fever and a sore throat. I have finals this week and I am "
    "scared this is strep,\" reports a 19-year-old college student. Sitting at "
    "triage drinking bottled water. No PMH, medications: birth control pills, "
    "no allergies. VS: T 100.6° F, HR 88, RR 18, BP 112/76.\n"
    "ESI: 4\n"
    "Rationale: One resource — rapid strep screen sent to lab.\n\n"
    "--- Example 5 ---\n"
    "Case: \"My dentist can't see me until Monday and my tooth is killing me. "
    "Can't you give me something for the pain?\" a 38-year-old healthy male "
    "asks the triage nurse. Pain started yesterday, rates pain 10/10. No "
    "obvious facial swelling. Allergic to Penicillin. VS: T 99.8° F, HR 78, "
    "RR 16, BP 128/74.\n"
    "ESI: 5\n"
    "Rationale: No resources needed. Exam + prescription only. Pain rating "
    "alone does not determine ESI level.\n\n"
)

# ── Private CoT instruction ──────────────────────────────────────────────
COT_PRIVATE_INSTRUCTION = (
    "Before giving your final answer, reason through the ESI algorithm "
    "step by step internally: (A) Does this patient require immediate "
    "life-saving intervention? (B) Is this a high-risk situation, or is the "
    "patient confused/lethargic/in severe pain? (C) How many resources will "
    "this patient need? (D) Are vital signs in the danger zone? "
    "Use this reasoning to determine the ESI level, but only output your "
    "final answer in the required JSON format.\n\n"
)

# ── E18 prompt variants (LLM-only) ──────────────────────────────────────

TRIAGE_PROMPT_COT_PRIVATE = (
    "You are a nurse with emergency and triage experience. "
    "Using the patient's history of present illness, their information and "
    "initial vitals, determine the triage level based on the Emergency Severity "
    "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
    "(lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    + COT_PRIVATE_INSTRUCTION
    + "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_FEWSHOT = (
    "You are a nurse with emergency and triage experience. "
    "Using the patient's history of present illness, their information and "
    "initial vitals, determine the triage level based on the Emergency Severity "
    "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
    "(lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    + FEWSHOT_EXAMPLES_BLOCK
    + "Now classify this patient:\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_FEWSHOT_COT_PRIVATE = (
    "You are a nurse with emergency and triage experience. "
    "Using the patient's history of present illness, their information and "
    "initial vitals, determine the triage level based on the Emergency Severity "
    "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
    "(lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    + FEWSHOT_EXAMPLES_BLOCK
    + COT_PRIVATE_INSTRUCTION
    + "Now classify this patient:\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

# ── Tool-use prompt ─────────────────────────────────────────────────────
# Used with tool-use mode: the model has access to a case-bank search tool
# and decides autonomously whether to consult reference cases.

TRIAGE_PROMPT_TOOL_USE = (
    "You are a nurse with emergency and triage experience. "
    "You have access to a tool called `search_esi_case_bank` that lets you "
    "search a bank of 60 expert-classified ESI triage cases from the ESI "
    "Implementation Handbook. Each case includes a clinical vignette, the "
    "correct ESI level, and the clinical rationale.\n\n"
    "Before making your triage decision, consider whether consulting similar "
    "reference cases would help you classify this patient more accurately. "
    "You may call the tool zero or more times with different search filters.\n\n"
    "Once you have enough information, determine the triage level based on "
    "the Emergency Severity Index (ESI), ranging from ESI level 1 (highest "
    "acuity) to ESI level 5 (lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_TOOL_USE_PMC = (
    "You are a nurse with emergency and triage experience. "
    "You have access to a tool called `search_pmc_articles` that lets you "
    "search PubMed Central for peer-reviewed medical literature relevant to "
    "the patient's presentation. You may call the tool zero or more times "
    "with different queries to find evidence that helps you classify this "
    "patient.\n\n"
    "Once you have enough information, determine the triage level based on "
    "the Emergency Severity Index (ESI), ranging from ESI level 1 (highest "
    "acuity) to ESI level 5 (lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_TOOL_USE_DUAL = (
    "You are a nurse with emergency and triage experience. "
    "You have access to two tools:\n"
    "1. `search_esi_case_bank` for handbook triage exemplars with gold ESI labels.\n"
    "2. `search_pmc_articles` for peer-reviewed medical literature from PubMed Central.\n\n"
    "Use either or both tools if they help you classify this patient more "
    "accurately. You may call each tool zero or more times.\n\n"
    "Once you have enough information, determine the triage level based on "
    "the Emergency Severity Index (ESI), ranging from ESI level 1 (highest "
    "acuity) to ESI level 5 (lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)
# ── Composite prompts: fewshot + RAG, tool-use + RAG ────────────────────
# These combine fewshot examples (or tool-use instructions) with the
# {context} slot for PMC articles, enabling experiments that use both
# retrieval context AND fewshot demonstrations (or tool access).

TRIAGE_PROMPT_FEWSHOT_RAG = (
    "You are a nurse with emergency and triage experience. "
    "The following PubMed literature may be relevant to this case:\n\n"
    "{context}\n\n"
    "Using the patient's history of present illness, their information and "
    "initial vitals, determine the triage level based on the Emergency Severity "
    "Index (ESI), ranging from ESI level 1 (highest acuity) to ESI level 5 "
    "(lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    + FEWSHOT_EXAMPLES_BLOCK
    + "Now classify this patient:\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_FEWSHOT_RAG_HANDBOOK_RAG = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "The following PubMed literature may also be relevant to this case:\n\n"
    "{context}\n\n"
    "Using the ESI algorithm from the handbook above, the patient's history "
    "of present illness, their information and initial vitals, determine the "
    "triage level (ESI 1-5).\n\n"
    + FEWSHOT_EXAMPLES_BLOCK
    + "Now classify this patient:\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_FEWSHOT_HANDBOOK_ONLY = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "Using the ESI algorithm from the handbook above, the patient's history "
    "of present illness, their information and initial vitals, determine the "
    "triage level (ESI 1-5).\n\n"
    + FEWSHOT_EXAMPLES_BLOCK
    + "Now classify this patient:\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_TOOL_USE_RAG = (
    "You are a nurse with emergency and triage experience. "
    "You have access to a tool called `search_esi_case_bank` that lets you "
    "search a bank of 60 expert-classified ESI triage cases from the ESI "
    "Implementation Handbook. Each case includes a clinical vignette, the "
    "correct ESI level, and the clinical rationale.\n\n"
    "The following PubMed literature may also be relevant to this case:\n\n"
    "{context}\n\n"
    "Before making your triage decision, consider whether consulting similar "
    "reference cases would help you classify this patient more accurately. "
    "You may call the tool zero or more times with different search filters.\n\n"
    "Once you have enough information, determine the triage level based on "
    "the Emergency Severity Index (ESI), ranging from ESI level 1 (highest "
    "acuity) to ESI level 5 (lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

TRIAGE_PROMPT_TOOL_USE_RAG_HANDBOOK_RAG = (
    "You are a nurse with emergency and triage experience. "
    "You have been trained using the following ESI Handbook:\n\n"
    "{esi_handbook}\n\n"
    "---\n\n"
    "You have access to a tool called `search_esi_case_bank` that lets you "
    "search a bank of 60 expert-classified ESI triage cases from the ESI "
    "Implementation Handbook. Each case includes a clinical vignette, the "
    "correct ESI level, and the clinical rationale.\n\n"
    "The following PubMed literature may also be relevant to this case:\n\n"
    "{context}\n\n"
    "Before making your triage decision, consider whether consulting similar "
    "reference cases would help you classify this patient more accurately. "
    "You may call the tool zero or more times with different search filters.\n\n"
    "Once you have enough information, determine the triage level based on "
    "the Emergency Severity Index (ESI), ranging from ESI level 1 (highest "
    "acuity) to ESI level 5 (lowest acuity): "
    "1: Requires immediate lifesaving intervention. "
    "2: High-risk situation (confused, lethargic, severe pain/distress). "
    "3: Two or more diagnostic/therapeutic interventions, stable vitals. "
    "4: One diagnostic/therapeutic intervention. "
    "5: No diagnostic/therapeutic interventions needed beyond physical exam.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

MULTI_ROLE_NURSE_PROMPT = (
    "You are the triage nurse in a three-role emergency triage workflow. "
    "Review the patient presentation only and make a provisional ESI decision "
    "before seeing any external evidence.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Output ONLY a JSON object with exactly these keys:\n"
    '- "acuity": integer 1-5\n'
    '- "confidence": string, one of "high", "medium", "low"\n'
    '- "rationale": short string explaining the provisional decision\n'
)

MULTI_ROLE_READER_PROMPT = (
    "You are the evidence reader in a three-role emergency triage workflow. "
    "You do not make the final ESI decision. Review the nurse draft and the "
    "evidence pack, then summarize how the evidence should influence the "
    "provisional triage judgement.\n\n"
    "Patient summary:\n{patient_summary}\n\n"
    "Nurse draft:\n{nurse_draft}\n\n"
    "Evidence pack:\n{evidence_block}\n\n"
    "Output ONLY a JSON object with exactly these keys:\n"
    '- "evidence_signal": string, one of "supports_higher", "supports_same", '
    '"supports_lower", "mixed", "insufficient"\n'
    '- "summary": short string synthesizing the evidence for the adjudicator\n'
)

MULTI_ROLE_ADJUDICATOR_PROMPT = (
    "You are the final adjudicator in a three-role emergency triage workflow. "
    "Review the patient presentation, the nurse draft, and the evidence "
    "reader's summary. Then return the final ESI triage level.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Nurse draft:\n{nurse_draft}\n\n"
    "Reader brief:\n{reader_brief}\n\n"
    "Output ONLY a JSON object with exactly one key:\n"
    '- "acuity": integer 1-5\n'
)

TWO_ROLE_NURSE_PROMPT = (
    "You are the triage nurse in a two-role emergency triage workflow. "
    "Review the patient presentation only and make a provisional ESI decision "
    "before any tool use or evidence review.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Output ONLY a JSON object with exactly these keys:\n"
    '- "acuity": integer 1-5\n'
    '- "confidence": string, one of "high", "medium", "low"\n'
    '- "rationale": short string explaining the provisional decision\n'
)

TWO_ROLE_CRITIC_CASE_BANK_PROMPT = (
    "You are the critic in a two-role emergency triage workflow. "
    "Review the patient's presentation and the nurse draft. "
    "You have access to the `search_esi_case_bank` tool with expert-classified "
    "ESI examples. Use it if it helps you challenge or refine the nurse's "
    "decision, then return the final ESI level.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Nurse draft:\n{nurse_draft}\n\n"
    "Output ONLY a JSON object with exactly one key:\n"
    '- "acuity": integer 1-5\n'
)

TWO_ROLE_CRITIC_CASE_BANK_PMC_PROMPT = (
    "You are the critic in a two-role emergency triage workflow. "
    "Review the patient's presentation and the nurse draft. "
    "You always have access to the `search_esi_case_bank` tool. "
    "You may also have access to `search_pmc_articles` when the workflow "
    "decides the case needs more evidence. Use whatever tools are available "
    "to challenge or refine the nurse's decision, then return the final ESI level.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Nurse draft:\n{nurse_draft}\n\n"
    "Output ONLY a JSON object with exactly one key:\n"
    '- "acuity": integer 1-5\n'
)

THREE_ROLE_RERANK_NURSE_PROMPT = TWO_ROLE_NURSE_PROMPT

THREE_ROLE_RERANK_SELECTION_PROMPT = (
    "You are the reranker in a three-role emergency triage workflow. "
    "You do not assign triage. Review the patient summary and the PubMed "
    "article snippets below, then select the article indices that are most "
    "useful for determining the correct ESI level.\n\n"
    "Patient summary:\n{patient_summary}\n\n"
    "Articles:\n{articles_block}\n\n"
    "Select between 1 and 3 article indices. Output ONLY a JSON object with "
    'keys "selected_indices" (a non-empty list of 0-based article indices) '
    'and "selection_rationale" (short string).'
)

THREE_ROLE_RERANK_CRITIC_PROMPT = (
    "You are the critic in a three-role emergency triage workflow. "
    "Review the patient's presentation, the nurse draft, and the reranked PMC "
    "evidence. You also have access to the `search_esi_case_bank` tool with "
    "expert-classified ESI examples. Use the available evidence to challenge "
    "or refine the nurse's decision, then return the final ESI level.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Nurse draft:\n{nurse_draft}\n\n"
    "Reranked PMC evidence:\n{reranked_evidence}\n\n"
    "Output ONLY a JSON object with exactly one key:\n"
    '- "acuity": integer 1-5\n'
)

# ── ESI decision-tree block ─────────────────────────────────────────────
# Structured algorithmic representation of the ESI v4 triage algorithm.
# Use alongside or instead of the flat prose ESI descriptions to encourage
# step-by-step decision procedure rather than keyword pattern matching.

ESI_DECISION_TREE_BLOCK = (
    "Apply the ESI triage algorithm as a sequential decision tree:\n\n"
    "Step A — Immediate lifesaving intervention needed?\n"
    "  Examples: intubation, surgical airway, emergency medications "
    "(naloxone, D50, epinephrine), fluid resuscitation for hemodynamic "
    "instability, electrical therapy (cardioversion, defibrillation).\n"
    "  YES → ESI-1. STOP.\n"
    "  NO  → go to Step B.\n\n"
    "Step B — High-risk situation?\n"
    "  Patient is confused, lethargic, or disoriented? OR severe "
    "pain/distress (≥7/10, or objective distress signs)?\n"
    "  YES → ESI-2. STOP.\n"
    "  NO  → go to Step C.\n\n"
    "Step C — Count expected resource CATEGORIES the patient will need.\n"
    "  Each category counts as ONE resource regardless of how many "
    "individual tests within it:\n"
    "    • Labs (all blood work = 1 resource)\n"
    "    • ECG\n"
    "    • Imaging (X-ray, CT, MRI, ultrasound = 1 resource)\n"
    "    • IV fluids or IV medications\n"
    "    • Specialty consultation\n"
    "    • Simple procedure (laceration repair, Foley catheter, I&D)\n"
    "  The following do NOT count as resources: history/physical, "
    "point-of-care testing, saline/heparin lock, oral medications, "
    "tetanus immunization, prescription refills, simple wound care, "
    "crutches/splints/slings.\n"
    "  0 resources → ESI-5. STOP.\n"
    "  1 resource  → ESI-4. STOP.\n"
    "  ≥2 resources → go to Step D.\n\n"
    "Step D — Vital-sign danger zone check.\n"
    "  For adults (>18y): HR >100 or RR >20 or SpO₂ <92%.\n"
    "  For pediatric patients, use age-adjusted thresholds:\n"
    "    <1mo: HR>190, RR>60 | 1-12mo: HR>180, RR>55 | "
    "1-3y: HR>140, RR>40 | 3-5y: HR>120, RR>35 | "
    "5-12y: HR>120, RR>30 | 12-18y: HR>100, RR>20.\n"
    "  ANY vital in danger zone → consider uptriage to ESI-2.\n"
    "  All vitals stable → ESI-3."
)

TRIAGE_PROMPT_TOOL_USE_DECISION_TREE = (
    "You are a nurse with emergency and triage experience. "
    "You have access to a tool called `search_esi_case_bank` that lets you "
    "search a bank of 60 expert-classified ESI triage cases from the ESI "
    "Implementation Handbook. Each case includes a clinical vignette, the "
    "correct ESI level, and the clinical rationale.\n\n"
    "Before making your triage decision, consider whether consulting similar "
    "reference cases would help you classify this patient more accurately. "
    "You may call the tool zero or more times with different search filters.\n\n"
    "Once you have enough information, determine the triage level by "
    "following this algorithm:\n\n"
    + ESI_DECISION_TREE_BLOCK + "\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}"
)

# ── Boundary review prompt ──────────────────────────────────────────────
# Targeted second pass for ESI-2/3 predictions. Asks the model to explicitly
# count resources (Decision Point C) and check vitals (Decision Point D).

ESI_BOUNDARY_REVIEW_PROMPT = (
    "You predicted ESI-{predicted_esi} for this patient. Before finalizing, "
    "review your decision by explicitly working through Steps B–D of the "
    "ESI algorithm.\n\n"
    "Patient:\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Step B: Is this patient high-risk (confused, lethargic, disoriented, "
    "severe pain ≥7/10, or objective distress)? If YES → ESI-2.\n"
    "Step C: If NOT high-risk, list each expected resource category:\n"
    "  • Labs (blood work)\n"
    "  • Imaging (X-ray/CT/MRI/ultrasound)\n"
    "  • ECG\n"
    "  • IV fluids or IV medications\n"
    "  • Specialty consultation\n"
    "  • Procedure (laceration repair, Foley, I&D)\n"
    "Count the total categories.\n"
    "  1 resource → ESI-4. 0 resources → ESI-5.\n"
    "  ≥2 resources → go to Step D.\n"
    "Step D: Vital-sign danger zone (adults >18y): HR >100, RR >20, "
    "SpO₂ <92%. Any in danger zone → consider ESI-2. "
    "All stable → ESI-3.\n\n"
    "Output ONLY a JSON object: {{\"acuity\": <integer 1-5>}}"
)

# ── Vitals danger zone prompt ───────────────────────────────────────────
# LLM-based structured check against the ESI handbook's age-stratified
# vital-sign danger zone table (Decision Point D).

VITALS_DANGER_ZONE_PROMPT = (
    "You are checking whether this patient's vital signs fall in the ESI "
    "danger zone (Decision Point D of the ESI v4 algorithm).\n\n"
    "Danger zone thresholds by age:\n"
    "  Age          | HR >  | RR >\n"
    "  < 1 month    | 190   | 60\n"
    "  1–12 months  | 180   | 55\n"
    "  1–3 years    | 140   | 40\n"
    "  3–5 years    | 120   | 35\n"
    "  5–12 years   | 120   | 30\n"
    "  12–18 years  | 100   | 20\n"
    "  > 18 years   | 100   | 20\n"
    "  Any age: SpO₂ < 92%\n\n"
    "Pediatric fever considerations:\n"
    "  1–28 days: ESI-2 if T > 38°C (100.4°F)\n"
    "  1–3 months: consider ESI-2 if T > 38°C\n"
    "  ≥3 months: consider ESI-2 or 3 if T > 39°C or < 36°C, "
    "incomplete immunizations, or no obvious source\n\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Determine the patient's age group from patient info, then check "
    "each vital against the appropriate threshold.\n\n"
    "Output ONLY a JSON object with these keys:\n"
    "- \"danger_zone\": boolean (true if ANY vital exceeds threshold)\n"
    "- \"flags\": list of strings describing each abnormal vital "
    "(e.g. [\"HR 120 > 100 (adult)\", \"SpO2 89% < 92%\"]). "
    "Empty list if no vitals in danger zone."
)

# ── Batch reranking prompt ────────────────────────────────────────────────
# Used by the LLM reranker step to score all retrieved articles in one call.

BATCH_RERANK_PROMPT = (
    "You are a medical literature relevance assessor. A patient is being "
    "triaged in the emergency department. Rate how useful each article "
    "excerpt below is for determining the correct ESI triage level.\n\n"
    "Patient summary:\n{patient_summary}\n\n"
    "Articles:\n{articles_block}\n\n"
    "For each article, output a relevance score from 1 (not useful) to 5 "
    "(very useful). Output ONLY a JSON array of objects with keys "
    '"article_index" (int, 0-based) and "score" (int, 1-5). '
    "Example: "
    '[{{"article_index": 0, "score": 3}}, {{"article_index": 1, "score": 5}}]'
)

# ── Uncertainty assessment prompt ────────────────────────────────────────
# Used by the uncertainty-gated two-pass mode to decide if retrieval is needed.

UNCERTAINTY_ASSESSMENT_PROMPT = (
    "You are a nurse with emergency and triage experience. "
    "Given the patient presentation below, predict the ESI triage level "
    "and indicate your confidence.\n\n"
    "History of present illness: {hpi}\n"
    "Patient info: {patient_info}\n"
    "Initial vitals: {initial_vitals}\n\n"
    "Be honest about uncertainty. Mark yourself as \"uncertain\" if:\n"
    "- The case is ambiguous between two ESI levels\n"
    "- Key clinical information seems missing\n"
    "- The presentation has atypical features\n"
    "- You are unsure about resource requirements"
)

UNCERTAINTY_ASSESSMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "acuity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "confidence": {"type": "string", "enum": ["confident", "uncertain"]},
    },
    "required": ["acuity", "confidence"],
}

# ── Prompt template registry ─────────────────────────────────────────────
# Each entry maps to the 4 prompt variants needed by the pipeline.

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "default": {
        "rag": TRIAGE_PROMPT_WITH_RAG,
        "llm": TRIAGE_PROMPT_LLM_ONLY,
        "handbook_rag": TRIAGE_PROMPT_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_HANDBOOK_ONLY,
    },
    "gaber": {
        "rag": TRIAGE_PROMPT_GABER_RAG,
        "llm": TRIAGE_PROMPT_GABER_LLM_ONLY,
        "handbook_rag": TRIAGE_PROMPT_GABER_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_GABER_HANDBOOK_ONLY,
    },
    "cot_private": {
        "rag": TRIAGE_PROMPT_WITH_RAG,
        "llm": TRIAGE_PROMPT_COT_PRIVATE,
        "handbook_rag": TRIAGE_PROMPT_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_HANDBOOK_ONLY,
    },
    "fewshot": {
        "rag": TRIAGE_PROMPT_WITH_RAG,
        "llm": TRIAGE_PROMPT_FEWSHOT,
        "handbook_rag": TRIAGE_PROMPT_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_HANDBOOK_ONLY,
    },
    "fewshot_cot_private": {
        "rag": TRIAGE_PROMPT_WITH_RAG,
        "llm": TRIAGE_PROMPT_FEWSHOT_COT_PRIVATE,
        "handbook_rag": TRIAGE_PROMPT_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_HANDBOOK_ONLY,
    },
    "tool_use": {
        "rag": TRIAGE_PROMPT_TOOL_USE,
        "llm": TRIAGE_PROMPT_TOOL_USE,
        "handbook_rag": TRIAGE_PROMPT_TOOL_USE,
        "handbook_only": TRIAGE_PROMPT_TOOL_USE,
    },
    "tool_use_pmc": {
        "rag": TRIAGE_PROMPT_TOOL_USE_PMC,
        "llm": TRIAGE_PROMPT_TOOL_USE_PMC,
        "handbook_rag": TRIAGE_PROMPT_TOOL_USE_PMC,
        "handbook_only": TRIAGE_PROMPT_TOOL_USE_PMC,
    },
    "tool_use_dual": {
        "rag": TRIAGE_PROMPT_TOOL_USE_DUAL,
        "llm": TRIAGE_PROMPT_TOOL_USE_DUAL,
        "handbook_rag": TRIAGE_PROMPT_TOOL_USE_DUAL,
        "handbook_only": TRIAGE_PROMPT_TOOL_USE_DUAL,
    },
    "tool_use_decision_tree": {
        "rag": TRIAGE_PROMPT_TOOL_USE_DECISION_TREE,
        "llm": TRIAGE_PROMPT_TOOL_USE_DECISION_TREE,
        "handbook_rag": TRIAGE_PROMPT_TOOL_USE_DECISION_TREE,
        "handbook_only": TRIAGE_PROMPT_TOOL_USE_DECISION_TREE,
    },
    "fewshot_rag": {
        "rag": TRIAGE_PROMPT_FEWSHOT_RAG,
        "llm": TRIAGE_PROMPT_FEWSHOT,
        "handbook_rag": TRIAGE_PROMPT_FEWSHOT_RAG_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_FEWSHOT_HANDBOOK_ONLY,
    },
    "tool_use_rag": {
        "rag": TRIAGE_PROMPT_TOOL_USE_RAG,
        "llm": TRIAGE_PROMPT_TOOL_USE,
        "handbook_rag": TRIAGE_PROMPT_TOOL_USE_RAG_HANDBOOK_RAG,
        "handbook_only": TRIAGE_PROMPT_TOOL_USE,
    },
    "multi_role": {
        "rag": MULTI_ROLE_ADJUDICATOR_PROMPT,
        "llm": MULTI_ROLE_ADJUDICATOR_PROMPT,
        "handbook_rag": MULTI_ROLE_ADJUDICATOR_PROMPT,
        "handbook_only": MULTI_ROLE_ADJUDICATOR_PROMPT,
        "nurse": MULTI_ROLE_NURSE_PROMPT,
        "reader": MULTI_ROLE_READER_PROMPT,
        "adjudicator": MULTI_ROLE_ADJUDICATOR_PROMPT,
    },
    "two_role_case_bank": {
        "rag": TWO_ROLE_CRITIC_CASE_BANK_PROMPT,
        "llm": TWO_ROLE_CRITIC_CASE_BANK_PROMPT,
        "handbook_rag": TWO_ROLE_CRITIC_CASE_BANK_PROMPT,
        "handbook_only": TWO_ROLE_CRITIC_CASE_BANK_PROMPT,
        "nurse": TWO_ROLE_NURSE_PROMPT,
        "critic": TWO_ROLE_CRITIC_CASE_BANK_PROMPT,
    },
    "two_role_case_bank_pmc_conditional": {
        "rag": TWO_ROLE_CRITIC_CASE_BANK_PMC_PROMPT,
        "llm": TWO_ROLE_CRITIC_CASE_BANK_PMC_PROMPT,
        "handbook_rag": TWO_ROLE_CRITIC_CASE_BANK_PMC_PROMPT,
        "handbook_only": TWO_ROLE_CRITIC_CASE_BANK_PMC_PROMPT,
        "nurse": TWO_ROLE_NURSE_PROMPT,
        "critic": TWO_ROLE_CRITIC_CASE_BANK_PMC_PROMPT,
    },
    "three_role_rerank_critic": {
        "rag": THREE_ROLE_RERANK_CRITIC_PROMPT,
        "llm": THREE_ROLE_RERANK_CRITIC_PROMPT,
        "handbook_rag": THREE_ROLE_RERANK_CRITIC_PROMPT,
        "handbook_only": THREE_ROLE_RERANK_CRITIC_PROMPT,
        "nurse": THREE_ROLE_RERANK_NURSE_PROMPT,
        "reranker": THREE_ROLE_RERANK_SELECTION_PROMPT,
        "critic": THREE_ROLE_RERANK_CRITIC_PROMPT,
    },
}


def get_prompt_template(name: str) -> dict[str, str]:
    """Look up a prompt template set by name. Raises KeyError if not found."""
    if name not in PROMPT_TEMPLATES:
        raise KeyError(
            f"Unknown prompt template '{name}'. "
            f"Available: {sorted(PROMPT_TEMPLATES.keys())}"
        )
    return PROMPT_TEMPLATES[name]

# ── Postprocessing ───────────────────────────────────────────────────────

def parse_triage(raw: str | None) -> int | None:
    """Parse raw model JSON into an ESI int. Returns None on any failure."""
    if not raw or not raw.strip():
        print("WARNING parse_triage: empty/None response")
        return None
    try:
        return TriagePrediction.model_validate_json(raw).acuity
    except Exception as e:
        print(f"WARNING parse_triage: {type(e).__name__}: {e} | raw={raw[:120]!r}")
        return None


# ── Pricing helpers ──────────────────────────────────────────────────────
# Fallback per-token prices (USD) from cloud.google.com/vertex-ai/generative-ai/pricing
# GA tier, text input/output/thinking. Updated 2026-03-03.
FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000, "thinking": 2.50 / 1_000_000},
    "gemini-2.5-pro":   {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000, "thinking": 10.00 / 1_000_000},
    "gemini-2.0-flash":       {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000, "thinking": 0.60 / 1_000_000},
    "gemini-3-flash-preview": {"input": 0.50 / 1_000_000, "output": 3.00 / 1_000_000, "thinking": 3.00 / 1_000_000},
    # Anthropic (api.anthropic.com pricing as of 2026-03)
    "claude-haiku-4-5":           {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000, "thinking": 5.00 / 1_000_000},
    "claude-3-5-haiku-20241022":  {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000, "thinking": 4.00 / 1_000_000},
    "claude-3-5-sonnet-20241022": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000, "thinking": 15.00 / 1_000_000},
    "claude-sonnet-4-20250514":   {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000, "thinking": 15.00 / 1_000_000},
}


def fetch_vertex_pricing(model_id: str) -> tuple[dict | None, str]:
    """Query Cloud Billing Catalog for per-token USD pricing, with static fallback.

    Returns (pricing_dict, pricing_source) where pricing_dict is
    {"input": price, "output": price, "thinking": price} or None.
    """
    import src.config as config

    # Try Billing Catalog first
    try:
        from google.cloud import billing_v1

        client = billing_v1.CloudCatalogClient(credentials=config.CREDENTIALS)
        VERTEX_SERVICE = "services/C7E2-9256-1C43"
        model_tokens = set(model_id.lower().replace("-", " ").split())

        # Collect (per_token, is_ga) candidates per pricing key
        candidates: dict[str, list[tuple[float, bool]]] = {
            "input": [], "output": [], "thinking": [],
        }
        excluded_terms = [
            "batch", "audio", "video", "long", "lite", "priority",
            "caching", "cache", "image", "tuning", "flex", "live",
        ]

        for sku in client.list_skus(parent=VERTEX_SERVICE):
            desc = sku.description.lower()
            if not all(t in desc for t in model_tokens):
                continue
            if any(term in desc for term in excluded_terms):
                continue

            if not sku.pricing_info:
                continue
            expr = sku.pricing_info[0].pricing_expression
            if not expr.tiered_rates:
                continue
            rate = expr.tiered_rates[0]
            raw_price = rate.unit_price.units + rate.unit_price.nanos / 1e9
            unit = expr.usage_unit.lower()
            if unit in ("1", "token", "character", "count"):
                per_token = raw_price
            elif unit in ("1k", "1000"):
                per_token = raw_price / 1_000
            elif unit in ("1m", "1000000"):
                per_token = raw_price / 1_000_000
            else:
                continue

            is_ga = " ga " in f" {desc} "  # word-boundary match

            # Classify: thinking output vs standard output vs input
            if "thinking" in desc and "output" in desc:
                candidates["thinking"].append((per_token, is_ga))
            elif "text input" in desc or ("input" in desc and "text" in desc):
                candidates["input"].append((per_token, is_ga))
            elif "text output" in desc or ("output" in desc and "text" in desc):
                candidates["output"].append((per_token, is_ga))

        # Pick best candidate per key: prefer GA, then take first match
        found: dict[str, float | None] = {"input": None, "output": None, "thinking": None}
        for key, cands in candidates.items():
            if not cands:
                continue
            ga_cands = [p for p, is_ga in cands if is_ga]
            found[key] = ga_cands[0] if ga_cands else cands[0][0]

        if found["input"] is not None or found["output"] is not None:
            print(f"Pricing fetched from Cloud Billing Catalog for '{model_id}': {found}")
            return found, "billing_catalog"
    except ImportError:
        pass
    except Exception as e:
        print(f"WARNING: Cloud Billing Catalog lookup failed: {e}")

    # Fallback to static pricing table
    if model_id in FALLBACK_PRICING:
        print(f"Using fallback pricing for '{model_id}': {FALLBACK_PRICING[model_id]}")
        return FALLBACK_PRICING[model_id], "fallback_static"

    print(f"WARNING: No pricing available for '{model_id}' -- cost will not be tracked.")
    return None, "unavailable"


# ── Context formatting ───────────────────────────────────────────────────

def build_context_block(articles: pd.DataFrame, context_chars: int) -> str:
    """Format retrieved articles into a context string for the triage prompt.

    Prefers ``context_text`` (cleaned body text from BM25 indexing) when
    available; falls back to raw ``article_text`` otherwise.

    Args:
        articles: DataFrame with columns pmc_id/pmid/article_text
                  (+ optional context_text, rank, score, score_type).
        context_chars: Max characters per article snippet.

    Returns:
        Formatted context block, or "No relevant articles found." if empty.
    """
    if articles.empty:
        return "No relevant articles found."

    has_context_text = "context_text" in articles.columns

    context_parts = []
    for i, art_row in articles.iterrows():
        text = prepare_article_excerpt(
            art_row.get("article_text"),
            context_chars,
            context_text=(
                art_row.get("context_text")
                if has_context_text and pd.notna(art_row.get("context_text"))
                else None
            ),
        )
        context_parts.append(f"--- Article {i + 1} (PMID {art_row['pmid']}) ---\n{text}")
    return "\n\n".join(context_parts) if context_parts else "No relevant articles found."
