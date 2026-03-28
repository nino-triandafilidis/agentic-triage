"""Pydantic schemas for structured LLM outputs."""
from typing import Literal

from pydantic import BaseModel, Field


class TriagePrediction(BaseModel):
    acuity: int = Field(..., ge=1, le=5,
        description="ESI triage level (1=highest acuity, 5=lowest)")


class NurseDraft(BaseModel):
    acuity: int = Field(..., ge=1, le=5,
        description="Provisional ESI triage level before evidence review.")
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Qualitative confidence in the provisional decision: high, medium, or low.",
    )
    rationale: str = Field(
        ...,
        description="Brief clinical rationale for the provisional ESI assignment.",
    )


class ReaderBrief(BaseModel):
    evidence_signal: Literal[
        "supports_higher",
        "supports_same",
        "supports_lower",
        "mixed",
        "insufficient",
    ] = Field(
        ...,
        description=(
            "How the evidence affects the provisional decision: supports_higher, "
            "supports_same, supports_lower, mixed, or insufficient."
        ),
    )
    summary: str = Field(
        ...,
        description="Concise evidence summary for the adjudicator.",
    )


class RerankerSelection(BaseModel):
    selected_indices: list[int] = Field(
        ...,
        description="0-based indices of the selected PMC articles, ordered by usefulness.",
        min_length=1,
    )
    selection_rationale: str = Field(
        ...,
        description="Short explanation for why the selected articles are the most useful.",
    )


class VitalsDangerZone(BaseModel):
    danger_zone: bool = Field(
        ...,
        description="True if ANY vital sign exceeds the age-adjusted ESI danger zone threshold.",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="List of abnormal vital descriptions, e.g. ['HR 120 > 100 (adult)', 'SpO2 89% < 92%']. Empty if no vitals in danger zone.",
    )
