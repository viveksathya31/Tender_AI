"""
Data models for Phase 2 — Criteria Extraction.

A tender document yields a CriteriaSet containing multiple Criterion objects.
Each Criterion is typed, mandatory/optional, and carries enough metadata
for Phase 3 to match it against bidder evidence.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CriterionType(str, Enum):
    FINANCIAL = "financial"         # turnover, net worth, bid capacity
    TECHNICAL = "technical"         # similar works, equipment, manpower
    COMPLIANCE = "compliance"       # GST, PAN, legal registrations
    CERTIFICATION = "certification" # ISO, BIS, licences
    EXPERIENCE = "experience"       # years in business, past projects
    OTHER = "other"


class CriterionStatus(str, Enum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"
    UNCLEAR = "unclear"             # ambiguous language — flagged for human review


class Criterion(BaseModel):
    id: str                         # e.g. "C001", "C002"
    type: CriterionType
    status: CriterionStatus
    description: str                # human-readable full description
    # Structured fields for matching (populated when extractable)
    threshold_value: Optional[float] = None     # numeric threshold if applicable
    threshold_unit: Optional[str] = None        # "INR", "years", "projects", etc.
    threshold_operator: Optional[str] = None    # ">=", "<=", "=", ">"
    verification_docs: list[str] = Field(default_factory=list)
    # e.g. ["audited balance sheet", "CA certificate"]
    source_page: Optional[int] = None
    source_text: str = ""           # raw excerpt from tender that produced this criterion
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    # confidence that this criterion was correctly extracted
    notes: str = ""                 # extractor notes, ambiguity flags


class CriteriaSet(BaseModel):
    tender_filename: str
    total_criteria: int
    mandatory_count: int
    optional_count: int
    unclear_count: int
    criteria: list[Criterion]
    extraction_model: str           # which Claude model was used
    extraction_warnings: list[str] = Field(default_factory=list)