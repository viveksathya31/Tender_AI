"""
Data models for Phase 4 (Verdict Engine) and Phase 5 (Report + Audit).
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class VerdictLabel(str, Enum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_REVIEW = "needs_review"


class CriterionVerdict(BaseModel):
    criterion_id: str
    criterion_description: str
    criterion_type: str
    is_mandatory: bool

    verdict: VerdictLabel
    reason: str                      # human-readable explanation — always populated
    confidence: float = Field(ge=0.0, le=1.0)

    # Evidence references — for full auditability
    evidence_status: str             # found / not_found / partial / unreadable
    evidence_value: Optional[str] = None
    evidence_source_doc: str = ""
    evidence_source_page: Optional[int] = None
    evidence_source_text: str = ""


class BidderVerdict(BaseModel):
    bidder_id: str
    overall_verdict: VerdictLabel
    overall_reason: str

    criterion_verdicts: list[CriterionVerdict]
    mandatory_passed: int
    mandatory_failed: int
    mandatory_review: int
    optional_passed: int

    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class ConsolidatedReport(BaseModel):
    tender_filename: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    total_bidders: int
    eligible_count: int
    not_eligible_count: int
    needs_review_count: int
    criteria_summary: list[dict]     # per-criterion pass/fail counts
    bidder_verdicts: list[BidderVerdict]
    audit_log: list[dict] = Field(default_factory=list)