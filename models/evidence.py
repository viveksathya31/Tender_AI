"""
Data models for Phase 3 — Bidder Evidence Extraction.

For each (bidder, criterion) pair, we produce an EvidenceResult.
All results for one bidder are collected into a BidderEvidence object.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    FOUND = "found"               # value clearly found in documents
    NOT_FOUND = "not_found"       # searched but no relevant content found
    PARTIAL = "partial"           # some evidence found but incomplete
    UNREADABLE = "unreadable"     # document too degraded to extract


class EvidenceResult(BaseModel):
    criterion_id: str             # links back to Criterion.id from Phase 2
    criterion_description: str
    status: ExtractionStatus

    # What was found
    found_value: Optional[str] = None        # raw extracted value as string
    found_value_numeric: Optional[float] = None  # parsed numeric if applicable
    found_unit: Optional[str] = None

    # Source traceability — mandatory for audit
    source_doc: str = ""          # filename of the document this came from
    source_page: Optional[int] = None
    source_text: str = ""         # exact snippet from the document

    # Confidence
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    notes: str = ""               # extractor notes, ambiguity, format issues


class BidderEvidence(BaseModel):
    bidder_id: str
    documents_processed: list[str]   # filenames of all docs submitted by this bidder
    total_criteria: int
    evidence: list[EvidenceResult]
    extraction_model: str
    extraction_warnings: list[str] = Field(default_factory=list)
    overall_extraction_confidence: float = Field(ge=0.0, le=1.0)