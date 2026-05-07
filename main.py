"""
Tender Evaluation Platform — Complete API
Phases 1-5: Ingest → Extract Criteria → Extract Evidence → Verdict → Report
"""

import hashlib
import json
import logging
import os
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ingestion.document_processor import DocumentProcessor
from models.document import DocType, ExtractedDocument
from models.criteria import CriteriaSet
from models.evidence import BidderEvidence
from models.verdict import BidderVerdict, ConsolidatedReport, VerdictLabel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tender Evaluation Platform",
    description="End-to-end AI-powered tender evaluation: ingest → criteria → evidence → verdict → report",
    version="1.0.0",
)

# -- Services (lazy-loaded so missing API key doesn't crash startup) ----------

processor = DocumentProcessor()

def _get_criteria_extractor():
    from ingestion.criteria_extractor import CriteriaExtractor
    return CriteriaExtractor()

def _get_evidence_extractor():
    from ingestion.evidence_extractor import EvidenceExtractor
    return EvidenceExtractor()

def _get_verdict_engine():
    from ingestion.verdict_engine import VerdictEngine
    return VerdictEngine()

def _get_report_generator():
    from ingestion.report_generator import ReportGenerator
    return ReportGenerator()

# -- In-memory stores (replace with PostgreSQL in production) -----------------

_document_store: dict[str, ExtractedDocument] = {}
_bidder_docs_store: dict[str, list[str]] = {}       # bidder_id -> [filenames]
_criteria_store: dict[str, CriteriaSet] = {}        # tender_filename -> CriteriaSet
_evidence_store: dict[str, BidderEvidence] = {}     # bidder_id -> BidderEvidence
_verdict_store: dict[str, BidderVerdict] = {}       # bidder_id -> BidderVerdict
_report_store: dict[str, ConsolidatedReport] = {}   # tender_filename -> report
_audit_log: list[dict] = []                         # append-only

# =============================================================================
# Response schemas
# =============================================================================

class IngestResponse(BaseModel):
    filename: str
    file_format: str
    doc_type: str
    total_pages: int
    overall_confidence: float
    ocr_engine_used: str | None
    warnings: list[str]
    full_text_preview: str
    processing_time_ms: float

class BatchIngestResponse(BaseModel):
    total_files: int
    succeeded: int
    failed: int
    results: list[IngestResponse | dict]

class CriteriaResponse(BaseModel):
    tender_filename: str
    total_criteria: int
    mandatory_count: int
    optional_count: int
    unclear_count: int
    extraction_model: str
    extraction_warnings: list[str]
    criteria: list[dict]
    processing_time_ms: float

class EvidenceResponse(BaseModel):
    bidder_id: str
    documents_processed: list[str]
    total_criteria: int
    overall_extraction_confidence: float
    extraction_model: str
    extraction_warnings: list[str]
    evidence: list[dict]
    processing_time_ms: float

class VerdictResponse(BaseModel):
    bidder_id: str
    overall_verdict: str
    overall_reason: str
    mandatory_passed: int
    mandatory_failed: int
    mandatory_review: int
    optional_passed: int
    criterion_verdicts: list[dict]
    processing_time_ms: float

class ReportResponse(BaseModel):
    tender_filename: str
    generated_at: str
    total_bidders: int
    eligible_count: int
    not_eligible_count: int
    needs_review_count: int
    criteria_summary: list[dict]
    bidder_verdicts: list[dict]
    audit_entries: int
    processing_time_ms: float

class HumanOverrideRequest(BaseModel):
    bidder_id: str
    criterion_id: str
    new_verdict: str    # "eligible" | "not_eligible" | "needs_review"
    officer_id: str
    reason: str

# =============================================================================
# Health
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "phases": ["ingest", "criteria", "evidence", "verdict", "report"],
        "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "documents_in_store": len(_document_store),
        "bidders_in_store": len(_bidder_docs_store),
        "reports_generated": len(_report_store),
    }

# =============================================================================
# Phase 1 — Document Ingestion
# =============================================================================

@app.post("/ingest/tender", response_model=IngestResponse, tags=["Phase 1 - Ingest"])
async def ingest_tender(file: UploadFile = File(...)):
    """Upload and process a tender document (PDF, DOCX, or image)."""
    return await _process_upload(file, DocType.TENDER)

@app.post("/ingest/bidder", response_model=IngestResponse, tags=["Phase 1 - Ingest"])
async def ingest_bidder(
    file: UploadFile = File(...),
    bidder_id: Annotated[str, Form()] = "unknown",
):
    """Upload one bidder submission. Use same bidder_id for all docs from one bidder."""
    result = await _process_upload(file, DocType.BIDDER_SUBMISSION)
    _bidder_docs_store.setdefault(bidder_id, [])
    if file.filename not in _bidder_docs_store[bidder_id]:
        _bidder_docs_store[bidder_id].append(file.filename)
    _audit_log.append(_get_report_generator().create_audit_entry(
        "document_ingested", "system",
        {"bidder_id": bidder_id, "filename": file.filename,
         "confidence": result.overall_confidence, "warnings": result.warnings}
    ))
    return result

@app.post("/ingest/batch", response_model=BatchIngestResponse, tags=["Phase 1 - Ingest"])
async def ingest_batch(files: list[UploadFile] = File(...)):
    """Upload multiple bidder documents at once."""
    results, succeeded, failed = [], 0, 0
    for upload in files:
        try:
            results.append(await _process_upload(upload, DocType.BIDDER_SUBMISSION))
            succeeded += 1
        except Exception as e:
            results.append({"filename": upload.filename, "error": str(e)})
            failed += 1
    return BatchIngestResponse(total_files=len(files), succeeded=succeeded, failed=failed, results=results)

@app.get("/documents/{filename}", tags=["Phase 1 - Ingest"])
async def get_document(filename: str):
    doc = _document_store.get(filename)
    if not doc:
        raise HTTPException(404, f"Document '{filename}' not found")
    return doc.model_dump()

# =============================================================================
# Phase 2 — Criteria Extraction
# =============================================================================

@app.post("/criteria/extract", response_model=CriteriaResponse, tags=["Phase 2 - Criteria"])
async def extract_criteria(filename: str):
    """Extract eligibility criteria from an ingested tender document."""
    _require_api_key()
    doc = _get_doc(filename, DocType.TENDER)
    t0 = time.perf_counter()
    try:
        cs = _get_criteria_extractor().extract(doc)
    except Exception as e:
        logger.exception("Criteria extraction failed")
        raise HTTPException(500, f"Extraction error: {e}")
    _criteria_store[filename] = cs
    _audit_log.append(_get_report_generator().create_audit_entry(
        "criteria_extracted", cs.extraction_model,
        {"tender": filename, "total_criteria": cs.total_criteria,
         "mandatory": cs.mandatory_count, "warnings": cs.extraction_warnings}
    ))
    return _criteria_response(cs, time.perf_counter() - t0)

@app.get("/criteria/{filename}", response_model=CriteriaResponse, tags=["Phase 2 - Criteria"])
async def get_criteria(filename: str):
    cs = _criteria_store.get(filename)
    if not cs:
        raise HTTPException(404, f"No criteria for '{filename}'. Run /criteria/extract first.")
    return _criteria_response(cs, 0)

# =============================================================================
# Phase 3 — Evidence Extraction
# =============================================================================

@app.post("/evidence/extract", response_model=EvidenceResponse, tags=["Phase 3 - Evidence"])
async def extract_evidence(bidder_id: str, tender_filename: str):
    """Extract evidence for a bidder against a tender's criteria."""
    _require_api_key()
    cs = _criteria_store.get(tender_filename)
    if not cs:
        raise HTTPException(404, f"No criteria for '{tender_filename}'. Run /criteria/extract first.")
    filenames = _bidder_docs_store.get(bidder_id, [])
    if not filenames:
        raise HTTPException(404, f"No docs for bidder '{bidder_id}'. Upload via /ingest/bidder first.")
    bidder_docs = [_document_store[f] for f in filenames if f in _document_store]
    t0 = time.perf_counter()
    try:
        ev = _get_evidence_extractor().extract(bidder_id, bidder_docs, cs)
    except Exception as e:
        logger.exception("Evidence extraction failed for bidder '%s'", bidder_id)
        raise HTTPException(500, f"Extraction error: {e}")
    _evidence_store[bidder_id] = ev
    _audit_log.append(_get_report_generator().create_audit_entry(
        "evidence_extracted", ev.extraction_model,
        {"bidder_id": bidder_id, "tender": tender_filename,
         "docs": ev.documents_processed, "overall_confidence": ev.overall_extraction_confidence,
         "warnings": ev.extraction_warnings}
    ))
    return _evidence_response(ev, time.perf_counter() - t0)

@app.get("/evidence/{bidder_id}", response_model=EvidenceResponse, tags=["Phase 3 - Evidence"])
async def get_evidence(bidder_id: str):
    ev = _evidence_store.get(bidder_id)
    if not ev:
        raise HTTPException(404, f"No evidence for '{bidder_id}'. Run /evidence/extract first.")
    return _evidence_response(ev, 0)

# =============================================================================
# Phase 4 — Verdict Engine
# =============================================================================

@app.post("/verdict/{bidder_id}", response_model=VerdictResponse, tags=["Phase 4 - Verdict"])
async def compute_verdict(bidder_id: str, tender_filename: str):
    """
    Compute Pass/Fail/Review verdict for a bidder against all criteria.
    Evidence must already be extracted via /evidence/extract.
    """
    cs = _criteria_store.get(tender_filename)
    if not cs:
        raise HTTPException(404, f"No criteria for '{tender_filename}'.")
    ev = _evidence_store.get(bidder_id)
    if not ev:
        raise HTTPException(404, f"No evidence for '{bidder_id}'. Run /evidence/extract first.")
    t0 = time.perf_counter()
    engine = _get_verdict_engine()
    bv = engine.evaluate_bidder(ev, cs)
    _verdict_store[bidder_id] = bv
    _audit_log.append(_get_report_generator().log_verdict(bv, ev.extraction_model))
    return _verdict_response(bv, time.perf_counter() - t0)

@app.get("/verdict/{bidder_id}", response_model=VerdictResponse, tags=["Phase 4 - Verdict"])
async def get_verdict(bidder_id: str):
    bv = _verdict_store.get(bidder_id)
    if not bv:
        raise HTTPException(404, f"No verdict for '{bidder_id}'. Run /verdict first.")
    return _verdict_response(bv, 0)

@app.post("/verdict/override", tags=["Phase 4 - Verdict"])
async def human_override(req: HumanOverrideRequest):
    """
    Officer manually overrides an automated verdict for a specific criterion.
    Override is logged in the audit trail.
    """
    bv = _verdict_store.get(req.bidder_id)
    if not bv:
        raise HTTPException(404, f"No verdict for bidder '{req.bidder_id}'.")
    try:
        new_verdict_label = VerdictLabel(req.new_verdict)
    except ValueError:
        raise HTTPException(400, f"Invalid verdict '{req.new_verdict}'. Must be eligible/not_eligible/needs_review.")

    # Find and update the criterion verdict
    updated = False
    for cv in bv.criterion_verdicts:
        if cv.criterion_id == req.criterion_id:
            old = cv.verdict.value
            cv.verdict = new_verdict_label
            cv.reason = f"[HUMAN OVERRIDE by {req.officer_id}] {req.reason} (was: {old})"
            updated = True
            break
    if not updated:
        raise HTTPException(404, f"Criterion '{req.criterion_id}' not found in verdict.")

    # Recompute overall verdict after override
    engine = _get_verdict_engine()
    new_overall, new_reason = engine._compute_overall(bv.criterion_verdicts)
    bv.overall_verdict = new_overall
    bv.overall_reason = new_reason

    _audit_log.append(_get_report_generator().log_human_override(
        req.bidder_id, req.criterion_id, old, req.new_verdict, req.officer_id, req.reason
    ))
    return {"status": "updated", "new_overall_verdict": new_overall.value}

# =============================================================================
# Phase 5 — Report Generation + Audit
# =============================================================================

@app.post("/report/generate", response_model=ReportResponse, tags=["Phase 5 - Report"])
async def generate_report(tender_filename: str):
    """
    Generate the consolidated evaluation report for all bidders on a tender.
    All bidders must have verdicts computed via /verdict/{bidder_id}.
    """
    cs = _criteria_store.get(tender_filename)
    if not cs:
        raise HTTPException(404, f"No criteria for '{tender_filename}'.")

    # Collect all verdicts for bidders who submitted docs for this tender
    bidder_verdicts = []
    for bidder_id, filenames in _bidder_docs_store.items():
        bv = _verdict_store.get(bidder_id)
        if bv:
            bidder_verdicts.append(bv)

    if not bidder_verdicts:
        raise HTTPException(400, "No bidder verdicts found. Run /verdict/{bidder_id} for each bidder first.")

    t0 = time.perf_counter()
    engine = _get_verdict_engine()
    rg = _get_report_generator()

    report = engine.build_report(tender_filename, bidder_verdicts, cs, list(_audit_log))
    _report_store[tender_filename] = report

    _audit_log.append(rg.create_audit_entry(
        "report_generated", "system",
        {"tender": tender_filename, "total_bidders": report.total_bidders,
         "eligible": report.eligible_count, "not_eligible": report.not_eligible_count,
         "needs_review": report.needs_review_count}
    ))

    return ReportResponse(
        tender_filename=report.tender_filename,
        generated_at=report.generated_at.isoformat(),
        total_bidders=report.total_bidders,
        eligible_count=report.eligible_count,
        not_eligible_count=report.not_eligible_count,
        needs_review_count=report.needs_review_count,
        criteria_summary=report.criteria_summary,
        bidder_verdicts=[bv.model_dump() for bv in report.bidder_verdicts],
        audit_entries=len(_audit_log),
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 1),
    )

@app.get("/report/{tender_filename}", response_class=PlainTextResponse, tags=["Phase 5 - Report"])
async def get_report_markdown(tender_filename: str):
    """Download the evaluation report as a human-readable Markdown document."""
    report = _report_store.get(tender_filename)
    if not report:
        raise HTTPException(404, f"No report for '{tender_filename}'. Run /report/generate first.")
    md = _get_report_generator().generate_markdown(report)
    return PlainTextResponse(content=md, media_type="text/markdown")

@app.get("/report/{tender_filename}/json", tags=["Phase 5 - Report"])
async def get_report_json(tender_filename: str):
    """Download the full evaluation report as JSON."""
    report = _report_store.get(tender_filename)
    if not report:
        raise HTTPException(404, f"No report for '{tender_filename}'. Run /report/generate first.")
    return report.model_dump()

@app.get("/audit", tags=["Phase 5 - Report"])
async def get_audit_log():
    """Retrieve the complete immutable audit trail."""
    return {"total_entries": len(_audit_log), "entries": _audit_log}

# =============================================================================
# Internal helpers
# =============================================================================

def _require_api_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set. Export it and restart.")

def _get_doc(filename: str, doc_type: DocType) -> ExtractedDocument:
    doc = _document_store.get(filename)
    if not doc:
        raise HTTPException(404, f"Document '{filename}' not found. Upload it first.")
    if doc.doc_type != doc_type:
        raise HTTPException(400, f"'{filename}' is not a {doc_type.value} document.")
    return doc

async def _process_upload(upload: UploadFile, doc_type: DocType) -> IngestResponse:
    if not upload.filename:
        raise HTTPException(400, "File must have a filename")
    t0 = time.perf_counter()
    file_bytes = await upload.read()
    if not file_bytes:
        raise HTTPException(400, "Uploaded file is empty")
    try:
        doc = processor.process(file_bytes, upload.filename, doc_type)
    except ValueError as e:
        raise HTTPException(415, str(e))
    except Exception as e:
        logger.exception("Processing failed for '%s'", upload.filename)
        raise HTTPException(500, f"Processing error: {e}")
    _document_store[upload.filename] = doc
    return IngestResponse(
        filename=doc.filename,
        file_format=doc.file_format.value,
        doc_type=doc.doc_type.value,
        total_pages=doc.total_pages,
        overall_confidence=doc.overall_confidence,
        ocr_engine_used=doc.ocr_engine_used,
        warnings=doc.warnings,
        full_text_preview=doc.full_text[:500] + ("..." if len(doc.full_text) > 500 else ""),
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 1),
    )

def _criteria_response(cs: CriteriaSet, elapsed: float) -> CriteriaResponse:
    return CriteriaResponse(
        tender_filename=cs.tender_filename,
        total_criteria=cs.total_criteria,
        mandatory_count=cs.mandatory_count,
        optional_count=cs.optional_count,
        unclear_count=cs.unclear_count,
        extraction_model=cs.extraction_model,
        extraction_warnings=cs.extraction_warnings,
        criteria=[c.model_dump() for c in cs.criteria],
        processing_time_ms=round(elapsed * 1000, 1),
    )

def _evidence_response(ev: BidderEvidence, elapsed: float) -> EvidenceResponse:
    return EvidenceResponse(
        bidder_id=ev.bidder_id,
        documents_processed=ev.documents_processed,
        total_criteria=ev.total_criteria,
        overall_extraction_confidence=ev.overall_extraction_confidence,
        extraction_model=ev.extraction_model,
        extraction_warnings=ev.extraction_warnings,
        evidence=[e.model_dump() for e in ev.evidence],
        processing_time_ms=round(elapsed * 1000, 1),
    )

def _verdict_response(bv: BidderVerdict, elapsed: float) -> VerdictResponse:
    return VerdictResponse(
        bidder_id=bv.bidder_id,
        overall_verdict=bv.overall_verdict.value,
        overall_reason=bv.overall_reason,
        mandatory_passed=bv.mandatory_passed,
        mandatory_failed=bv.mandatory_failed,
        mandatory_review=bv.mandatory_review,
        optional_passed=bv.optional_passed,
        criterion_verdicts=[cv.model_dump() for cv in bv.criterion_verdicts],
        processing_time_ms=round(elapsed * 1000, 1),
    )