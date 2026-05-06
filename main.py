"""
FastAPI application — Tender Evaluation Platform.

Phase 1 endpoints:
  POST /ingest/tender
  POST /ingest/bidder
  POST /ingest/batch
  GET  /documents/{filename}

Phase 2 endpoints:
  POST /criteria/extract
  GET  /criteria/{filename}

GET  /health
"""

import logging
import os
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ingestion.document_processor import DocumentProcessor
from models.document import DocType, ExtractedDocument
from models.criteria import CriteriaSet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tender Evaluation Platform",
    description="Phase 1: Document ingestion | Phase 2: Criteria extraction",
    version="0.2.0",
)

processor = DocumentProcessor()

_extractor = None
def get_extractor():
    global _extractor
    if _extractor is None:
        from ingestion.criteria_extractor import CriteriaExtractor
        _extractor = CriteriaExtractor()
    return _extractor

_document_store: dict[str, ExtractedDocument] = {}
_criteria_store: dict[str, CriteriaSet] = {}


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


@app.get("/health")
async def health():
    return {"status": "ok", "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/ingest/tender", response_model=IngestResponse)
async def ingest_tender(file: UploadFile = File(...)):
    return await _process_upload(file, DocType.TENDER)


@app.post("/ingest/bidder", response_model=IngestResponse)
async def ingest_bidder(
    file: UploadFile = File(...),
    bidder_id: Annotated[str, Form()] = "unknown",
):
    result = await _process_upload(file, DocType.BIDDER_SUBMISSION)
    # Track which docs belong to which bidder (for Phase 3)
    _bidder_docs_store.setdefault(bidder_id, [])
    if file.filename not in _bidder_docs_store[bidder_id]:
        _bidder_docs_store[bidder_id].append(file.filename)
    logger.info("Bidder '%s' ingested: %s", bidder_id, file.filename)
    return result


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(files: list[UploadFile] = File(...)):
    results = []
    succeeded = 0
    failed = 0
    for upload in files:
        try:
            results.append(await _process_upload(upload, DocType.BIDDER_SUBMISSION))
            succeeded += 1
        except Exception as e:
            results.append({"filename": upload.filename, "error": str(e)})
            failed += 1
    return BatchIngestResponse(total_files=len(files), succeeded=succeeded, failed=failed, results=results)


@app.get("/documents/{filename}")
async def get_document(filename: str):
    doc = _document_store.get(filename)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    return doc.model_dump()


@app.post("/criteria/extract", response_model=CriteriaResponse)
async def extract_criteria(filename: str):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set.")
    doc = _document_store.get(filename)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Tender '{filename}' not found. Upload via /ingest/tender first.")
    if doc.doc_type != DocType.TENDER:
        raise HTTPException(status_code=400, detail=f"'{filename}' is not a tender document.")
    t0 = time.perf_counter()
    try:
        cs = get_extractor().extract(doc)
    except Exception as e:
        logger.exception("Criteria extraction failed for '%s'", filename)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    _criteria_store[filename] = cs
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return CriteriaResponse(
        tender_filename=cs.tender_filename,
        total_criteria=cs.total_criteria,
        mandatory_count=cs.mandatory_count,
        optional_count=cs.optional_count,
        unclear_count=cs.unclear_count,
        extraction_model=cs.extraction_model,
        extraction_warnings=cs.extraction_warnings,
        criteria=[c.model_dump() for c in cs.criteria],
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.get("/criteria/{filename}", response_model=CriteriaResponse)
async def get_criteria(filename: str):
    cs = _criteria_store.get(filename)
    if not cs:
        raise HTTPException(status_code=404, detail=f"No criteria for '{filename}'. Run /criteria/extract first.")
    return CriteriaResponse(
        tender_filename=cs.tender_filename,
        total_criteria=cs.total_criteria,
        mandatory_count=cs.mandatory_count,
        optional_count=cs.optional_count,
        unclear_count=cs.unclear_count,
        extraction_model=cs.extraction_model,
        extraction_warnings=cs.extraction_warnings,
        criteria=[c.model_dump() for c in cs.criteria],
        processing_time_ms=0,
    )


async def _process_upload(upload: UploadFile, doc_type: DocType) -> IngestResponse:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
    t0 = time.perf_counter()
    file_bytes = await upload.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        doc = processor.process(file_bytes, upload.filename, doc_type)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        logger.exception("Processing failed for '%s'", upload.filename)
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    _document_store[upload.filename] = doc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return IngestResponse(
        filename=doc.filename,
        file_format=doc.file_format.value,
        doc_type=doc.doc_type.value,
        total_pages=doc.total_pages,
        overall_confidence=doc.overall_confidence,
        ocr_engine_used=doc.ocr_engine_used,
        warnings=doc.warnings,
        full_text_preview=doc.full_text[:500] + ("..." if len(doc.full_text) > 500 else ""),
        processing_time_ms=round(elapsed_ms, 1),
    )


# =============================================================================
# Phase 3 — Bidder Evidence Extraction
# =============================================================================

from models.evidence import BidderEvidence

_evidence_store: dict[str, BidderEvidence] = {}  # key: bidder_id
_bidder_docs_store: dict[str, list[str]] = {}     # key: bidder_id → list of filenames

_evidence_extractor = None
def get_evidence_extractor():
    global _evidence_extractor
    if _evidence_extractor is None:
        from ingestion.evidence_extractor import EvidenceExtractor
        _evidence_extractor = EvidenceExtractor()
    return _evidence_extractor


class EvidenceResponse(BaseModel):
    bidder_id: str
    documents_processed: list[str]
    total_criteria: int
    overall_extraction_confidence: float
    extraction_model: str
    extraction_warnings: list[str]
    evidence: list[dict]
    processing_time_ms: float


@app.post("/evidence/extract", response_model=EvidenceResponse)
async def extract_evidence(bidder_id: str, tender_filename: str):
    """
    Extract evidence for a specific bidder against a tender's criteria.

    Prerequisites:
      - Tender uploaded via POST /ingest/tender
      - Criteria extracted via POST /criteria/extract
      - Bidder docs uploaded via POST /ingest/bidder with matching bidder_id
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set.")

    # Validate criteria exist
    criteria_set = _criteria_store.get(tender_filename)
    if not criteria_set:
        raise HTTPException(
            status_code=404,
            detail=f"No criteria found for '{tender_filename}'. Run /criteria/extract first.",
        )

    # Validate bidder docs exist
    bidder_filenames = _bidder_docs_store.get(bidder_id, [])
    if not bidder_filenames:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for bidder '{bidder_id}'. Upload via /ingest/bidder first.",
        )

    bidder_docs = [_document_store[f] for f in bidder_filenames if f in _document_store]
    if not bidder_docs:
        raise HTTPException(status_code=404, detail=f"Bidder documents missing from store.")

    t0 = time.perf_counter()
    try:
        result = get_evidence_extractor().extract(
            bidder_id=bidder_id,
            bidder_docs=bidder_docs,
            criteria_set=criteria_set,
        )
    except Exception as e:
        logger.exception("Evidence extraction failed for bidder '%s'", bidder_id)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")

    _evidence_store[bidder_id] = result
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return EvidenceResponse(
        bidder_id=result.bidder_id,
        documents_processed=result.documents_processed,
        total_criteria=result.total_criteria,
        overall_extraction_confidence=result.overall_extraction_confidence,
        extraction_model=result.extraction_model,
        extraction_warnings=result.extraction_warnings,
        evidence=[e.model_dump() for e in result.evidence],
        processing_time_ms=round(elapsed_ms, 1),
    )


@app.get("/evidence/{bidder_id}", response_model=EvidenceResponse)
async def get_evidence(bidder_id: str):
    """Retrieve previously extracted evidence for a bidder."""
    result = _evidence_store.get(bidder_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No evidence for bidder '{bidder_id}'. Run /evidence/extract first.",
        )
    return EvidenceResponse(
        bidder_id=result.bidder_id,
        documents_processed=result.documents_processed,
        total_criteria=result.total_criteria,
        overall_extraction_confidence=result.overall_extraction_confidence,
        extraction_model=result.extraction_model,
        extraction_warnings=result.extraction_warnings,
        evidence=[e.model_dump() for e in result.evidence],
        processing_time_ms=0,
    )