"""
FastAPI application for Phase 1 — Document Ingestion.

Endpoints:
  POST /ingest/tender       — upload and process a tender document
  POST /ingest/bidder       — upload and process one bidder submission
  POST /ingest/batch        — upload and process multiple bidder docs at once
  GET  /health              — liveness check
"""

import logging
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingestion.document_processor import DocumentProcessor
from models.document import DocType, ExtractedDocument

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tender Evaluation Platform — Ingestion API",
    description="Phase 1: Document ingestion, OCR, and text extraction",
    version="0.1.0",
)

processor = DocumentProcessor()


# ── Response schema ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    filename: str
    file_format: str
    doc_type: str
    total_pages: int
    overall_confidence: float
    ocr_engine_used: str | None
    warnings: list[str]
    full_text_preview: str      # first 500 chars — full text stored separately
    processing_time_ms: float


class BatchIngestResponse(BaseModel):
    total_files: int
    succeeded: int
    failed: int
    results: list[IngestResponse | dict]  # dict for error entries


# ── In-memory document store (replace with DB in prod) ───────────────────────

_document_store: dict[str, ExtractedDocument] = {}


def _store_document(doc: ExtractedDocument) -> None:
    _document_store[doc.filename] = doc


def _build_response(doc: ExtractedDocument, elapsed_ms: float) -> IngestResponse:
    return IngestResponse(
        filename=doc.filename,
        file_format=doc.file_format.value,
        doc_type=doc.doc_type.value,
        total_pages=doc.total_pages,
        overall_confidence=doc.overall_confidence,
        ocr_engine_used=doc.ocr_engine_used,
        warnings=doc.warnings,
        full_text_preview=doc.full_text[:500] + ("…" if len(doc.full_text) > 500 else ""),
        processing_time_ms=round(elapsed_ms, 1),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest/tender", response_model=IngestResponse)
async def ingest_tender(file: UploadFile = File(...)):
    """
    Upload a tender document (PDF, DOCX, or image).
    Extracts text, detects scanned pages, runs OCR where needed.
    """
    return await _process_upload(file, DocType.TENDER)


@app.post("/ingest/bidder", response_model=IngestResponse)
async def ingest_bidder(
    file: UploadFile = File(...),
    bidder_id: Annotated[str, Form()] = "unknown",
):
    """
    Upload one bidder submission document.
    bidder_id is used for logging and downstream matching.
    """
    doc = await _process_upload(file, DocType.BIDDER_SUBMISSION)
    # Tag filename with bidder_id for traceability
    logger.info("Bidder '%s' document ingested: %s", bidder_id, file.filename)
    return doc


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(files: list[UploadFile] = File(...)):
    """
    Upload multiple bidder documents in one request.
    Each file is processed independently; failures don't abort the batch.
    """
    results = []
    succeeded = 0
    failed = 0

    for upload in files:
        try:
            result = await _process_upload(upload, DocType.BIDDER_SUBMISSION)
            results.append(result)
            succeeded += 1
        except Exception as e:
            logger.error("Batch item failed '%s': %s", upload.filename, e)
            results.append({"filename": upload.filename, "error": str(e)})
            failed += 1

    return BatchIngestResponse(
        total_files=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.get("/documents/{filename}")
async def get_document(filename: str):
    """
    Retrieve a previously processed document's full text and metadata.
    In production this would query PostgreSQL.
    """
    doc = _document_store.get(filename)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    return doc.model_dump()


# ── Internal helpers ──────────────────────────────────────────────────────────

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

    _store_document(doc)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return _build_response(doc, elapsed_ms)