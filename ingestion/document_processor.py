"""
DocumentProcessor — the single entry point for Phase 1.

Given raw file bytes + filename + declared doc_type, it:
  1. Detects the file format
  2. Routes to the correct parser (pdf_parser / other_parsers)
  3. Collects page results
  4. Computes overall confidence
  5. Assembles and returns an ExtractedDocument
"""

import logging
from pathlib import Path

from models.document import (
    DocType,
    ExtractedDocument,
    FileFormat,
    PageResult,
)
from ingestion.pdf_parser import parse_pdf
from ingestion.other_parsers import (
    parse_docx,
    parse_image,
    SUPPORTED_IMAGE_FORMATS,
)

logger = logging.getLogger(__name__)

# Pages with confidence below this get flagged in warnings
LOW_CONFIDENCE_THRESHOLD = 0.65


class DocumentProcessor:

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        doc_type: DocType = DocType.UNKNOWN,
    ) -> ExtractedDocument:
        """
        Main method. Returns a fully-populated ExtractedDocument.
        Raises ValueError for unsupported file types.
        Raises RuntimeError if parsing fails entirely.
        """
        suffix = Path(filename).suffix.lower()
        logger.info("Processing '%s' (%d bytes) as %s", filename, len(file_bytes), doc_type)

        # ── Route to parser ───────────────────────────────────────────────
        if suffix == ".pdf":
            pages, file_format, ocr_engine = parse_pdf(file_bytes, filename)

        elif suffix in (".docx", ".doc"):
            pages, file_format, ocr_engine = parse_docx(file_bytes, filename)

        elif suffix in SUPPORTED_IMAGE_FORMATS:
            pages, file_format, ocr_engine = parse_image(file_bytes, filename)

        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Accepted: .pdf, .docx, {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )

        # ── Build full text ───────────────────────────────────────────────
        full_text = _assemble_full_text(pages)

        # ── Compute overall confidence ────────────────────────────────────
        overall_confidence = _compute_overall_confidence(pages)

        # ── Collect warnings ──────────────────────────────────────────────
        warnings = _collect_warnings(pages, filename)

        doc = ExtractedDocument(
            filename=filename,
            file_format=file_format,
            doc_type=doc_type,
            total_pages=len(pages),
            full_text=full_text,
            pages=pages,
            overall_confidence=round(overall_confidence, 3),
            ocr_engine_used=ocr_engine,
            warnings=warnings,
        )

        logger.info(
            "Processed '%s': %d pages, format=%s, confidence=%.2f, warnings=%d",
            filename, len(pages), file_format, overall_confidence, len(warnings),
        )
        return doc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assemble_full_text(pages: list[PageResult]) -> str:
    parts = []
    for p in pages:
        if p.text.strip():
            parts.append(f"[Page {p.page_number}]\n{p.text.strip()}")
    return "\n\n".join(parts)


def _compute_overall_confidence(pages: list[PageResult]) -> float:
    if not pages:
        return 0.0
    # Weight by page — low-confidence pages drag the score down proportionally
    total = sum(p.confidence for p in pages)
    return total / len(pages)


def _collect_warnings(pages: list[PageResult], filename: str) -> list[str]:
    warnings = []
    for p in pages:
        if p.confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Page {p.page_number}: low OCR confidence ({p.confidence:.0%}). "
                "Manual review recommended for this page."
            )
        if p.is_ocr and p.confidence == 0.0:
            warnings.append(
                f"Page {p.page_number}: OCR returned no text. "
                "The page may be blank, heavily degraded, or non-Latin script."
            )
    return warnings