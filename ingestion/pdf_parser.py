"""
PDF parser — two modes:
  1. Digital PDF: use pdfplumber (fast, preserves tables, no quality loss)
  2. Scanned PDF: rasterise each page → OCR via ocr_engine

Detection: if a page has < MIN_CHARS_FOR_DIGITAL characters of extractable text,
it is treated as a scanned page and OCR is applied.
"""

import logging
from pathlib import Path
from typing import Optional

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image

from ingestion.ocr_engine import run_ocr
from models.document import FileFormat, PageResult

logger = logging.getLogger(__name__)

MIN_CHARS_FOR_DIGITAL = 50      # pages with fewer chars → treat as scanned
PDF_RENDER_DPI = 300            # render resolution for scanned pages


def parse_pdf(file_bytes: bytes, filename: str) -> tuple[list[PageResult], FileFormat, Optional[str]]:
    """
    Parse a PDF file.
    Returns (pages, file_format, ocr_engine_used).
    file_format is PDF_DIGITAL if all pages were native text, PDF_SCANNED if any needed OCR.
    """
    pages: list[PageResult] = []
    any_scanned = False
    ocr_engine_used: Optional[str] = None

    try:
        with pdfplumber.open(file_bytes if isinstance(file_bytes, (str, Path)) else __bytes_to_stream(file_bytes)) as pdf:
            total = len(pdf.pages)
            logger.info("Opened PDF '%s': %d pages", filename, total)

            # Pre-rasterise all pages once (needed only for scanned pages)
            # We do this lazily — only rasterise a page when pdfplumber finds < MIN_CHARS
            pil_pages: Optional[list[Image.Image]] = None

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                native_text = (page.extract_text() or "").strip()
                tables = _extract_tables(page)

                if len(native_text) >= MIN_CHARS_FOR_DIGITAL:
                    # Good digital page
                    pages.append(PageResult(
                        page_number=page_num,
                        text=native_text,
                        confidence=1.0,
                        has_tables=bool(tables),
                        tables=tables,
                        is_ocr=False,
                    ))
                else:
                    # Scanned page — rasterise and OCR
                    any_scanned = True
                    if pil_pages is None:
                        logger.info("Rasterising scanned PDF at %d DPI...", PDF_RENDER_DPI)
                        pil_pages = _rasterise_pdf(file_bytes)

                    if pil_pages and i < len(pil_pages):
                        ocr_result = run_ocr(pil_pages[i])
                        if not ocr_engine_used:
                            ocr_engine_used = ocr_result.engine

                        page_result = PageResult(
                            page_number=page_num,
                            text=ocr_result.text,
                            confidence=ocr_result.confidence,
                            has_tables=bool(tables),
                            tables=tables,
                            is_ocr=True,
                        )
                        if ocr_result.confidence < 0.6:
                            logger.warning(
                                "Low OCR confidence (%.2f) on page %d of '%s'",
                                ocr_result.confidence, page_num, filename,
                            )
                        pages.append(page_result)
                    else:
                        # Rasterisation failed — record empty page with warning
                        logger.error("Could not rasterise page %d of '%s'", page_num, filename)
                        pages.append(PageResult(
                            page_number=page_num,
                            text="",
                            confidence=0.0,
                            is_ocr=True,
                        ))

    except Exception as e:
        logger.error("PDF parsing failed for '%s': %s", filename, e)
        raise

    file_format = FileFormat.PDF_SCANNED if any_scanned else FileFormat.PDF_DIGITAL
    return pages, file_format, ocr_engine_used


def _extract_tables(page) -> list[list[list[str]]]:
    """Extract tables from a pdfplumber page as list of 2D string grids."""
    result = []
    try:
        for table in page.extract_tables() or []:
            cleaned = [
                [cell.strip() if cell else "" for cell in row]
                for row in table
            ]
            result.append(cleaned)
    except Exception:
        pass
    return result


def _rasterise_pdf(file_bytes: bytes) -> list[Image.Image]:
    """Convert all PDF pages to PIL Images at PDF_RENDER_DPI."""
    try:
        return convert_from_bytes(file_bytes, dpi=PDF_RENDER_DPI)
    except Exception as e:
        logger.error("pdf2image rasterisation failed: %s", e)
        return []


def __bytes_to_stream(b: bytes):
    import io
    return io.BytesIO(b)