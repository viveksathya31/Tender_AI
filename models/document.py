from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocType(str, Enum):
    TENDER = "tender"
    BIDDER_SUBMISSION = "bidder_submission"
    UNKNOWN = "unknown"


class FileFormat(str, Enum):
    PDF_DIGITAL = "pdf_digital"     # native text PDF
    PDF_SCANNED = "pdf_scanned"     # image-based PDF
    DOCX = "docx"
    IMAGE = "image"                 # jpg, png, tiff, etc.
    UNKNOWN = "unknown"


class PageResult(BaseModel):
    page_number: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    has_tables: bool = False
    tables: list[list[list[str]]] = Field(default_factory=list)
    # tables[i] = one table, table[row][col] = cell text
    is_ocr: bool = False            # True if this page came through OCR


class ExtractedDocument(BaseModel):
    filename: str
    file_format: FileFormat
    doc_type: DocType
    total_pages: int
    full_text: str                  # concatenated clean text
    pages: list[PageResult]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    ocr_engine_used: Optional[str] = None   # "tesseract" | "textract" | None
    warnings: list[str] = Field(default_factory=list)
    # warnings: low-confidence pages, rotated images, partial failures, etc.