# Tender Evaluation Platform — Phase 1: Document Ingestion

## What this does

Accepts any combination of tender documents and bidder submissions (PDF, DOCX, scanned images, photographs) and returns clean extracted text with per-page confidence scores. This feeds directly into Phase 2 (criteria extraction) and Phase 3 (evidence extraction).

## File structure

```
tender_platform/
├── main.py                        # FastAPI app — all HTTP endpoints
├── requirements.txt
├── ingestion/
│   ├── document_processor.py      # Orchestrator — single entry point
│   ├── pdf_parser.py              # Digital + scanned PDF handling
│   ├── other_parsers.py           # DOCX and image parsers
│   └── ocr_engine.py              # Tesseract + Textract OCR backends
├── models/
│   └── document.py                # Pydantic schemas (ExtractedDocument etc.)
├── utils/
│   └── image_preprocessing.py     # Deskew, denoise, binarise
└── tests/
    └── test_ingestion.py          # 17 tests, all passing
```

## Setup

### System dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

### Python dependencies

```bash
pip install -r requirements.txt
```

### AWS Textract (optional — cloud fallback for low-quality scans)

Set environment variables:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=ap-south-1

# Force all scans through Textract instead of Tesseract:
export USE_TEXTRACT_ALWAYS=true
```

If AWS credentials are not set, the system falls back to Tesseract only.

## Run the API

```bash
cd /path/to/project
uvicorn tender_platform.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## API endpoints

### Upload a tender document
```bash
curl -X POST http://localhost:8000/ingest/tender \
  -F "file=@/path/to/tender.pdf"
```

### Upload a bidder submission
```bash
curl -X POST http://localhost:8000/ingest/bidder \
  -F "file=@/path/to/bid.pdf" \
  -F "bidder_id=B001"
```

### Upload multiple bidder docs at once
```bash
curl -X POST http://localhost:8000/ingest/batch \
  -F "files=@bid1.pdf" \
  -F "files=@bid2.docx" \
  -F "files=@certificate.png"
```

### Sample response

```json
{
  "filename": "crpf_tender_2024.pdf",
  "file_format": "pdf_digital",
  "doc_type": "tender",
  "total_pages": 24,
  "overall_confidence": 1.0,
  "ocr_engine_used": null,
  "warnings": [],
  "full_text_preview": "[Page 1]\nCentral Reserve Police Force\nTender for Construction Services...",
  "processing_time_ms": 312.4
}
```

For a scanned document:

```json
{
  "filename": "gst_certificate_scan.jpg",
  "file_format": "image",
  "doc_type": "bidder_submission",
  "total_pages": 1,
  "overall_confidence": 0.84,
  "ocr_engine_used": "tesseract",
  "warnings": [],
  "full_text_preview": "[Page 1]\nGSTIN: 27AABCU9603R1ZX\nLegal Name: ABC Construction Pvt Ltd...",
  "processing_time_ms": 1840.2
}
```

If confidence is below 0.65, a warning is included:

```json
{
  "warnings": [
    "Page 3: low OCR confidence (52%). Manual review recommended for this page."
  ]
}
```

## Run tests

```bash
cd /path/to/project
python -m pytest tender_platform/tests/test_ingestion.py -v
```

## Key design decisions

**Why pdfplumber for digital PDFs?** It preserves table structure natively, which is critical for financial statements and eligibility criteria tables. PyPDF2/pypdf strip table layout.

**Why Tesseract first, Textract as fallback?** Cost control. Textract charges per page. For clean scans, Tesseract with preprocessing achieves comparable results for free. The confidence threshold (0.70) gates the fallback automatically.

**Why per-page confidence scoring?** A 20-page document might have 18 clean pages and 2 unreadable pages. Silently losing those 2 pages could cause a bidder to be incorrectly disqualified. Per-page confidence lets Phase 3 flag specific pages for human review rather than failing the whole submission.

**Why image preprocessing before OCR?** Government scans are frequently skewed, low-contrast, and noisy. The deskew + binarise pipeline lifts Tesseract accuracy by 15-30% on typical government scan quality.

## Next: Phase 2

Phase 2 takes the `full_text` output from this phase and sends it to Claude API with a structured extraction prompt to pull out all eligibility criteria as a typed JSON schema.