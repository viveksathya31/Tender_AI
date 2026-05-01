"""
OCR engine with two backends:
  1. Tesseract  — local, free, good for clean scans
  2. AWS Textract — cloud, paid, best for low-quality / handwritten / complex layouts

Selection strategy:
  - Try Tesseract first.
  - If mean word confidence < TESSERACT_CONFIDENCE_THRESHOLD, fall back to Textract.
  - Textract is always used when USE_TEXTRACT_ALWAYS=True in config.
"""

import io
import os
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from utils.image_preprocessing import preprocess_for_ocr, image_to_bytes

logger = logging.getLogger(__name__)

TESSERACT_CONFIDENCE_THRESHOLD = 70.0   # out of 100
USE_TEXTRACT_ALWAYS = os.getenv("USE_TEXTRACT_ALWAYS", "false").lower() == "true"
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")


@dataclass
class OCRResult:
    text: str
    confidence: float           # 0.0 – 1.0
    engine: str                 # "tesseract" | "textract" | "none"
    words_found: int


def run_ocr(image: Image.Image) -> OCRResult:
    """
    Main entry point. Accepts a PIL Image, returns OCRResult.
    Preprocessing is applied internally before either backend.
    """
    preprocessed = preprocess_for_ocr(image)

    if USE_TEXTRACT_ALWAYS and BOTO3_AVAILABLE:
        return _run_textract(preprocessed)

    if TESSERACT_AVAILABLE:
        result = _run_tesseract(preprocessed)
        if result.confidence >= TESSERACT_CONFIDENCE_THRESHOLD / 100:
            return result
        logger.info(
            "Tesseract confidence %.2f below threshold — falling back to Textract",
            result.confidence,
        )

    if BOTO3_AVAILABLE:
        return _run_textract(preprocessed)

    # No OCR backend available — return empty result with warning
    logger.warning("No OCR backend available (Tesseract=%s, boto3=%s)", TESSERACT_AVAILABLE, BOTO3_AVAILABLE)
    return OCRResult(text="", confidence=0.0, engine="none", words_found=0)


def _run_tesseract(img: Image.Image) -> OCRResult:
    """
    Run Tesseract and parse per-word confidence scores.
    Returns mean confidence across all words found.
    """
    try:
        # osd = orientation+script detection, useful for mixed-script docs
        data = pytesseract.image_to_data(
            img,
            lang="eng",
            config="--psm 3 --oem 3",   # psm3=auto, oem3=LSTM
            output_type=pytesseract.Output.DICT,
        )

        words = []
        confidences = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = int(data["conf"][i])
            if word and conf > 0:
                words.append(word)
                confidences.append(conf)

        text = pytesseract.image_to_string(img, lang="eng", config="--psm 3 --oem 3")
        mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        return OCRResult(
            text=text.strip(),
            confidence=round(mean_conf, 3),
            engine="tesseract",
            words_found=len(words),
        )
    except Exception as e:
        logger.error("Tesseract failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="tesseract", words_found=0)


def _run_textract(img: Image.Image) -> OCRResult:
    """
    Send image to AWS Textract DetectDocumentText.
    Returns extracted text and mean block confidence.
    """
    try:
        client = boto3.client("textract", region_name=AWS_REGION)
        img_bytes = image_to_bytes(img, fmt="PNG")

        response = client.detect_document_text(
            Document={"Bytes": img_bytes}
        )

        lines = []
        confidences = []
        for block in response.get("Blocks", []):
            if block["BlockType"] == "LINE":
                lines.append(block.get("Text", ""))
                confidences.append(block.get("Confidence", 0.0))

        text = "\n".join(lines)
        mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        return OCRResult(
            text=text.strip(),
            confidence=round(mean_conf, 3),
            engine="textract",
            words_found=len(lines),
        )
    except Exception as e:
        logger.error("Textract failed: %s", e)
        return OCRResult(text="", confidence=0.0, engine="textract", words_found=0)