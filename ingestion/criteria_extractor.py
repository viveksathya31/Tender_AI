"""
Phase 2 — Criteria Extractor.

Takes the full_text output from Phase 1 (ExtractedDocument) and returns
a CriteriaSet with all eligibility criteria structured and typed.

Pipeline:
  1. Split tender text into chunks (Claude has a context window limit)
  2. Extract criteria from each chunk via Claude API
  3. If multiple chunks, deduplicate via a second Claude call
  4. Validate and assemble CriteriaSet
"""

import json
import logging
import os
import re
import httpx
from typing import Any

from models.criteria import Criterion, CriteriaSet, CriterionStatus, CriterionType
from models.document import ExtractedDocument
from ingestion.criteria_prompts import (
    SYSTEM_PROMPT,
    build_extraction_prompt,
    build_dedup_prompt,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 4096

# Chunk size in characters — keeps each prompt well within Claude's context window
# ~12,000 chars ≈ ~3,000 tokens of tender text, leaving room for output
CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 500     # overlap so criteria spanning chunk boundaries are not missed


class CriteriaExtractor:

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
            )

    def extract(self, document: ExtractedDocument) -> CriteriaSet:
        """
        Main entry point. Accepts an ExtractedDocument from Phase 1.
        Returns a fully-populated CriteriaSet.
        """
        text = document.full_text
        filename = document.filename
        logger.info("Extracting criteria from '%s' (%d chars)", filename, len(text))

        chunks = _chunk_text(text)
        logger.info("Split into %d chunk(s)", len(chunks))

        all_raw_criteria: list[dict] = []
        all_warnings: list[str] = []

        for i, chunk in enumerate(chunks):
            prompt = build_extraction_prompt(chunk, chunk_index=i, total_chunks=len(chunks))
            raw = self._call_claude(prompt)
            parsed = _parse_claude_response(raw)
            all_raw_criteria.extend(parsed.get("criteria", []))
            all_warnings.extend(parsed.get("extraction_warnings", []))

        # Deduplicate if we had multiple chunks
        if len(chunks) > 1:
            logger.info("Deduplicating criteria across %d chunks...", len(chunks))
            dedup_prompt = build_dedup_prompt(json.dumps({"criteria": all_raw_criteria}, indent=2))
            dedup_raw = self._call_claude(dedup_prompt)
            dedup_parsed = _parse_claude_response(dedup_raw)
            all_raw_criteria = dedup_parsed.get("criteria", all_raw_criteria)
            all_warnings.extend(dedup_parsed.get("extraction_warnings", []))

        criteria = [_dict_to_criterion(c, idx) for idx, c in enumerate(all_raw_criteria)]

        criteria_set = CriteriaSet(
            tender_filename=filename,
            total_criteria=len(criteria),
            mandatory_count=sum(1 for c in criteria if c.status == CriterionStatus.MANDATORY),
            optional_count=sum(1 for c in criteria if c.status == CriterionStatus.OPTIONAL),
            unclear_count=sum(1 for c in criteria if c.status == CriterionStatus.UNCLEAR),
            criteria=criteria,
            extraction_model=CLAUDE_MODEL,
            extraction_warnings=all_warnings,
        )

        logger.info(
            "Extracted %d criteria (%d mandatory, %d optional, %d unclear)",
            criteria_set.total_criteria,
            criteria_set.mandatory_count,
            criteria_set.optional_count,
            criteria_set.unclear_count,
        )
        return criteria_set

    def _call_claude(self, user_prompt: str) -> str:
        """
        Call Claude API and return the raw text response.
        Raises RuntimeError on API failure.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": CLAUDE_MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            response = httpx.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=body,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

        except httpx.HTTPStatusError as e:
            logger.error("Claude API HTTP error: %s — %s", e.response.status_code, e.response.text)
            raise RuntimeError(f"Claude API error {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            raise RuntimeError(f"Claude API call failed: {e}") from e


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks of CHUNK_SIZE characters.
    Tries to split on double newlines (paragraph boundaries) where possible.
    """
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            # Try to find a paragraph boundary near the end of the chunk
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP  # overlap to catch cross-boundary criteria
    return chunks


def _parse_claude_response(raw: str) -> dict[str, Any]:
    """
    Parse Claude's JSON response. Strips markdown fences if present.
    Returns empty structure on parse failure.
    """
    # Strip ```json ... ``` fences if Claude adds them
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Claude response as JSON: %s\nRaw: %s", e, raw[:300])
        return {"criteria": [], "extraction_warnings": [f"JSON parse error: {e}"]}


def _dict_to_criterion(data: dict, idx: int) -> Criterion:
    """
    Convert a raw dict from Claude's response into a validated Criterion.
    Falls back gracefully on missing or invalid fields.
    """
    # Normalise type
    raw_type = (data.get("type") or "other").lower()
    try:
        criterion_type = CriterionType(raw_type)
    except ValueError:
        criterion_type = CriterionType.OTHER

    # Normalise status
    raw_status = (data.get("status") or "unclear").lower()
    try:
        criterion_status = CriterionStatus(raw_status)
    except ValueError:
        criterion_status = CriterionStatus.UNCLEAR

    # Ensure ID
    criterion_id = data.get("id") or f"C{idx + 1:03d}"

    # Clamp confidence
    confidence = float(data.get("confidence", 1.0))
    confidence = max(0.0, min(1.0, confidence))

    return Criterion(
        id=criterion_id,
        type=criterion_type,
        status=criterion_status,
        description=data.get("description", ""),
        threshold_value=data.get("threshold_value"),
        threshold_unit=data.get("threshold_unit"),
        threshold_operator=data.get("threshold_operator"),
        verification_docs=data.get("verification_docs") or [],
        source_page=data.get("source_page"),
        source_text=data.get("source_text", ""),
        confidence=confidence,
        notes=data.get("notes", ""),
    )