"""
Phase 3 — Evidence Extractor.

For a given bidder (one or more documents) and a CriteriaSet from Phase 2,
extracts evidence for every criterion from every document, then merges.

Pipeline per bidder:
  1. For each submitted document, call Claude with all criteria + doc text
  2. If bidder submitted multiple docs, merge evidence across docs (keep best per criterion)
  3. Validate, compute confidence, assemble BidderEvidence
"""

import json
import logging
import os
import re
import httpx
from typing import Any

from models.criteria import CriteriaSet, Criterion
from models.document import ExtractedDocument
from models.evidence import BidderEvidence, EvidenceResult, ExtractionStatus
from ingestion.evidence_prompts import (
    SYSTEM_PROMPT,
    build_evidence_prompt,
    build_merge_prompt,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 4096

# Max chars of bidder doc text sent per Claude call
# Keeps prompt within context window even with all criteria included
MAX_DOC_CHARS = 10_000

# Evidence confidence below this threshold → flagged as needing human review
LOW_CONFIDENCE_THRESHOLD = 0.70


class EvidenceExtractor:

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Export it before running."
            )

    def extract(
        self,
        bidder_id: str,
        bidder_docs: list[ExtractedDocument],
        criteria_set: CriteriaSet,
    ) -> BidderEvidence:
        """
        Main entry point.

        bidder_id     — unique identifier for this bidder (e.g. "B001")
        bidder_docs   — list of ExtractedDocuments from Phase 1 for this bidder
        criteria_set  — CriteriaSet from Phase 2 for the tender

        Returns a BidderEvidence with one EvidenceResult per criterion.
        """
        logger.info(
            "Extracting evidence for bidder '%s' across %d document(s)",
            bidder_id, len(bidder_docs),
        )

        criteria_json = _criteria_to_json(criteria_set.criteria)
        all_raw_evidence: list[dict] = []
        all_warnings: list[str] = []

        for doc in bidder_docs:
            # A single document may be very long — process in windows
            windows = _split_doc(doc.full_text, doc.filename)
            for window_text, window_label in windows:
                prompt = build_evidence_prompt(
                    criteria_json=criteria_json,
                    bidder_text=window_text,
                    bidder_id=bidder_id,
                    doc_filename=window_label,
                )
                raw = self._call_claude(prompt)
                parsed = _parse_response(raw)
                # Tag each result with source doc
                for item in parsed.get("evidence", []):
                    item.setdefault("source_doc", doc.filename)
                all_raw_evidence.extend(parsed.get("evidence", []))
                all_warnings.extend(parsed.get("extraction_warnings", []))

        # If multiple docs/windows → merge (keep best evidence per criterion)
        if len(bidder_docs) > 1 or sum(len(_split_doc(d.full_text, d.filename)) for d in bidder_docs) > 1:
            logger.info("Merging evidence across documents for bidder '%s'", bidder_id)
            merge_prompt = build_merge_prompt(
                json.dumps({"evidence": all_raw_evidence}, indent=2),
                bidder_id=bidder_id,
            )
            merged_raw = self._call_claude(merge_prompt)
            merged_parsed = _parse_response(merged_raw)
            all_raw_evidence = merged_parsed.get("evidence", all_raw_evidence)
            all_warnings.extend(merged_parsed.get("extraction_warnings", []))

        # Build typed EvidenceResult objects
        criterion_map = {c.id: c for c in criteria_set.criteria}
        evidence_results = [
            _dict_to_evidence(item, criterion_map)
            for item in all_raw_evidence
        ]

        # Ensure every criterion has an entry (fill not_found for missing ones)
        evidence_results = _fill_missing_criteria(evidence_results, criteria_set.criteria)

        # Collect low-confidence warnings
        for ev in evidence_results:
            if ev.confidence < LOW_CONFIDENCE_THRESHOLD and ev.status == ExtractionStatus.FOUND:
                all_warnings.append(
                    f"Criterion {ev.criterion_id}: evidence found but confidence is low "
                    f"({ev.confidence:.0%}) — manual review recommended."
                )

        overall_conf = (
            sum(e.confidence for e in evidence_results) / len(evidence_results)
            if evidence_results else 0.0
        )

        return BidderEvidence(
            bidder_id=bidder_id,
            documents_processed=[d.filename for d in bidder_docs],
            total_criteria=len(evidence_results),
            evidence=evidence_results,
            extraction_model=CLAUDE_MODEL,
            extraction_warnings=all_warnings,
            overall_extraction_confidence=round(overall_conf, 3),
        )

    def _call_claude(self, user_prompt: str) -> str:
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
            return response.json()["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            logger.error("Claude API error %s: %s", e.response.status_code, e.response.text)
            raise RuntimeError(f"Claude API error {e.response.status_code}") from e
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            raise RuntimeError(f"Claude API call failed: {e}") from e


# -- Helpers ------------------------------------------------------------------

def _criteria_to_json(criteria: list[Criterion]) -> str:
    """Serialize criteria list to compact JSON for inclusion in prompts."""
    items = []
    for c in criteria:
        items.append({
            "id": c.id,
            "type": c.type.value,
            "status": c.status.value,
            "description": c.description,
            "threshold_value": c.threshold_value,
            "threshold_unit": c.threshold_unit,
            "threshold_operator": c.threshold_operator,
            "verification_docs": c.verification_docs,
        })
    return json.dumps(items, indent=2)


def _split_doc(full_text: str, filename: str) -> list[tuple[str, str]]:
    """
    Split a long document into windows that fit within MAX_DOC_CHARS.
    Returns list of (text_window, label) tuples.
    """
    if len(full_text) <= MAX_DOC_CHARS:
        return [(full_text, filename)]

    windows = []
    start = 0
    part = 1
    while start < len(full_text):
        end = start + MAX_DOC_CHARS
        # Try to split at paragraph boundary
        boundary = full_text.rfind("\n\n", start, end)
        if boundary > start + MAX_DOC_CHARS // 2:
            end = boundary
        windows.append((full_text[start:end], f"{filename} [part {part}]"))
        start = end
        part += 1
    return windows


def _parse_response(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse evidence response: %s", e)
        return {"evidence": [], "extraction_warnings": [f"JSON parse error: {e}"]}


def _dict_to_evidence(data: dict, criterion_map: dict[str, Criterion]) -> EvidenceResult:
    criterion_id = data.get("criterion_id", "UNKNOWN")
    criterion = criterion_map.get(criterion_id)
    description = (
        criterion.description if criterion
        else data.get("criterion_description", "")
    )

    raw_status = (data.get("status") or "not_found").lower()
    try:
        status = ExtractionStatus(raw_status)
    except ValueError:
        status = ExtractionStatus.NOT_FOUND

    confidence = float(data.get("confidence", 1.0))
    confidence = max(0.0, min(1.0, confidence))

    return EvidenceResult(
        criterion_id=criterion_id,
        criterion_description=description,
        status=status,
        found_value=data.get("found_value"),
        found_value_numeric=data.get("found_value_numeric"),
        found_unit=data.get("found_unit"),
        source_doc=data.get("source_doc", ""),
        source_page=data.get("source_page"),
        source_text=data.get("source_text", ""),
        confidence=confidence,
        notes=data.get("notes", ""),
    )


def _fill_missing_criteria(
    evidence: list[EvidenceResult],
    criteria: list[Criterion],
) -> list[EvidenceResult]:
    """
    Ensure every criterion has an EvidenceResult.
    Missing ones are added as not_found with zero confidence.
    """
    found_ids = {e.criterion_id for e in evidence}
    for c in criteria:
        if c.id not in found_ids:
            logger.warning("No evidence entry returned for criterion %s — filling as not_found", c.id)
            evidence.append(EvidenceResult(
                criterion_id=c.id,
                criterion_description=c.description,
                status=ExtractionStatus.NOT_FOUND,
                confidence=0.0,
                notes="No evidence entry returned by extractor — treated as not found.",
            ))
    return evidence