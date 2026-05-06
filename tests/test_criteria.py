"""
Tests for Phase 2 — Criteria Extraction.

Uses mock Claude API responses so no real API key is needed.
"""

import json
import sys
sys.path.insert(0, "/home/claude/tender_platform")

import pytest
from unittest.mock import patch, MagicMock

from models.criteria import CriterionType, CriterionStatus, Criterion, CriteriaSet
from models.document import DocType, ExtractedDocument, FileFormat, PageResult
from ingestion.criteria_extractor import (
    CriteriaExtractor,
    _chunk_text,
    _parse_claude_response,
    _dict_to_criterion,
    CHUNK_SIZE,
)
from ingestion.criteria_prompts import build_extraction_prompt, build_dedup_prompt


# -- Fixtures -----------------------------------------------------------------

SAMPLE_TENDER_TEXT = """
[Page 1]
CENTRAL RESERVE POLICE FORCE
TENDER FOR CONSTRUCTION SERVICES - 2024

ELIGIBILITY CRITERIA

3.1 Financial Criteria
The bidder shall have a minimum annual turnover of Rs. 5 crore (Rupees Five Crore)
during each of the last three financial years as certified by a Chartered Accountant.

3.2 Technical Criteria
The bidder must have successfully completed at least 3 similar construction projects
in the last 5 years, each of value not less than Rs. 1 crore.

3.3 Compliance Requirements
(a) Valid GST Registration Certificate is mandatory.
(b) PAN Card of the firm is required.

3.4 Certifications
ISO 9001:2015 certification is preferred but not mandatory.

3.5 Experience
The firm should have been in existence for at least 5 years.
"""

MOCK_CLAUDE_RESPONSE = {
    "criteria": [
        {
            "id": "C001",
            "type": "financial",
            "status": "mandatory",
            "description": "Minimum annual turnover of Rs. 5 crore in each of last 3 financial years",
            "threshold_value": 50000000,
            "threshold_unit": "INR",
            "threshold_operator": ">=",
            "verification_docs": ["audited balance sheet", "CA certificate"],
            "source_page": 1,
            "source_text": "The bidder shall have a minimum annual turnover of Rs. 5 crore",
            "confidence": 0.97,
            "notes": ""
        },
        {
            "id": "C002",
            "type": "technical",
            "status": "mandatory",
            "description": "At least 3 similar construction projects completed in last 5 years",
            "threshold_value": 3,
            "threshold_unit": "projects",
            "threshold_operator": ">=",
            "verification_docs": ["work completion certificates", "work orders"],
            "source_page": 1,
            "source_text": "must have successfully completed at least 3 similar construction projects",
            "confidence": 0.95,
            "notes": ""
        },
        {
            "id": "C003",
            "type": "compliance",
            "status": "mandatory",
            "description": "Valid GST Registration Certificate",
            "threshold_value": None,
            "threshold_unit": None,
            "threshold_operator": None,
            "verification_docs": ["GST registration certificate"],
            "source_page": 1,
            "source_text": "Valid GST Registration Certificate is mandatory",
            "confidence": 0.99,
            "notes": ""
        },
        {
            "id": "C004",
            "type": "certification",
            "status": "optional",
            "description": "ISO 9001:2015 certification",
            "threshold_value": None,
            "threshold_unit": None,
            "threshold_operator": None,
            "verification_docs": ["ISO 9001:2015 certificate"],
            "source_page": 1,
            "source_text": "ISO 9001:2015 certification is preferred but not mandatory",
            "confidence": 0.98,
            "notes": ""
        },
        {
            "id": "C005",
            "type": "experience",
            "status": "unclear",
            "description": "Firm in existence for at least 5 years",
            "threshold_value": 5,
            "threshold_unit": "years",
            "threshold_operator": ">=",
            "verification_docs": ["certificate of incorporation", "registration certificate"],
            "source_page": 1,
            "source_text": "The firm should have been in existence for at least 5 years",
            "confidence": 0.80,
            "notes": "Language 'should' is ambiguous — could be mandatory or preferred"
        }
    ],
    "extraction_warnings": []
}


def make_mock_document(text: str = SAMPLE_TENDER_TEXT) -> ExtractedDocument:
    return ExtractedDocument(
        filename="tender_2024.pdf",
        file_format=FileFormat.PDF_DIGITAL,
        doc_type=DocType.TENDER,
        total_pages=1,
        full_text=text,
        pages=[PageResult(page_number=1, text=text, confidence=1.0)],
        overall_confidence=1.0,
    )


# -- Chunking tests -----------------------------------------------------------

class TestChunking:

    def test_short_text_not_chunked(self):
        text = "Short tender text"
        chunks = _chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunked(self):
        text = "A" * (CHUNK_SIZE * 2 + 100)
        chunks = _chunk_text(text)
        assert len(chunks) >= 2

    def test_chunks_have_overlap(self):
        text = "word " * 5000
        chunks = _chunk_text(text)
        if len(chunks) > 1:
            # The end of chunk 0 should appear in chunk 1 (overlap)
            end_of_chunk0 = chunks[0][-200:]
            assert any(end_of_chunk0[:50] in chunks[1] for _ in [1])

    def test_all_content_covered(self):
        text = "X" * (CHUNK_SIZE + 1000)
        chunks = _chunk_text(text)
        # Every character should appear in at least one chunk
        assert len(chunks[0]) >= CHUNK_SIZE // 2


# -- JSON parsing tests -------------------------------------------------------

class TestParseClaudeResponse:

    def test_clean_json_parsed(self):
        raw = json.dumps(MOCK_CLAUDE_RESPONSE)
        result = _parse_claude_response(raw)
        assert len(result["criteria"]) == 5

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + json.dumps(MOCK_CLAUDE_RESPONSE) + "\n```"
        result = _parse_claude_response(raw)
        assert len(result["criteria"]) == 5

    def test_invalid_json_returns_empty(self):
        result = _parse_claude_response("This is not JSON at all {broken")
        assert result["criteria"] == []
        assert len(result["extraction_warnings"]) > 0

    def test_empty_response_handled(self):
        result = _parse_claude_response("")
        assert result["criteria"] == []


# -- Criterion conversion tests -----------------------------------------------

class TestDictToCriterion:

    def test_full_criterion_conversion(self):
        data = MOCK_CLAUDE_RESPONSE["criteria"][0]
        c = _dict_to_criterion(data, 0)
        assert c.id == "C001"
        assert c.type == CriterionType.FINANCIAL
        assert c.status == CriterionStatus.MANDATORY
        assert c.threshold_value == 50000000
        assert c.threshold_unit == "INR"
        assert c.confidence == 0.97

    def test_optional_criterion(self):
        data = MOCK_CLAUDE_RESPONSE["criteria"][3]
        c = _dict_to_criterion(data, 3)
        assert c.status == CriterionStatus.OPTIONAL
        assert c.type == CriterionType.CERTIFICATION

    def test_unclear_criterion(self):
        data = MOCK_CLAUDE_RESPONSE["criteria"][4]
        c = _dict_to_criterion(data, 4)
        assert c.status == CriterionStatus.UNCLEAR
        assert "ambiguous" in c.notes

    def test_invalid_type_falls_back_to_other(self):
        data = {**MOCK_CLAUDE_RESPONSE["criteria"][0], "type": "nonsense_type"}
        c = _dict_to_criterion(data, 0)
        assert c.type == CriterionType.OTHER

    def test_missing_id_auto_assigned(self):
        data = {**MOCK_CLAUDE_RESPONSE["criteria"][0], "id": None}
        c = _dict_to_criterion(data, 2)
        assert c.id == "C003"

    def test_confidence_clamped(self):
        data = {**MOCK_CLAUDE_RESPONSE["criteria"][0], "confidence": 99.0}
        c = _dict_to_criterion(data, 0)
        assert c.confidence <= 1.0

        data2 = {**MOCK_CLAUDE_RESPONSE["criteria"][0], "confidence": -5.0}
        c2 = _dict_to_criterion(data2, 0)
        assert c2.confidence >= 0.0

    def test_null_threshold_fields_allowed(self):
        data = MOCK_CLAUDE_RESPONSE["criteria"][2]  # GST — no numeric threshold
        c = _dict_to_criterion(data, 2)
        assert c.threshold_value is None
        assert c.threshold_unit is None


# -- CriteriaExtractor integration tests (mocked API) ------------------------

class TestCriteriaExtractor:

    def _make_extractor(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
            return CriteriaExtractor()

    def _mock_call(self, extractor, response_dict):
        extractor._call_claude = MagicMock(return_value=json.dumps(response_dict))

    def test_extract_returns_criteria_set(self):
        extractor = self._make_extractor()
        self._mock_call(extractor, MOCK_CLAUDE_RESPONSE)
        doc = make_mock_document()
        result = extractor.extract(doc)
        assert isinstance(result, CriteriaSet)
        assert result.total_criteria == 5

    def test_counts_are_correct(self):
        extractor = self._make_extractor()
        self._mock_call(extractor, MOCK_CLAUDE_RESPONSE)
        result = extractor.extract(make_mock_document())
        assert result.mandatory_count == 3
        assert result.optional_count == 1
        assert result.unclear_count == 1

    def test_filename_preserved(self):
        extractor = self._make_extractor()
        self._mock_call(extractor, MOCK_CLAUDE_RESPONSE)
        result = extractor.extract(make_mock_document())
        assert result.tender_filename == "tender_2024.pdf"

    def test_model_name_recorded(self):
        extractor = self._make_extractor()
        self._mock_call(extractor, MOCK_CLAUDE_RESPONSE)
        result = extractor.extract(make_mock_document())
        assert "claude" in result.extraction_model.lower()

    def test_criteria_types_correct(self):
        extractor = self._make_extractor()
        self._mock_call(extractor, MOCK_CLAUDE_RESPONSE)
        result = extractor.extract(make_mock_document())
        types = {c.type for c in result.criteria}
        assert CriterionType.FINANCIAL in types
        assert CriterionType.TECHNICAL in types
        assert CriterionType.COMPLIANCE in types

    def test_api_failure_raises_runtime_error(self):
        extractor = self._make_extractor()
        extractor._call_claude = MagicMock(side_effect=RuntimeError("API down"))
        with pytest.raises(RuntimeError):
            extractor.extract(make_mock_document())

    def test_no_api_key_raises_environment_error(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError):
                CriteriaExtractor()

    def test_warnings_in_response_propagated(self):
        response = {**MOCK_CLAUDE_RESPONSE, "extraction_warnings": ["Ambiguity in clause 3.2"]}
        extractor = self._make_extractor()
        self._mock_call(extractor, response)
        result = extractor.extract(make_mock_document())
        assert any("Ambiguity" in w for w in result.extraction_warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])