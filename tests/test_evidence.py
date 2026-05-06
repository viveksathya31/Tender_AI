"""
Tests for Phase 3 — Bidder Evidence Extraction.
All Claude API calls are mocked — no real API key needed.
"""

import json
import sys
sys.path.insert(0, "/home/claude/tender_platform")

import pytest
from unittest.mock import MagicMock, patch

from models.evidence import BidderEvidence, EvidenceResult, ExtractionStatus
from models.criteria import (
    Criterion, CriteriaSet, CriterionType, CriterionStatus
)
from models.document import DocType, ExtractedDocument, FileFormat, PageResult
from ingestion.evidence_extractor import (
    EvidenceExtractor,
    _criteria_to_json,
    _dict_to_evidence,
    _fill_missing_criteria,
    _split_doc,
    _parse_response,
    MAX_DOC_CHARS,
)


# -- Fixtures -----------------------------------------------------------------

def make_criteria_set() -> CriteriaSet:
    return CriteriaSet(
        tender_filename="tender.pdf",
        total_criteria=3,
        mandatory_count=2,
        optional_count=1,
        unclear_count=0,
        criteria=[
            Criterion(
                id="C001", type=CriterionType.FINANCIAL, status=CriterionStatus.MANDATORY,
                description="Minimum annual turnover of Rs. 5 crore",
                threshold_value=50000000, threshold_unit="INR", threshold_operator=">=",
                verification_docs=["audited balance sheet"],
            ),
            Criterion(
                id="C002", type=CriterionType.COMPLIANCE, status=CriterionStatus.MANDATORY,
                description="Valid GST Registration Certificate",
                verification_docs=["GST certificate"],
            ),
            Criterion(
                id="C003", type=CriterionType.CERTIFICATION, status=CriterionStatus.OPTIONAL,
                description="ISO 9001:2015 certification",
                verification_docs=["ISO certificate"],
            ),
        ],
        extraction_model="claude-sonnet-4-20250514",
    )


def make_bidder_doc(text: str, filename: str = "bid.pdf") -> ExtractedDocument:
    return ExtractedDocument(
        filename=filename,
        file_format=FileFormat.PDF_DIGITAL,
        doc_type=DocType.BIDDER_SUBMISSION,
        total_pages=1,
        full_text=text,
        pages=[PageResult(page_number=1, text=text, confidence=1.0)],
        overall_confidence=1.0,
    )


MOCK_EVIDENCE_RESPONSE = {
    "evidence": [
        {
            "criterion_id": "C001",
            "criterion_description": "Minimum annual turnover of Rs. 5 crore",
            "status": "found",
            "found_value": "Rs. 7.2 crore",
            "found_value_numeric": 72000000.0,
            "found_unit": "INR",
            "source_doc": "financial_statement.pdf",
            "source_page": 3,
            "source_text": "Total Annual Turnover: Rs. 7,20,00,000",
            "confidence": 0.93,
            "notes": ""
        },
        {
            "criterion_id": "C002",
            "criterion_description": "Valid GST Registration Certificate",
            "status": "found",
            "found_value": "27AABCU9603R1ZX",
            "found_value_numeric": None,
            "found_unit": None,
            "source_doc": "gst_certificate.pdf",
            "source_page": 1,
            "source_text": "GSTIN: 27AABCU9603R1ZX — Valid",
            "confidence": 0.99,
            "notes": ""
        },
        {
            "criterion_id": "C003",
            "criterion_description": "ISO 9001:2015 certification",
            "status": "not_found",
            "found_value": None,
            "found_value_numeric": None,
            "found_unit": None,
            "source_doc": "bid.pdf",
            "source_page": None,
            "source_text": "",
            "confidence": 0.95,
            "notes": "No ISO certificate found in submitted documents"
        }
    ],
    "extraction_warnings": []
}


def make_extractor() -> EvidenceExtractor:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        return EvidenceExtractor()


# -- Document splitting tests -------------------------------------------------

class TestSplitDoc:

    def test_short_doc_not_split(self):
        text = "Short document text"
        windows = _split_doc(text, "doc.pdf")
        assert len(windows) == 1
        assert windows[0][0] == text
        assert windows[0][1] == "doc.pdf"

    def test_long_doc_split(self):
        text = "word " * (MAX_DOC_CHARS // 4)
        windows = _split_doc(text, "long.pdf")
        assert len(windows) >= 2

    def test_window_labels_include_part_number(self):
        text = "x " * (MAX_DOC_CHARS // 2 + 500)
        windows = _split_doc(text, "doc.pdf")
        if len(windows) > 1:
            assert "part 2" in windows[1][1]


# -- Criteria JSON serialisation tests ----------------------------------------

class TestCriteriaToJson:

    def test_produces_valid_json(self):
        cs = make_criteria_set()
        result = _criteria_to_json(cs.criteria)
        parsed = json.loads(result)
        assert len(parsed) == 3

    def test_all_ids_present(self):
        cs = make_criteria_set()
        result = json.loads(_criteria_to_json(cs.criteria))
        ids = {item["id"] for item in result}
        assert ids == {"C001", "C002", "C003"}

    def test_threshold_fields_included(self):
        cs = make_criteria_set()
        result = json.loads(_criteria_to_json(cs.criteria))
        c001 = next(r for r in result if r["id"] == "C001")
        assert c001["threshold_value"] == 50000000
        assert c001["threshold_unit"] == "INR"
        assert c001["threshold_operator"] == ">="


# -- Response parsing tests ---------------------------------------------------

class TestParseResponse:

    def test_clean_json_parsed(self):
        raw = json.dumps(MOCK_EVIDENCE_RESPONSE)
        result = _parse_response(raw)
        assert len(result["evidence"]) == 3

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + json.dumps(MOCK_EVIDENCE_RESPONSE) + "\n```"
        result = _parse_response(raw)
        assert len(result["evidence"]) == 3

    def test_bad_json_returns_empty(self):
        result = _parse_response("{broken json")
        assert result["evidence"] == []
        assert len(result["extraction_warnings"]) > 0


# -- Dict to EvidenceResult tests ---------------------------------------------

class TestDictToEvidence:

    def make_criterion_map(self):
        cs = make_criteria_set()
        return {c.id: c for c in cs.criteria}

    def test_found_evidence_parsed(self):
        data = MOCK_EVIDENCE_RESPONSE["evidence"][0]
        ev = _dict_to_evidence(data, self.make_criterion_map())
        assert ev.criterion_id == "C001"
        assert ev.status == ExtractionStatus.FOUND
        assert ev.found_value_numeric == 72000000.0
        assert ev.confidence == 0.93
        assert ev.source_page == 3

    def test_not_found_evidence_parsed(self):
        data = MOCK_EVIDENCE_RESPONSE["evidence"][2]
        ev = _dict_to_evidence(data, self.make_criterion_map())
        assert ev.status == ExtractionStatus.NOT_FOUND
        assert ev.found_value is None

    def test_invalid_status_defaults_to_not_found(self):
        data = {**MOCK_EVIDENCE_RESPONSE["evidence"][0], "status": "gibberish"}
        ev = _dict_to_evidence(data, self.make_criterion_map())
        assert ev.status == ExtractionStatus.NOT_FOUND

    def test_confidence_clamped(self):
        data = {**MOCK_EVIDENCE_RESPONSE["evidence"][0], "confidence": 150.0}
        ev = _dict_to_evidence(data, self.make_criterion_map())
        assert ev.confidence <= 1.0

    def test_description_filled_from_criterion_map(self):
        data = {**MOCK_EVIDENCE_RESPONSE["evidence"][0], "criterion_description": ""}
        ev = _dict_to_evidence(data, self.make_criterion_map())
        assert "turnover" in ev.criterion_description.lower()


# -- Fill missing criteria tests ----------------------------------------------

class TestFillMissingCriteria:

    def test_missing_criterion_added_as_not_found(self):
        cs = make_criteria_set()
        # Only provide evidence for C001 and C002 — C003 missing
        partial = [
            EvidenceResult(
                criterion_id="C001", criterion_description="Turnover",
                status=ExtractionStatus.FOUND, confidence=0.9,
            ),
            EvidenceResult(
                criterion_id="C002", criterion_description="GST",
                status=ExtractionStatus.FOUND, confidence=0.95,
            ),
        ]
        filled = _fill_missing_criteria(partial, cs.criteria)
        ids = {e.criterion_id for e in filled}
        assert "C003" in ids
        c003 = next(e for e in filled if e.criterion_id == "C003")
        assert c003.status == ExtractionStatus.NOT_FOUND

    def test_no_duplicates_when_all_present(self):
        cs = make_criteria_set()
        all_ev = [
            EvidenceResult(criterion_id=c.id, criterion_description=c.description,
                           status=ExtractionStatus.FOUND, confidence=0.9)
            for c in cs.criteria
        ]
        filled = _fill_missing_criteria(all_ev, cs.criteria)
        assert len(filled) == 3


# -- EvidenceExtractor integration tests (mocked API) ------------------------

class TestEvidenceExtractor:

    def test_extract_returns_bidder_evidence(self):
        extractor = make_extractor()
        extractor._call_claude = MagicMock(return_value=json.dumps(MOCK_EVIDENCE_RESPONSE))
        cs = make_criteria_set()
        doc = make_bidder_doc("Annual turnover Rs 7.2 crore. GSTIN: 27AABCU9603R1ZX")
        result = extractor.extract("B001", [doc], cs)
        assert isinstance(result, BidderEvidence)
        assert result.bidder_id == "B001"

    def test_all_criteria_have_evidence(self):
        extractor = make_extractor()
        extractor._call_claude = MagicMock(return_value=json.dumps(MOCK_EVIDENCE_RESPONSE))
        cs = make_criteria_set()
        doc = make_bidder_doc("Some bidder text")
        result = extractor.extract("B001", [doc], cs)
        ids = {e.criterion_id for e in result.evidence}
        assert "C001" in ids and "C002" in ids and "C003" in ids

    def test_low_confidence_generates_warning(self):
        low_conf_response = {
            "evidence": [{
                **MOCK_EVIDENCE_RESPONSE["evidence"][0],
                "confidence": 0.50,   # below LOW_CONFIDENCE_THRESHOLD
                "status": "found",
            }],
            "extraction_warnings": []
        }
        extractor = make_extractor()
        extractor._call_claude = MagicMock(return_value=json.dumps(low_conf_response))
        cs = make_criteria_set()
        cs.criteria = [cs.criteria[0]]  # only C001
        cs.total_criteria = 1
        doc = make_bidder_doc("Some text")
        result = extractor.extract("B002", [doc], cs)
        assert any("manual review" in w.lower() for w in result.extraction_warnings)

    def test_documents_processed_list_populated(self):
        extractor = make_extractor()
        extractor._call_claude = MagicMock(return_value=json.dumps(MOCK_EVIDENCE_RESPONSE))
        cs = make_criteria_set()
        doc = make_bidder_doc("text", filename="financials.pdf")
        result = extractor.extract("B003", [doc], cs)
        assert "financials.pdf" in result.documents_processed

    def test_api_failure_raises(self):
        extractor = make_extractor()
        extractor._call_claude = MagicMock(side_effect=RuntimeError("API down"))
        cs = make_criteria_set()
        doc = make_bidder_doc("text")
        with pytest.raises(RuntimeError):
            extractor.extract("B004", [doc], cs)

    def test_overall_confidence_computed(self):
        extractor = make_extractor()
        extractor._call_claude = MagicMock(return_value=json.dumps(MOCK_EVIDENCE_RESPONSE))
        cs = make_criteria_set()
        doc = make_bidder_doc("text")
        result = extractor.extract("B005", [doc], cs)
        assert 0.0 <= result.overall_extraction_confidence <= 1.0

    def test_no_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(EnvironmentError):
                EvidenceExtractor()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])