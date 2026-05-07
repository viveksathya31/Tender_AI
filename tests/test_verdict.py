"""Tests for Phase 4 — Verdict Engine."""

import sys
sys.path.insert(0, "/home/claude/tender_platform")

import pytest
from models.criteria import Criterion, CriteriaSet, CriterionType, CriterionStatus
from models.evidence import BidderEvidence, EvidenceResult, ExtractionStatus
from models.verdict import VerdictLabel
from ingestion.verdict_engine import VerdictEngine, _check_threshold

engine = VerdictEngine()


def make_criteria(overrides=None):
    defaults = [
        Criterion(id="C001", type=CriterionType.FINANCIAL, status=CriterionStatus.MANDATORY,
                  description="Turnover >= 5cr", threshold_value=50000000,
                  threshold_unit="INR", threshold_operator=">=", verification_docs=[]),
        Criterion(id="C002", type=CriterionType.COMPLIANCE, status=CriterionStatus.MANDATORY,
                  description="GST Registration", verification_docs=[]),
        Criterion(id="C003", type=CriterionType.CERTIFICATION, status=CriterionStatus.OPTIONAL,
                  description="ISO 9001", verification_docs=[]),
    ]
    return CriteriaSet(tender_filename="t.pdf", total_criteria=3,
                       mandatory_count=2, optional_count=1, unclear_count=0,
                       criteria=overrides or defaults, extraction_model="claude")


def make_evidence(criterion_id, status, value_numeric=None, value=None,
                  confidence=0.95, source_doc="doc.pdf", source_page=1,
                  source_text="some text"):
    return EvidenceResult(
        criterion_id=criterion_id,
        criterion_description="",
        status=status,
        found_value=value,
        found_value_numeric=value_numeric,
        found_unit="INR" if value_numeric else None,
        source_doc=source_doc,
        source_page=source_page,
        source_text=source_text,
        confidence=confidence,
    )


def make_bidder_evidence(evidence_list):
    return BidderEvidence(
        bidder_id="B001",
        documents_processed=["doc.pdf"],
        total_criteria=len(evidence_list),
        evidence=evidence_list,
        extraction_model="claude",
        overall_extraction_confidence=0.9,
    )


# -- Threshold tests ----------------------------------------------------------

class TestCheckThreshold:
    def test_gte_pass(self): assert _check_threshold(60000000, 50000000, ">=")
    def test_gte_exact(self): assert _check_threshold(50000000, 50000000, ">=")
    def test_gte_fail(self): assert not _check_threshold(40000000, 50000000, ">=")
    def test_gt_pass(self): assert _check_threshold(50000001, 50000000, ">")
    def test_lte_pass(self): assert _check_threshold(3, 5, "<=")
    def test_eq_pass(self): assert _check_threshold(5, 5, "=")
    def test_unknown_operator(self): assert not _check_threshold(5, 5, "??")


# -- Criterion verdict tests --------------------------------------------------

class TestCriterionVerdict:

    def test_numeric_found_passes_threshold(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=72000000, value="7.2cr"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GSTIN valid"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c001 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C001")
        assert c001.verdict == VerdictLabel.ELIGIBLE
        assert "7.2cr" in c001.reason or "72000000" in c001.reason

    def test_numeric_fails_threshold(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=20000000, value="2cr"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST valid"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c001 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C001")
        assert c001.verdict == VerdictLabel.NOT_ELIGIBLE

    def test_not_found_mandatory_is_not_eligible(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.NOT_FOUND),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST valid"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c001 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C001")
        assert c001.verdict == VerdictLabel.NOT_ELIGIBLE

    def test_partial_evidence_needs_review(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.PARTIAL, value="unclear figure"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST valid"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c001 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C001")
        assert c001.verdict == VerdictLabel.NEEDS_REVIEW

    def test_unreadable_doc_needs_review(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.UNREADABLE),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c001 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C001")
        assert c001.verdict == VerdictLabel.NEEDS_REVIEW

    def test_low_confidence_found_needs_review(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=72000000,
                          value="7.2cr", confidence=0.50),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c001 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C001")
        assert c001.verdict == VerdictLabel.NEEDS_REVIEW

    def test_non_numeric_found_high_conf_eligible(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=60000000, value="6cr"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GSTIN: 27AAB"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        c002 = next(cv for cv in bv.criterion_verdicts if cv.criterion_id == "C002")
        assert c002.verdict == VerdictLabel.ELIGIBLE

    def test_reason_always_populated(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.NOT_FOUND),
            make_evidence("C002", ExtractionStatus.NOT_FOUND),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        for cv in bv.criterion_verdicts:
            assert cv.reason.strip() != ""


# -- Overall verdict tests ----------------------------------------------------

class TestOverallVerdict:

    def test_all_pass_eligible(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=72000000, value="7.2cr"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST valid"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        assert bv.overall_verdict == VerdictLabel.ELIGIBLE

    def test_mandatory_fail_not_eligible(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=10000000, value="1cr"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        assert bv.overall_verdict == VerdictLabel.NOT_ELIGIBLE
        assert bv.mandatory_failed >= 1

    def test_mandatory_review_needs_review(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.PARTIAL, value="unclear"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        assert bv.overall_verdict == VerdictLabel.NEEDS_REVIEW
        assert bv.mandatory_review >= 1

    def test_optional_failure_does_not_affect_overall(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=72000000, value="7.2cr"),
            make_evidence("C002", ExtractionStatus.FOUND, value="GST"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),  # optional
        ])
        bv = engine.evaluate_bidder(ev, cs)
        assert bv.overall_verdict == VerdictLabel.ELIGIBLE

    def test_fail_takes_priority_over_review(self):
        cs = make_criteria()
        ev = make_bidder_evidence([
            make_evidence("C001", ExtractionStatus.FOUND, value_numeric=10000000, value="1cr"),
            make_evidence("C002", ExtractionStatus.PARTIAL, value="unclear"),
            make_evidence("C003", ExtractionStatus.NOT_FOUND),
        ])
        bv = engine.evaluate_bidder(ev, cs)
        assert bv.overall_verdict == VerdictLabel.NOT_ELIGIBLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])