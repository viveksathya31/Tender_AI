"""Tests for Phase 5 — Report Generator and Audit Trail."""

import sys, json
sys.path.insert(0, "/home/claude/tender_platform")

import pytest
from datetime import datetime
from models.verdict import BidderVerdict, ConsolidatedReport, CriterionVerdict, VerdictLabel
from models.criteria import Criterion, CriteriaSet, CriterionType, CriterionStatus
from ingestion.report_generator import ReportGenerator
from ingestion.verdict_engine import VerdictEngine

rg = ReportGenerator()
engine = VerdictEngine()


def make_criterion_verdict(cid, verdict, mandatory=True, confidence=0.95):
    return CriterionVerdict(
        criterion_id=cid,
        criterion_description=f"Criterion {cid}",
        criterion_type="financial",
        is_mandatory=mandatory,
        verdict=verdict,
        reason=f"Test reason for {cid}",
        confidence=confidence,
        evidence_status="found",
        evidence_source_doc="doc.pdf",
        evidence_source_page=1,
        evidence_source_text="sample text",
    )


def make_bidder_verdict(bidder_id, overall, criterion_verdicts):
    return BidderVerdict(
        bidder_id=bidder_id,
        overall_verdict=overall,
        overall_reason=f"Overall: {overall.value}",
        criterion_verdicts=criterion_verdicts,
        mandatory_passed=sum(1 for cv in criterion_verdicts if cv.verdict == VerdictLabel.ELIGIBLE and cv.is_mandatory),
        mandatory_failed=sum(1 for cv in criterion_verdicts if cv.verdict == VerdictLabel.NOT_ELIGIBLE and cv.is_mandatory),
        mandatory_review=sum(1 for cv in criterion_verdicts if cv.verdict == VerdictLabel.NEEDS_REVIEW and cv.is_mandatory),
        optional_passed=0,
        evaluated_at=datetime.utcnow(),
    )


def make_criteria_set():
    return CriteriaSet(
        tender_filename="tender.pdf",
        total_criteria=2,
        mandatory_count=2,
        optional_count=0,
        unclear_count=0,
        criteria=[
            Criterion(id="C001", type=CriterionType.FINANCIAL, status=CriterionStatus.MANDATORY,
                      description="Turnover >= 5cr", verification_docs=[]),
            Criterion(id="C002", type=CriterionType.COMPLIANCE, status=CriterionStatus.MANDATORY,
                      description="GST Registration", verification_docs=[]),
        ],
        extraction_model="claude",
    )


# -- Audit trail tests --------------------------------------------------------

class TestAuditTrail:

    def test_audit_entry_has_required_fields(self):
        entry = rg.create_audit_entry("test_event", "system", {"key": "value"})
        assert "event_type" in entry
        assert "actor" in entry
        assert "timestamp" in entry
        assert "data" in entry
        assert "hash" in entry

    def test_audit_entry_hash_is_16_chars(self):
        entry = rg.create_audit_entry("test", "system", {})
        assert len(entry["hash"]) == 16

    def test_different_data_different_hash(self):
        e1 = rg.create_audit_entry("test", "system", {"a": 1})
        e2 = rg.create_audit_entry("test", "system", {"a": 2})
        assert e1["hash"] != e2["hash"]

    def test_verdict_audit_entry_correct_type(self):
        bv = make_bidder_verdict("B001", VerdictLabel.ELIGIBLE, [
            make_criterion_verdict("C001", VerdictLabel.ELIGIBLE),
        ])
        entry = rg.log_verdict(bv, "claude-sonnet")
        assert entry["event_type"] == "verdict_computed"
        assert entry["data"]["bidder_id"] == "B001"
        assert entry["data"]["overall_verdict"] == "eligible"

    def test_human_override_audit_entry(self):
        entry = rg.log_human_override(
            "B001", "C001", "needs_review", "eligible",
            "officer_007", "Certificate verified manually"
        )
        assert entry["event_type"] == "human_override"
        assert entry["actor"] == "officer_007"
        assert entry["data"]["old_verdict"] == "needs_review"
        assert entry["data"]["new_verdict"] == "eligible"

    def test_audit_entry_timestamp_is_string(self):
        entry = rg.create_audit_entry("test", "system", {})
        assert isinstance(entry["timestamp"], str)


# -- Markdown report tests ----------------------------------------------------

class TestMarkdownReport:

    def make_report(self):
        cs = make_criteria_set()
        bv1 = make_bidder_verdict("B001", VerdictLabel.ELIGIBLE, [
            make_criterion_verdict("C001", VerdictLabel.ELIGIBLE),
            make_criterion_verdict("C002", VerdictLabel.ELIGIBLE),
        ])
        bv2 = make_bidder_verdict("B002", VerdictLabel.NOT_ELIGIBLE, [
            make_criterion_verdict("C001", VerdictLabel.NOT_ELIGIBLE),
            make_criterion_verdict("C002", VerdictLabel.ELIGIBLE),
        ])
        bv3 = make_bidder_verdict("B003", VerdictLabel.NEEDS_REVIEW, [
            make_criterion_verdict("C001", VerdictLabel.NEEDS_REVIEW),
            make_criterion_verdict("C002", VerdictLabel.ELIGIBLE),
        ])
        return engine.build_report("tender.pdf", [bv1, bv2, bv3], cs, [])

    def test_markdown_contains_title(self):
        report = self.make_report()
        md = rg.generate_markdown(report)
        assert "Tender Evaluation Report" in md

    def test_markdown_contains_all_bidders(self):
        report = self.make_report()
        md = rg.generate_markdown(report)
        assert "B001" in md
        assert "B002" in md
        assert "B003" in md

    def test_markdown_contains_summary_counts(self):
        report = self.make_report()
        md = rg.generate_markdown(report)
        assert "Eligible" in md
        assert "Not Eligible" in md or "Not_Eligible" in md or "not_eligible" in md

    def test_markdown_contains_officer_signature(self):
        report = self.make_report()
        md = rg.generate_markdown(report)
        assert "Officer signature" in md or "signature" in md.lower()

    def test_markdown_contains_criteria_section(self):
        report = self.make_report()
        md = rg.generate_markdown(report)
        assert "Criteria Summary" in md
        assert "C001" in md
        assert "C002" in md


# -- Consolidated report tests ------------------------------------------------

class TestConsolidatedReport:

    def test_counts_correct(self):
        cs = make_criteria_set()
        verdicts = [
            make_bidder_verdict("B001", VerdictLabel.ELIGIBLE, [make_criterion_verdict("C001", VerdictLabel.ELIGIBLE)]),
            make_bidder_verdict("B002", VerdictLabel.NOT_ELIGIBLE, [make_criterion_verdict("C001", VerdictLabel.NOT_ELIGIBLE)]),
            make_bidder_verdict("B003", VerdictLabel.NEEDS_REVIEW, [make_criterion_verdict("C001", VerdictLabel.NEEDS_REVIEW)]),
        ]
        report = engine.build_report("tender.pdf", verdicts, cs, [])
        assert report.total_bidders == 3
        assert report.eligible_count == 1
        assert report.not_eligible_count == 1
        assert report.needs_review_count == 1

    def test_criteria_summary_has_all_criteria(self):
        cs = make_criteria_set()
        verdicts = [make_bidder_verdict("B001", VerdictLabel.ELIGIBLE, [
            make_criterion_verdict("C001", VerdictLabel.ELIGIBLE),
            make_criterion_verdict("C002", VerdictLabel.ELIGIBLE),
        ])]
        report = engine.build_report("tender.pdf", verdicts, cs, [])
        ids = {s["criterion_id"] for s in report.criteria_summary}
        assert "C001" in ids
        assert "C002" in ids

    def test_audit_log_included_in_report(self):
        cs = make_criteria_set()
        verdicts = [make_bidder_verdict("B001", VerdictLabel.ELIGIBLE, [])]
        audit = [rg.create_audit_entry("test", "system", {"x": 1})]
        report = engine.build_report("tender.pdf", verdicts, cs, audit)
        assert len(report.audit_log) == 1

    def test_report_serialisable_to_json(self):
        cs = make_criteria_set()
        verdicts = [make_bidder_verdict("B001", VerdictLabel.ELIGIBLE, [])]
        report = engine.build_report("tender.pdf", verdicts, cs, [])
        json_str = json.dumps(report.model_dump(), default=str)
        parsed = json.loads(json_str)
        assert parsed["tender_filename"] == "tender.pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])