"""
Phase 4 — Verdict Engine.

For each (criterion, evidence) pair, decides:
  ELIGIBLE      — evidence clearly satisfies the criterion
  NOT_ELIGIBLE  — evidence clearly fails the criterion
  NEEDS_REVIEW  — ambiguous, low confidence, partial, or unreadable

Rules are deterministic and fully explainable — no black-box decisions.
Every verdict carries a reason string referencing the specific evidence.

Overall bidder verdict:
  NOT_ELIGIBLE  — if any mandatory criterion is NOT_ELIGIBLE
  NEEDS_REVIEW  — if any mandatory criterion is NEEDS_REVIEW (and none failed)
  ELIGIBLE      — only if ALL mandatory criteria pass
"""

import logging
from datetime import datetime

from models.criteria import CriteriaSet, Criterion, CriterionStatus
from models.evidence import BidderEvidence, EvidenceResult, ExtractionStatus
from models.verdict import (
    BidderVerdict,
    ConsolidatedReport,
    CriterionVerdict,
    VerdictLabel,
)

logger = logging.getLogger(__name__)

# Confidence below this → NEEDS_REVIEW even if evidence was "found"
CONFIDENCE_REVIEW_THRESHOLD = 0.70

# Tolerance for numeric comparisons (e.g. rounding in financial docs)
NUMERIC_TOLERANCE = 0.01


class VerdictEngine:

    def evaluate_bidder(
        self,
        bidder_evidence: BidderEvidence,
        criteria_set: CriteriaSet,
    ) -> BidderVerdict:
        """
        Produce a full criterion-by-criterion verdict for one bidder.
        """
        criterion_map = {c.id: c for c in criteria_set.criteria}
        evidence_map = {e.criterion_id: e for e in bidder_evidence.evidence}

        criterion_verdicts: list[CriterionVerdict] = []

        for criterion in criteria_set.criteria:
            evidence = evidence_map.get(criterion.id)
            cv = self._evaluate_criterion(criterion, evidence)
            criterion_verdicts.append(cv)

        overall_verdict, overall_reason = self._compute_overall(criterion_verdicts)

        mandatory_passed = sum(
            1 for cv in criterion_verdicts
            if cv.is_mandatory and cv.verdict == VerdictLabel.ELIGIBLE
        )
        mandatory_failed = sum(
            1 for cv in criterion_verdicts
            if cv.is_mandatory and cv.verdict == VerdictLabel.NOT_ELIGIBLE
        )
        mandatory_review = sum(
            1 for cv in criterion_verdicts
            if cv.is_mandatory and cv.verdict == VerdictLabel.NEEDS_REVIEW
        )
        optional_passed = sum(
            1 for cv in criterion_verdicts
            if not cv.is_mandatory and cv.verdict == VerdictLabel.ELIGIBLE
        )

        logger.info(
            "Bidder '%s': %s | mandatory passed=%d failed=%d review=%d",
            bidder_evidence.bidder_id, overall_verdict,
            mandatory_passed, mandatory_failed, mandatory_review,
        )

        return BidderVerdict(
            bidder_id=bidder_evidence.bidder_id,
            overall_verdict=overall_verdict,
            overall_reason=overall_reason,
            criterion_verdicts=criterion_verdicts,
            mandatory_passed=mandatory_passed,
            mandatory_failed=mandatory_failed,
            mandatory_review=mandatory_review,
            optional_passed=optional_passed,
            evaluated_at=datetime.utcnow(),
        )

    def build_report(
        self,
        tender_filename: str,
        bidder_verdicts: list[BidderVerdict],
        criteria_set: CriteriaSet,
        audit_log: list[dict],
    ) -> ConsolidatedReport:
        """Assemble the consolidated report across all bidders."""
        eligible = sum(1 for v in bidder_verdicts if v.overall_verdict == VerdictLabel.ELIGIBLE)
        not_eligible = sum(1 for v in bidder_verdicts if v.overall_verdict == VerdictLabel.NOT_ELIGIBLE)
        needs_review = sum(1 for v in bidder_verdicts if v.overall_verdict == VerdictLabel.NEEDS_REVIEW)

        # Per-criterion summary
        criteria_summary = []
        for c in criteria_set.criteria:
            counts = {"criterion_id": c.id, "description": c.description,
                      "mandatory": c.status == CriterionStatus.MANDATORY,
                      "eligible": 0, "not_eligible": 0, "needs_review": 0}
            for bv in bidder_verdicts:
                for cv in bv.criterion_verdicts:
                    if cv.criterion_id == c.id:
                        counts[cv.verdict.value] += 1
            criteria_summary.append(counts)

        return ConsolidatedReport(
            tender_filename=tender_filename,
            generated_at=datetime.utcnow(),
            total_bidders=len(bidder_verdicts),
            eligible_count=eligible,
            not_eligible_count=not_eligible,
            needs_review_count=needs_review,
            criteria_summary=criteria_summary,
            bidder_verdicts=bidder_verdicts,
            audit_log=audit_log,
        )

    # -- Internal helpers -----------------------------------------------------

    def _evaluate_criterion(
        self,
        criterion: Criterion,
        evidence: EvidenceResult | None,
    ) -> CriterionVerdict:
        is_mandatory = criterion.status == CriterionStatus.MANDATORY

        # No evidence at all
        if evidence is None:
            return CriterionVerdict(
                criterion_id=criterion.id,
                criterion_description=criterion.description,
                criterion_type=criterion.type.value,
                is_mandatory=is_mandatory,
                verdict=VerdictLabel.NEEDS_REVIEW if is_mandatory else VerdictLabel.NOT_ELIGIBLE,
                reason=f"No evidence entry found for this criterion. Manual review required.",
                confidence=0.0,
                evidence_status="not_found",
            )

        # Document unreadable
        if evidence.status == ExtractionStatus.UNREADABLE:
            return CriterionVerdict(
                criterion_id=criterion.id,
                criterion_description=criterion.description,
                criterion_type=criterion.type.value,
                is_mandatory=is_mandatory,
                verdict=VerdictLabel.NEEDS_REVIEW,
                reason=(
                    f"Document '{evidence.source_doc}' could not be read clearly. "
                    "Manual review required."
                ),
                confidence=evidence.confidence,
                evidence_status=evidence.status.value,
                evidence_source_doc=evidence.source_doc,
            )

        # Not found
        if evidence.status == ExtractionStatus.NOT_FOUND:
            verdict = VerdictLabel.NOT_ELIGIBLE if is_mandatory else VerdictLabel.NOT_ELIGIBLE
            return CriterionVerdict(
                criterion_id=criterion.id,
                criterion_description=criterion.description,
                criterion_type=criterion.type.value,
                is_mandatory=is_mandatory,
                verdict=verdict,
                reason=(
                    f"Required evidence not found in submitted documents. "
                    f"Expected: {', '.join(criterion.verification_docs) or 'relevant document'}."
                ),
                confidence=evidence.confidence,
                evidence_status=evidence.status.value,
                evidence_source_doc=evidence.source_doc,
            )

        # Partial evidence
        if evidence.status == ExtractionStatus.PARTIAL:
            return CriterionVerdict(
                criterion_id=criterion.id,
                criterion_description=criterion.description,
                criterion_type=criterion.type.value,
                is_mandatory=is_mandatory,
                verdict=VerdictLabel.NEEDS_REVIEW,
                reason=(
                    f"Partial evidence found in '{evidence.source_doc}'"
                    + (f" page {evidence.source_page}" if evidence.source_page else "")
                    + f": \"{evidence.source_text[:120]}\". Incomplete — manual review required."
                ),
                confidence=evidence.confidence,
                evidence_status=evidence.status.value,
                evidence_value=evidence.found_value,
                evidence_source_doc=evidence.source_doc,
                evidence_source_page=evidence.source_page,
                evidence_source_text=evidence.source_text,
            )

        # Evidence found — check confidence first
        if evidence.confidence < CONFIDENCE_REVIEW_THRESHOLD:
            return CriterionVerdict(
                criterion_id=criterion.id,
                criterion_description=criterion.description,
                criterion_type=criterion.type.value,
                is_mandatory=is_mandatory,
                verdict=VerdictLabel.NEEDS_REVIEW,
                reason=(
                    f"Evidence found in '{evidence.source_doc}' but confidence is low "
                    f"({evidence.confidence:.0%}). Value: '{evidence.found_value}'. "
                    "Manual verification recommended."
                ),
                confidence=evidence.confidence,
                evidence_status=evidence.status.value,
                evidence_value=evidence.found_value,
                evidence_source_doc=evidence.source_doc,
                evidence_source_page=evidence.source_page,
                evidence_source_text=evidence.source_text,
            )

        # High-confidence found — apply threshold check if numeric
        if criterion.threshold_value is not None and evidence.found_value_numeric is not None:
            passed = _check_threshold(
                evidence.found_value_numeric,
                criterion.threshold_value,
                criterion.threshold_operator or ">=",
            )
            if passed:
                return CriterionVerdict(
                    criterion_id=criterion.id,
                    criterion_description=criterion.description,
                    criterion_type=criterion.type.value,
                    is_mandatory=is_mandatory,
                    verdict=VerdictLabel.ELIGIBLE,
                    reason=(
                        f"Criterion satisfied. Found {evidence.found_value} "
                        f"(required {criterion.threshold_operator} {criterion.threshold_value} "
                        f"{criterion.threshold_unit or ''}). "
                        f"Source: '{evidence.source_doc}'"
                        + (f" page {evidence.source_page}" if evidence.source_page else "")
                        + f": \"{evidence.source_text[:120]}\"."
                    ),
                    confidence=evidence.confidence,
                    evidence_status=evidence.status.value,
                    evidence_value=evidence.found_value,
                    evidence_source_doc=evidence.source_doc,
                    evidence_source_page=evidence.source_page,
                    evidence_source_text=evidence.source_text,
                )
            else:
                return CriterionVerdict(
                    criterion_id=criterion.id,
                    criterion_description=criterion.description,
                    criterion_type=criterion.type.value,
                    is_mandatory=is_mandatory,
                    verdict=VerdictLabel.NOT_ELIGIBLE,
                    reason=(
                        f"Criterion not met. Found {evidence.found_value} "
                        f"but required {criterion.threshold_operator} {criterion.threshold_value} "
                        f"{criterion.threshold_unit or ''}. "
                        f"Source: '{evidence.source_doc}'"
                        + (f" page {evidence.source_page}" if evidence.source_page else "")
                        + f": \"{evidence.source_text[:120]}\"."
                    ),
                    confidence=evidence.confidence,
                    evidence_status=evidence.status.value,
                    evidence_value=evidence.found_value,
                    evidence_source_doc=evidence.source_doc,
                    evidence_source_page=evidence.source_page,
                    evidence_source_text=evidence.source_text,
                )

        # Non-numeric criterion, evidence found at high confidence → ELIGIBLE
        return CriterionVerdict(
            criterion_id=criterion.id,
            criterion_description=criterion.description,
            criterion_type=criterion.type.value,
            is_mandatory=is_mandatory,
            verdict=VerdictLabel.ELIGIBLE,
            reason=(
                f"Evidence found in '{evidence.source_doc}'"
                + (f" page {evidence.source_page}" if evidence.source_page else "")
                + f": \"{evidence.source_text[:120]}\"."
            ),
            confidence=evidence.confidence,
            evidence_status=evidence.status.value,
            evidence_value=evidence.found_value,
            evidence_source_doc=evidence.source_doc,
            evidence_source_page=evidence.source_page,
            evidence_source_text=evidence.source_text,
        )

    def _compute_overall(
        self,
        criterion_verdicts: list[CriterionVerdict],
    ) -> tuple[VerdictLabel, str]:
        mandatory = [cv for cv in criterion_verdicts if cv.is_mandatory]

        failed = [cv for cv in mandatory if cv.verdict == VerdictLabel.NOT_ELIGIBLE]
        if failed:
            ids = ", ".join(cv.criterion_id for cv in failed)
            return (
                VerdictLabel.NOT_ELIGIBLE,
                f"Bidder does not meet mandatory criteria: {ids}. "
                + " | ".join(f"{cv.criterion_id}: {cv.reason}" for cv in failed),
            )

        reviews = [cv for cv in mandatory if cv.verdict == VerdictLabel.NEEDS_REVIEW]
        if reviews:
            ids = ", ".join(cv.criterion_id for cv in reviews)
            return (
                VerdictLabel.NEEDS_REVIEW,
                f"Mandatory criteria require human review: {ids}. "
                + " | ".join(f"{cv.criterion_id}: {cv.reason}" for cv in reviews),
            )

        return (
            VerdictLabel.ELIGIBLE,
            f"All {len(mandatory)} mandatory criteria satisfied.",
        )


def _check_threshold(value: float, threshold: float, operator: str) -> bool:
    tol = abs(threshold) * NUMERIC_TOLERANCE
    if operator == ">=":
        return value >= threshold - tol
    elif operator == ">":
        return value > threshold - tol
    elif operator == "<=":
        return value <= threshold + tol
    elif operator == "<":
        return value < threshold + tol
    elif operator == "=":
        return abs(value - threshold) <= tol
    return False