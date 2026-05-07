"""
Phase 5 — Report Generator + Audit Trail.

Produces:
  1. A structured JSON report (ConsolidatedReport) for the API
  2. A human-readable Markdown report for export/sign-off
  3. An append-only audit log entry for every decision

Audit design:
  - Every automated verdict is logged with timestamp, model, inputs, and output
  - Human overrides are logged separately with officer ID and reason
  - The log is returned as part of the report for full traceability
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from models.verdict import BidderVerdict, ConsolidatedReport, VerdictLabel

logger = logging.getLogger(__name__)


class ReportGenerator:

    def generate_markdown(self, report: ConsolidatedReport) -> str:
        """
        Produce a human-readable Markdown evaluation report
        suitable for officer sign-off.
        """
        lines = []
        ts = report.generated_at.strftime("%d %b %Y, %H:%M UTC")

        lines += [
            f"# Tender Evaluation Report",
            f"",
            f"**Tender:** {report.tender_filename}",
            f"**Generated:** {ts}",
            f"**Total Bidders:** {report.total_bidders}",
            f"",
            f"## Summary",
            f"",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| ✅ Eligible | {report.eligible_count} |",
            f"| ❌ Not Eligible | {report.not_eligible_count} |",
            f"| ⚠️ Needs Review | {report.needs_review_count} |",
            f"",
            f"## Criteria Summary",
            f"",
        ]

        for cs in report.criteria_summary:
            mandatory_tag = "*(mandatory)*" if cs["mandatory"] else "*(optional)*"
            lines.append(
                f"- **{cs['criterion_id']}** {mandatory_tag}: {cs['description']}  "
                f"— ✅ {cs['eligible']} | ❌ {cs['not_eligible']} | ⚠️ {cs['needs_review']}"
            )

        lines += ["", "---", "", "## Bidder Evaluations", ""]

        for bv in report.bidder_verdicts:
            icon = {"eligible": "✅", "not_eligible": "❌", "needs_review": "⚠️"}.get(
                bv.overall_verdict.value, "?"
            )
            lines += [
                f"### {icon} Bidder: {bv.bidder_id}",
                f"",
                f"**Overall:** {bv.overall_verdict.value.replace('_', ' ').title()}",
                f"",
                f"> {bv.overall_reason}",
                f"",
                f"| Criterion | Type | Mandatory | Verdict | Evidence | Confidence |",
                f"|-----------|------|-----------|---------|----------|------------|",
            ]
            for cv in bv.criterion_verdicts:
                v_icon = {"eligible": "✅", "not_eligible": "❌", "needs_review": "⚠️"}.get(
                    cv.verdict.value, "?"
                )
                mandatory = "Yes" if cv.is_mandatory else "No"
                source = cv.evidence_source_doc or "—"
                if cv.evidence_source_page:
                    source += f" p.{cv.evidence_source_page}"
                lines.append(
                    f"| {cv.criterion_id} | {cv.criterion_type} | {mandatory} "
                    f"| {v_icon} {cv.verdict.value.replace('_',' ').title()} "
                    f"| {source} | {cv.confidence:.0%} |"
                )
            lines += [
                f"",
                f"**Criterion Details:**",
                f"",
            ]
            for cv in bv.criterion_verdicts:
                lines += [
                    f"- **{cv.criterion_id}** — {cv.reason}",
                ]
            lines += ["", "---", ""]

        lines += [
            f"## Audit Trail",
            f"",
            f"This report was generated automatically. All decisions are logged in the audit trail.",
            f"Human review is required for all ⚠️ cases before a final procurement decision is made.",
            f"",
            f"*Officer signature: ___________________    Date: ___________*",
        ]

        return "\n".join(lines)

    def create_audit_entry(
        self,
        event_type: str,
        actor: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create a single immutable audit log entry.
        Each entry is hashed for tamper detection.
        """
        entry = {
            "event_type": event_type,     # e.g. "criteria_extracted", "verdict_computed", "human_override"
            "actor": actor,               # e.g. "claude-sonnet-4-20250514", "officer_001"
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        # Hash for tamper detection (chain-of-custody)
        entry["hash"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return entry

    def log_verdict(self, bidder_verdict: BidderVerdict, model: str) -> dict[str, Any]:
        """Create an audit entry for an automated verdict."""
        return self.create_audit_entry(
            event_type="verdict_computed",
            actor=model,
            data={
                "bidder_id": bidder_verdict.bidder_id,
                "overall_verdict": bidder_verdict.overall_verdict.value,
                "mandatory_passed": bidder_verdict.mandatory_passed,
                "mandatory_failed": bidder_verdict.mandatory_failed,
                "mandatory_review": bidder_verdict.mandatory_review,
                "criterion_verdicts": [
                    {
                        "criterion_id": cv.criterion_id,
                        "verdict": cv.verdict.value,
                        "confidence": cv.confidence,
                        "evidence_source": cv.evidence_source_doc,
                    }
                    for cv in bidder_verdict.criterion_verdicts
                ],
            },
        )

    def log_human_override(
        self,
        bidder_id: str,
        criterion_id: str,
        old_verdict: str,
        new_verdict: str,
        officer_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Create an audit entry when a human officer overrides an automated verdict."""
        return self.create_audit_entry(
            event_type="human_override",
            actor=officer_id,
            data={
                "bidder_id": bidder_id,
                "criterion_id": criterion_id,
                "old_verdict": old_verdict,
                "new_verdict": new_verdict,
                "reason": reason,
            },
        )