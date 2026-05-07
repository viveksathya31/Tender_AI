"""
Prompt templates for Phase 3 — Bidder Evidence Extraction.
"""

SYSTEM_PROMPT = """You are an expert government procurement evaluator specialising in Indian public tenders.

Your job is to read a bidder's submitted documents and find evidence for specific eligibility criteria.

RULES:
1. For each criterion, search the document text carefully and extract the most relevant evidence.
2. Copy the EXACT text snippet from the document that supports your finding into source_text.
3. If a numeric value is found (turnover, years, project count), extract it as a number into found_value_numeric.
4. Be conservative — only mark status as "found" if you are confident the evidence clearly satisfies the criterion.
5. If evidence is present but ambiguous or incomplete, mark status as "partial".
6. If no relevant content is found at all, mark status as "not_found".
7. If the document appears degraded or unreadable for a section, mark status as "unreadable".
8. Never invent evidence. If it is not in the document text, it is not found.
9. Note any discrepancies, unusual formats, or concerns in the notes field.
10. Assign a confidence score (0.0–1.0) for how certain you are about the extraction.

OUTPUT FORMAT — return ONLY valid JSON, no preamble, no markdown:
{
  "evidence": [
    {
      "criterion_id": "C001",
      "criterion_description": "Minimum annual turnover of Rs. 5 crore",
      "status": "found|not_found|partial|unreadable",
      "found_value": "Rs. 7.2 crore",
      "found_value_numeric": 72000000,
      "found_unit": "INR",
      "source_doc": "financial_statement.pdf",
      "source_page": 3,
      "source_text": "Total Annual Turnover: Rs. 7,20,00,000",
      "confidence": 0.93,
      "notes": ""
    }
  ],
  "extraction_warnings": []
}

found_value_numeric should be in base units (INR not crore, years as integer, etc.).
Return null for found_value, found_value_numeric, found_unit if status is not_found or unreadable.
"""


def build_evidence_prompt(
    criteria_json: str,
    bidder_text: str,
    bidder_id: str,
    doc_filename: str,
) -> str:
    return f"""Extract evidence for the following eligibility criteria from this bidder's document.

BIDDER ID: {bidder_id}
DOCUMENT: {doc_filename}

CRITERIA TO CHECK (JSON):
{criteria_json}

BIDDER DOCUMENT TEXT:
---
{bidder_text}
---

For each criterion above, find the relevant evidence in the document text.
Return ONLY the JSON object. No explanation, no markdown.
"""


def build_merge_prompt(all_evidence_json: str, bidder_id: str) -> str:
    """
    When a bidder submits multiple documents, evidence for the same criterion
    may appear across different docs. This prompt merges them, keeping the
    strongest evidence per criterion.
    """
    return f"""A bidder has submitted multiple documents. Evidence for the same criterion
has been extracted from each document separately and may contain duplicates or complementary findings.

BIDDER ID: {bidder_id}

YOUR TASK:
1. For each criterion_id, keep the single best evidence entry:
   - Prefer "found" over "partial" over "not_found"
   - Among entries with the same status, prefer higher confidence
   - If two docs provide complementary evidence, merge notes and keep the higher-confidence source
2. Preserve all source_doc and source_text references accurately.
3. Return the merged list in the same JSON format.

ALL EXTRACTED EVIDENCE (from all documents):
{all_evidence_json}

Return ONLY valid JSON with the merged evidence list. No preamble, no markdown.
"""