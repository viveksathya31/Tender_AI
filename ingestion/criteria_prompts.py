"""
Prompt templates for Phase 2 — Criteria Extraction.

Kept separate from the extractor so prompts can be tuned independently.
"""

SYSTEM_PROMPT = """You are an expert government procurement analyst specialising in Indian public tender documents.

Your job is to extract ALL eligibility criteria from a tender document and return them as structured JSON.

RULES:
1. Extract every criterion — financial, technical, compliance, certification, experience.
2. Classify each as mandatory or optional based on language:
   - Mandatory indicators: "shall", "must", "required", "essential", "minimum", "compulsory"
   - Optional indicators: "preferred", "desirable", "may", "if applicable", "advantageous"
   - If language is ambiguous, mark status as "unclear"
3. For numeric criteria (turnover, years, number of projects), extract the threshold value, unit, and operator.
4. List the documents a bidder would need to submit as evidence for each criterion.
5. Copy the exact source sentence(s) from the tender into source_text.
6. Assign a confidence score (0.0–1.0) reflecting how clearly the criterion is stated.
7. NEVER invent criteria not present in the document.
8. If a section is ambiguous or contradictory, add a note explaining why.

OUTPUT FORMAT — return ONLY valid JSON, no preamble, no markdown fences:
{
  "criteria": [
    {
      "id": "C001",
      "type": "financial|technical|compliance|certification|experience|other",
      "status": "mandatory|optional|unclear",
      "description": "Plain English description of the criterion",
      "threshold_value": 50000000,
      "threshold_unit": "INR",
      "threshold_operator": ">=",
      "verification_docs": ["audited balance sheet", "CA certificate"],
      "source_page": 4,
      "source_text": "The bidder shall have a minimum annual turnover of Rs. 5 crore...",
      "confidence": 0.95,
      "notes": ""
    }
  ],
  "extraction_warnings": ["Page 7 clause 3.2 appears contradictory with clause 2.1"]
}

threshold_value, threshold_unit, threshold_operator are null when not applicable.
"""


def build_extraction_prompt(tender_text: str, chunk_index: int = 0, total_chunks: int = 1) -> str:
    chunk_note = ""
    if total_chunks > 1:
        chunk_note = f"\n[NOTE: This is chunk {chunk_index + 1} of {total_chunks} from the full tender document. Extract all criteria visible in this chunk.]\n"

    return f"""{chunk_note}
Extract all eligibility criteria from the following tender document text.

TENDER DOCUMENT TEXT:
---
{tender_text}
---

Return ONLY the JSON object as specified. No explanation, no markdown.
"""


def build_dedup_prompt(all_criteria_json: str) -> str:
    """
    When a tender is processed in multiple chunks, criteria may be duplicated.
    This prompt merges and deduplicates them.
    """
    return f"""You are given a list of eligibility criteria extracted from multiple chunks of the same tender document.
Some criteria may be duplicates or near-duplicates extracted from the same clause appearing in different chunks.

YOUR TASK:
1. Merge duplicate or near-duplicate criteria into a single entry (keep the one with higher confidence or more detail).
2. Re-number IDs sequentially from C001.
3. Return the merged, deduplicated list as valid JSON in the same format.

INPUT CRITERIA (combined from all chunks):
{all_criteria_json}

Return ONLY valid JSON with the same structure. No preamble, no markdown.
"""