"""
Data models for the Hallucination Detection Framework.
Definitions aligned with the project specification document.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class HallucinationType(str, Enum):
    """
    Hallucination taxonomy — definitions from the specification document.

    1A  – Out-of-Context Entity Hallucination
          The answer introduces an entity that does NOT appear in the paragraph
          and cannot be logically inferred from it. The entity may be real-world
          valid, but it is not grounded in the provided context.
          Example: paragraph has Nolan as director → answer says Spielberg.
                   "Spielberg" is not in the paragraph = 1A.

    1B  – Tuple Verification Hallucination
          Entities individually exist and may even be real, but their
          RELATIONSHIP or PAIRING is incorrect. The issue is wrong linking,
          not fabricated entities.
          Example: Apple's founders listed wrong — Bill Gates & Paul Allen
                   (Microsoft founders) incorrectly paired with Apple = 1B.
          Example (no paragraph): Brazil incorrectly paired with
                   "2018 World Cup winner" attribute (France won) = 1B.

    2A  – Out-of-Context Intent Hallucination
          The entities are correct, but the ACTION, VERB, or RELATIONSHIP
          associated with them is wrong or inverted. Predicate distortion.
          Example: 1984 "promotes surveillance" when paragraph says it
                   critiques/depicts surveillance as oppressive = 2A.

    3A  – Triple Verification Hallucination
          The FULL semantic triple (subject + predicate + object) is wrong.
          No part of the triple aligns with the paragraph. Most severe type.
          Example: monument built by Person A for Reason X → answer says
                   Person B built it for Reason Y. Entire triple wrong = 3A.
    """
    ENTITY_OUT_OF_CONTEXT = "1A"
    ENTITY_TUPLE          = "1B"
    INTENT_OUT_OF_CONTEXT = "2A"
    SEMANTIC_TRIPLE       = "3A"

    @property
    def display_name(self) -> str:
        return {
            "1A": "Entity – Out-of-Context Entity Hallucination",
            "1B": "Entity – Tuple Verification Hallucination",
            "2A": "Intent – Out-of-Context Intent Hallucination",
            "3A": "Semantic – Triple Verification Hallucination",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "1A": "Answer introduces an entity not present in / inferable from the paragraph",
            "1B": "Real entities exist but are incorrectly paired or linked together",
            "2A": "Entities are correct but the verb / action / relationship is wrong or inverted",
            "3A": "The full subject–predicate–object triple is wrong at every structural level",
        }[self.value]

    @property
    def icon(self) -> str:
        return {"1A": "⚡", "1B": "🔗", "2A": "🎯", "3A": "🔺"}[self.value]


@dataclass
class DetectionInput:
    """Input to the hallucination detector."""
    question:  str
    answer:    str
    paragraph: str = ""   # Optional — leave blank to use world knowledge

    def summary(self) -> str:
        lines = []
        if self.paragraph:
            short = self.paragraph[:120] + ("…" if len(self.paragraph) > 120 else "")
            lines.append(f"  Paragraph : {short}")
        else:
            lines.append("  Paragraph : (none – world knowledge used as ground truth)")
        lines.append(f"  Question  : {self.question}")
        lines.append(f"  Answer    : {self.answer}")
        return "\n".join(lines)


@dataclass
class HallucinationResult:
    """Full result returned by the detector."""
    is_hallucinated:      bool
    confidence:           int                    # 0–100
    hallucination_types:  List[HallucinationType] = field(default_factory=list)
    hallucinated_elements: List[str]             = field(default_factory=list)
    explanation:          str  = ""
    correct_answer:       str  = ""
    raw_response:         Optional[dict] = field(default=None, repr=False)

    @property
    def verdict(self) -> str:
        return "HALLUCINATED" if self.is_hallucinated else "CLEAN"

    @property
    def type_codes(self) -> List[str]:
        return [t.value for t in self.hallucination_types]
