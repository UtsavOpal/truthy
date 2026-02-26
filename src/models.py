"""
Data models for the Hallucination Detection Framework.
Taxonomy updated to simple 1/2/3/4 numbering.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class HallucinationType(str, Enum):
    """
    Hallucination taxonomy — simple 1/2/3/4 numbering.

    1  – Out-of-Context Entity Hallucination
    2  – Tuple Verification Hallucination
    3  – Out-of-Context Intent Hallucination
    4  – Triple Verification Hallucination
    """
    ENTITY_OUT_OF_CONTEXT = "1"
    ENTITY_TUPLE          = "2"
    INTENT_OUT_OF_CONTEXT = "3"
    SEMANTIC_TRIPLE       = "4"

    @property
    def display_name(self) -> str:
        return {
            "1": "Out-of-Context Entity Hallucination",
            "2": "Tuple Verification Hallucination",
            "3": "Out-of-Context Intent Hallucination",
            "4": "Triple Verification Hallucination",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "1": "Answer introduces an entity not present in / inferable from the paragraph",
            "2": "Real entities exist but are incorrectly paired or linked together",
            "3": "Entities are correct but the verb / action / relationship is wrong or inverted",
            "4": "The full subject–predicate–object triple is wrong at every structural level",
        }[self.value]

    @property
    def icon(self) -> str:
        return {"1": "⚡", "2": "🔗", "3": "🎯", "4": "🔺"}[self.value]


@dataclass
class DetectionInput:
    """Input to the hallucination detector."""
    question:  str
    answer:    str
    paragraph: str = ""

    def summary(self) -> str:
        lines = []
        if self.paragraph:
            short = self.paragraph[:120] + ("…" if len(self.paragraph) > 120 else "")
            lines.append(f"  Paragraph : {short}")
        else:
            lines.append("  Paragraph : (none – world knowledge used)")
        lines.append(f"  Question  : {self.question}")
        lines.append(f"  Answer    : {self.answer}")
        return "\n".join(lines)


@dataclass
class HallucinationResult:
    """Full result returned by the detector."""
    is_hallucinated:       bool
    confidence:            int
    hallucination_types:   List[HallucinationType] = field(default_factory=list)
    hallucinated_elements: List[str]               = field(default_factory=list)
    explanation:           str  = ""
    correct_answer:        str  = ""
    raw_response:          Optional[dict] = field(default=None, repr=False)

    @property
    def verdict(self) -> str:
        return "HALLUCINATED" if self.is_hallucinated else "CLEAN"

    @property
    def type_codes(self) -> List[str]:
        return [t.value for t in self.hallucination_types]
