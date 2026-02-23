"""
Hallucination detection engine.
Three modes: Free (rule-based), OpenAI (GPT-4o), Gemini (1.5 Pro).
"""

import json, re
from src.models import DetectionInput, HallucinationResult, HallucinationType

# ── Shared system prompt for AI backends ──────────────────────────────────────
SYSTEM_PROMPT = """You are an expert hallucination detection engine for LLM outputs.

Given a PARAGRAPH (context), a QUESTION, and a MODEL'S ANSWER, your job is to:
  1. Decide whether the answer is hallucinated.
  2. If yes, classify into EXACTLY ONE type.

TAXONOMY:
TYPE 1A — Out-of-Context Entity: answer introduces an entity NOT in the paragraph and not inferable.
TYPE 1B — Tuple Verification: real entities exist but their pairing/relationship is wrong.
TYPE 2A — Out-of-Context Intent: correct entities, but verb/action/relationship is distorted or inverted.
TYPE 3A — Triple Verification: entire subject-predicate-object triple is wrong at every level.

RULES:
1. If paragraph provided → ground against it. No paragraph → use world knowledge.
2. EXACTLY ONE type. Priority: 1A → 1B → 2A → 3A.
3. Correct extra facts = NOT hallucinated. Wrong/contradicting facts = hallucinated.
4. Contradictory answer → classify on the wrong claim only.

OUTPUT: Valid JSON only. No markdown.
{
  "is_hallucinated": true|false,
  "confidence": <0-100>,
  "hallucination_types": ["1A"],
  "hallucinated_elements": ["specific wrong element"],
  "explanation": "Precise explanation of what is wrong and why.",
  "correct_answer": "What the answer should have said"
}
HARD CONSTRAINT: hallucination_types must have AT MOST ONE value.
"""

def _parse_raw(raw: dict) -> HallucinationResult:
    type_map = {t.value: t for t in HallucinationType}
    h_types  = [type_map[c] for c in raw.get("hallucination_types", []) if c in type_map]
    if len(h_types) > 1:
        h_types = h_types[:1]
    return HallucinationResult(
        is_hallucinated       = bool(raw.get("is_hallucinated", False)),
        confidence            = int(raw.get("confidence", 0)),
        hallucination_types   = h_types,
        hallucinated_elements = raw.get("hallucinated_elements", []),
        explanation           = raw.get("explanation", ""),
        correct_answer        = raw.get("correct_answer", ""),
        raw_response          = raw,
    )

def _build_user_message(inp: DetectionInput) -> str:
    para = inp.paragraph.strip() or "(No paragraph — use world knowledge)"
    return f"PARAGRAPH:\n{para}\n\nQUESTION:\n{inp.question.strip()}\n\nMODEL'S ANSWER:\n{inp.answer.strip()}\n\nAnalyze and return JSON."

def _clean_json(text: str) -> dict:
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ── Free (rule-based) detector ────────────────────────────────────────────────
class FreeDetector:
    """
    Lightweight heuristic detector. No API key required.
    Uses token overlap and simple NLP rules. Less accurate than AI backends.
    """
    def detect(self, inp: DetectionInput) -> HallucinationResult:
        import difflib, string

        def tokens(t):
            t = t.lower().translate(str.maketrans('', '', string.punctuation))
            return set(t.split())

        para_tokens   = tokens(inp.paragraph)
        answer_tokens = tokens(inp.answer)
        question_tokens = tokens(inp.question)

        is_hallucinated = False
        confidence      = 50
        h_type          = None
        hallucinated_elements = []
        explanation     = ""
        correct_answer  = ""

        if inp.paragraph.strip():
            # Compute overlap between answer and paragraph
            overlap = len(answer_tokens & para_tokens) / max(len(answer_tokens), 1)

            # New tokens in answer not in paragraph (potential new entities)
            # Filter stopwords roughly
            stopwords = {"the","a","an","is","was","are","were","be","been","being",
                         "have","has","had","do","does","did","will","would","shall",
                         "should","may","might","must","can","could","to","of","in",
                         "for","on","with","at","by","from","up","about","into","through",
                         "during","it","its","this","that","these","those","and","but",
                         "or","nor","not","so","yet","both","either","neither","because",
                         "as","if","then","than","when","where","who","which","what",
                         "he","she","they","we","you","i","me","him","her","us","them"}

            new_answer_tokens = (answer_tokens - para_tokens - question_tokens) - stopwords

            if overlap < 0.25 and len(new_answer_tokens) > 2:
                is_hallucinated = True
                h_type          = "1A"
                confidence      = 65
                hallucinated_elements = list(new_answer_tokens)[:3]
                explanation     = (
                    f"The answer contains terms not found in the paragraph "
                    f"({', '.join(list(new_answer_tokens)[:3])}), suggesting out-of-context entity introduction. "
                    f"Token overlap with paragraph is only {overlap:.0%}."
                )
                correct_answer  = "Answer should be grounded in the provided paragraph."
            elif overlap >= 0.25:
                # Check for intent inversion signals
                inversion_pairs = [
                    ({"promotes","supports","endorses","advocates","favors"},
                     {"critiques","opposes","criticizes","condemns","depicts"}),
                    ({"invented","created"},{"discovered","found"}),
                    ({"won","victory","champion"},{"lost","defeated","runner"}),
                ]
                para_l  = inp.paragraph.lower()
                ans_l   = inp.answer.lower()
                for pos_words, neg_words in inversion_pairs:
                    ans_has_pos  = any(w in ans_l  for w in pos_words)
                    para_has_neg = any(w in para_l for w in neg_words)
                    ans_has_neg  = any(w in ans_l  for w in neg_words)
                    para_has_pos = any(w in para_l for w in pos_words)
                    if (ans_has_pos and para_has_neg) or (ans_has_neg and para_has_pos):
                        is_hallucinated = True
                        h_type          = "2A"
                        confidence      = 72
                        explanation     = "Detected possible intent inversion: the answer uses a verb/relationship that contradicts the paragraph's framing."
                        correct_answer  = "The answer's relationship should align with the paragraph."
                        break
                if not is_hallucinated:
                    confidence = 82

        else:
            # No paragraph — limited heuristics
            confidence = 45
            explanation = "No paragraph provided. Free mode uses limited heuristics without a reference paragraph. Use AI mode for world-knowledge grounding."

        if not is_hallucinated:
            return HallucinationResult(
                is_hallucinated=False, confidence=confidence,
                hallucination_types=[], hallucinated_elements=[],
                explanation=explanation or "No clear hallucination signals detected by rule-based analysis.",
                correct_answer="",
            )

        type_map = {t.value: t for t in HallucinationType}
        return HallucinationResult(
            is_hallucinated       = True,
            confidence            = confidence,
            hallucination_types   = [type_map[h_type]] if h_type in type_map else [],
            hallucinated_elements = hallucinated_elements,
            explanation           = explanation,
            correct_answer        = correct_answer,
        )


# ── OpenAI backend ────────────────────────────────────────────────────────────
class OpenAIDetector:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        r = self.client.chat.completions.create(
            model="gpt-4o", max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(inp)},
            ]
        )
        return _parse_raw(_clean_json(r.choices[0].message.content))


# ── Gemini backend ────────────────────────────────────────────────────────────
class GeminiDetector:
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json", "max_output_tokens": 1024},
        )
        return _parse_raw(_clean_json(model.generate_content(_build_user_message(inp)).text))


# ── Unified entry point ───────────────────────────────────────────────────────
class HallucinationDetector:
    def __init__(self, provider: str = "free", api_key: str = ""):
        provider = provider.lower()
        if provider == "free":
            self._backend = FreeDetector()
        elif provider == "openai":
            self._backend = OpenAIDetector(api_key)
        elif provider == "gemini":
            self._backend = GeminiDetector(api_key)
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use: free, openai, gemini.")

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        return self._backend.detect(inp)
