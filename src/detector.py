"""
Hallucination detection engine  v7.0
- Free mode  : Groq API (Llama 3.1 8B Instant — free 14k req/day) → heuristic fallback
- OpenAI     : GPT-4o
- Gemini     : Gemini 1.5 Pro
- Language   : All backends respond in the specified language
Taxonomy uses simple 1 / 2 / 3 / 4 numbering.
"""

import json, re, os, urllib.request
from src.models import DetectionInput, HallucinationResult, HallucinationType

SYSTEM_PROMPT_TEMPLATE = """You are an expert hallucination detection engine for LLM outputs.
You MUST respond entirely in {language}. All fields — explanation, correct_answer — must be in {language}.

Given a PARAGRAPH (context), a QUESTION, and a MODEL'S ANSWER, your job is to:
  1. Decide whether the answer is hallucinated.
  2. If yes, classify into EXACTLY ONE type.
  3. If hallucinated AND web context is available, derive the correct answer from the context.

TAXONOMY:
TYPE 1 — Out-of-Context Entity: answer introduces an entity NOT in the paragraph and not inferable.
TYPE 2 — Tuple Verification: real entities exist but their pairing/relationship is wrong.
TYPE 3 — Out-of-Context Intent: correct entities, but verb/action/relationship is distorted or inverted.
TYPE 4 — Triple Verification: entire subject-predicate-object triple is wrong at every level.

RULES:
1. If paragraph provided → ground strictly against it. No paragraph → use world knowledge.
2. Pick EXACTLY ONE type. Priority: 1 → 2 → 3 → 4.
3. Correct extra facts = NOT hallucinated. Wrong/contradicting facts = hallucinated.
4. Be decisive. High confidence when evidence is clear.
5. If hallucinated, always provide a correct_answer based on the paragraph/context.
6. Write explanation and correct_answer in {language}.

OUTPUT: Valid JSON only. No markdown. No extra text.
{{
  "is_hallucinated": true|false,
  "confidence": <0-100>,
  "hallucination_types": ["1"],
  "hallucinated_elements": ["specific wrong element"],
  "explanation": "Precise explanation of what is wrong and why. (in {language})",
  "correct_answer": "What the correct answer should be, based on the context. (in {language})"
}}
HARD CONSTRAINT: hallucination_types must contain AT MOST ONE value from: "1", "2", "3", "4".
"""


def _get_system_prompt(language: str = "English") -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(language=language)


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
    return (
        f"PARAGRAPH:\n{para}\n\n"
        f"QUESTION:\n{inp.question.strip()}\n\n"
        f"MODEL'S ANSWER:\n{inp.answer.strip()}\n\n"
        f"Analyze and return JSON."
    )


def _clean_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# ── FREE MODE — Groq (Llama 3.1 8B Instant) ──────────────────────────────────
class FreeDetector:
    """
    Primary  : Groq API — Llama 3.1 8B Instant (free tier)
               14,400 requests/day | 500,000 tokens/day
               Get key: https://console.groq.com
               Set env: GROQ_API_KEY=gsk_...
    Fallback : Improved heuristics (no key, lower accuracy)
    """
    GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama-3.1-8b-instant"

    def __init__(self, language: str = "English"):
        self.language = language

    def _groq_detect(self, inp: DetectionInput) -> HallucinationResult:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        payload = json.dumps({
            "model": self.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": _get_system_prompt(self.language)},
                {"role": "user",   "content": _build_user_message(inp)},
            ],
            "temperature": 0.1,
            "max_tokens":  600,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            self.GROQ_URL, data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent":    "TruthyDetector/7.0",
            },
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            resp = json.loads(r.read())
        return _parse_raw(_clean_json(resp["choices"][0]["message"]["content"]))

    def _heuristic_fallback(self, inp: DetectionInput) -> HallucinationResult:
        import string

        def tokens(t):
            t = t.lower().translate(str.maketrans('', '', string.punctuation))
            return set(t.split())

        STOPWORDS = {
            "the","a","an","is","was","are","were","be","been","being","have","has",
            "had","do","does","did","will","would","shall","should","may","might",
            "must","can","could","to","of","in","for","on","with","at","by","from",
            "up","about","into","through","during","it","its","this","that","these",
            "those","and","but","or","nor","not","so","yet","both","either","neither",
            "because","as","if","then","than","when","where","who","which","what",
            "he","she","they","we","you","i","me","him","her","us","them","also",
        }

        para_tok = tokens(inp.paragraph) - STOPWORDS
        ans_tok  = tokens(inp.answer)    - STOPWORDS
        q_tok    = tokens(inp.question)  - STOPWORDS

        is_hallucinated = False
        confidence      = 50
        h_type          = None
        hallucinated_elements = []
        explanation     = ""
        correct_answer  = ""

        if inp.paragraph.strip():
            overlap    = len(ans_tok & para_tok) / max(len(ans_tok), 1)
            caps_ans   = {w for w in inp.answer.split()    if w[0].isupper() and len(w) > 2}
            caps_para  = {w for w in inp.paragraph.split() if w[0].isupper() and len(w) > 2}
            caps_q     = {w for w in inp.question.split()  if w[0].isupper() and len(w) > 2}
            novel_caps = caps_ans - caps_para - caps_q

            if novel_caps and overlap < 0.35:
                is_hallucinated       = True
                h_type                = "1"
                confidence            = 72
                hallucinated_elements = list(novel_caps)[:4]
                explanation           = (
                    f"Answer introduces named entities not found in context: "
                    f"{', '.join(list(novel_caps)[:3])}. "
                    f"Token overlap with paragraph: {overlap:.0%}."
                )
                correct_answer = "Answer should be grounded in the provided paragraph."
            else:
                inversion_pairs = [
                    ({"promotes","supports","endorses","advocates","favors"},
                     {"critiques","opposes","criticizes","condemns","depicts","warns"}),
                    ({"won","victory","champion","defeated","beat"},
                     {"lost","surrendered","conceded"}),
                    ({"invented","created","founded","built"},
                     {"discovered","found","explored"}),
                ]
                para_l, ans_l = inp.paragraph.lower(), inp.answer.lower()
                for pos_w, neg_w in inversion_pairs:
                    if (any(w in ans_l for w in pos_w) and any(w in para_l for w in neg_w)) or \
                       (any(w in ans_l for w in neg_w) and any(w in para_l for w in pos_w)):
                        is_hallucinated = True
                        h_type          = "3"
                        confidence      = 70
                        explanation     = "Answer intent contradicts paragraph framing — predicate inversion detected."
                        correct_answer  = "Answer relationship should align with paragraph intent."
                        break

                if not is_hallucinated:
                    confidence  = 80
                    explanation = "No clear hallucination signals detected by rule-based analysis."
        else:
            confidence  = 40
            explanation = "GROQ_API_KEY not configured — using heuristic fallback. Set GROQ_API_KEY for Llama 3.1 8B detection."

        if not is_hallucinated:
            return HallucinationResult(
                is_hallucinated=False, confidence=confidence,
                hallucination_types=[], hallucinated_elements=[],
                explanation=explanation, correct_answer="",
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

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        try:
            return self._groq_detect(inp)
        except Exception:
            return self._heuristic_fallback(inp)


# ── OpenAI — GPT-4o ──────────────────────────────────────────────────────────
class OpenAIDetector:
    def __init__(self, api_key: str, language: str = "English"):
        from openai import OpenAI
        self.client   = OpenAI(api_key=api_key)
        self.language = language

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        r = self.client.chat.completions.create(
            model="gpt-4o", max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _get_system_prompt(self.language)},
                {"role": "user",   "content": _build_user_message(inp)},
            ]
        )
        return _parse_raw(_clean_json(r.choices[0].message.content))


# ── Gemini — Gemini 1.5 Pro ───────────────────────────────────────────────────
class GeminiDetector:
    def __init__(self, api_key: str, language: str = "English"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.language = language

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=_get_system_prompt(self.language),
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 1024,
                "temperature": 0.1,
            },
        )
        return _parse_raw(_clean_json(model.generate_content(_build_user_message(inp)).text))


# ── Unified entry point ───────────────────────────────────────────────────────
class HallucinationDetector:
    def __init__(self, provider: str = "free", api_key: str = "", language: str = "English"):
        provider = provider.lower()
        if provider == "free":
            self._backend = FreeDetector(language=language)
        elif provider == "openai":
            self._backend = OpenAIDetector(api_key, language=language)
        elif provider == "gemini":
            self._backend = GeminiDetector(api_key, language=language)
        else:
            raise ValueError(f"Unknown provider '{provider}'.")

    def detect(self, inp: DetectionInput) -> HallucinationResult:
        return self._backend.detect(inp)
