"""
Hallucination Detector — FastAPI Application
Web UI + REST API. Three modes: free (no key), openai, gemini.
Web search is automatically used when no context paragraph is provided.
"""
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import pathlib
import urllib.request
import urllib.parse
import json
import re

from src.detector import HallucinationDetector
from src.models   import DetectionInput

app = FastAPI(
    title="Hallucination Detector API", version="4.1.0",
    description=(
        "Detect LLM hallucinations. Provider via `X-Provider` header: `free`, `openai`, `gemini`.\n"
        "For `openai`/`gemini`, pass key in `X-API-Key`.\n"
        "When no paragraph context is provided, the API automatically searches the web for context."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = pathlib.Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Web Search Helper ─────────────────────────────────────────────────────────

def web_search_context(question: str, answer: str) -> tuple:
    """
    Search the web using DuckDuckGo Instant Answer API + Wikipedia fallback.
    Returns (context_paragraph, sources_list).
    """
    query = question.strip()
    query_clean = re.sub(r'^(what|who|when|where|why|how|which|is|are|was|were|did|do|does)\s+', '', query, flags=re.IGNORECASE)
    search_q = query_clean[:200] if len(query_clean) > 10 else query[:200]

    sources = []
    context_parts = []

    # 1. DuckDuckGo Instant Answer API (no key needed)
    try:
        ddg_url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(search_q) + "&format=json&no_redirect=1&no_html=1&skip_disambig=1"
        req = urllib.request.Request(ddg_url, headers={"User-Agent": "TruthyDetector/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            ddg = json.loads(resp.read().decode("utf-8"))

        if ddg.get("AbstractText"):
            context_parts.append(ddg["AbstractText"])
            sources.append({
                "title": ddg.get("Heading", "Wikipedia"),
                "url": ddg.get("AbstractURL", "https://en.wikipedia.org"),
                "snippet": ddg["AbstractText"][:200]
            })

        if ddg.get("Infobox"):
            infobox = ddg["Infobox"]
            if isinstance(infobox, dict) and infobox.get("content"):
                facts = []
                for item in infobox["content"][:6]:
                    if item.get("label") and item.get("value"):
                        facts.append(f"{item['label']}: {item['value']}")
                if facts:
                    context_parts.append("Key facts: " + "; ".join(facts))

        if ddg.get("RelatedTopics"):
            related_texts = []
            for rt in ddg["RelatedTopics"][:3]:
                if isinstance(rt, dict) and rt.get("Text"):
                    related_texts.append(rt["Text"])
                    if rt.get("FirstURL"):
                        sources.append({
                            "title": rt.get("Text", "")[:60],
                            "url": rt["FirstURL"],
                            "snippet": rt["Text"][:200]
                        })
            if related_texts:
                context_parts.append("Related: " + " | ".join(related_texts))

        if ddg.get("Answer"):
            context_parts.insert(0, f"Direct answer: {ddg['Answer']}")

    except Exception:
        pass

    # 2. Wikipedia API fallback
    if not context_parts or (len(" ".join(context_parts)) < 100):
        try:
            wiki_query = urllib.parse.quote(search_q[:100])
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_query}"
            req = urllib.request.Request(wiki_url, headers={"User-Agent": "TruthyDetector/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                wiki = json.loads(resp.read().decode("utf-8"))
            if wiki.get("extract"):
                context_parts.append(wiki["extract"][:600])
                sources.append({
                    "title": wiki.get("title", "Wikipedia"),
                    "url": wiki.get("content_urls", {}).get("desktop", {}).get("page", "https://en.wikipedia.org"),
                    "snippet": wiki["extract"][:200]
                })
        except Exception:
            pass

    # 3. Broader Wikipedia search if still nothing
    if not context_parts:
        try:
            wiki_search_url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + urllib.parse.quote(search_q[:100]) + "&format=json&srlimit=2"
            req = urllib.request.Request(wiki_search_url, headers={"User-Agent": "TruthyDetector/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                wsearch = json.loads(resp.read().decode("utf-8"))
            results = wsearch.get("query", {}).get("search", [])
            if results:
                top = results[0]
                snippet = re.sub(r'<[^>]+>', '', top.get("snippet", ""))
                context_parts.append(top.get("title", "") + ". " + snippet)
                sources.append({
                    "title": top.get("title", "Wikipedia"),
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(top.get('title','').replace(' ','_'))}",
                    "snippet": snippet[:200]
                })
        except Exception:
            pass

    context = "\n\n".join(context_parts).strip()
    return context, sources[:4]


# ── Models ────────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    paragraph: str = Field(default="")
    question:  str = Field(...)
    answer:    str = Field(...)


class SourceInfo(BaseModel):
    title:   str
    url:     str
    snippet: str


class DetectResponse(BaseModel):
    is_hallucinated:       bool
    confidence:            int
    hallucination_types:   list[str]
    hallucination_names:   list[str]
    hallucinated_elements: list[str]
    explanation:           str
    correct_answer:        str
    provider:              str
    web_search_used:       bool = False
    web_context:           str  = ""
    sources:               list[SourceInfo] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    p = pathlib.Path("templates/index.html")
    return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h1>Not found</h1>", 404)

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "4.1.0", "providers": ["free", "openai", "gemini"], "features": ["web_search"]}

@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
def detect(
    req: DetectRequest,
    x_api_key : Optional[str] = Header(default="",      alias="X-API-Key"),
    x_provider: Optional[str] = Header(default="free",  alias="X-Provider"),
):
    provider = (x_provider or "free").strip().lower()
    if provider not in ("free", "openai", "gemini"):
        raise HTTPException(400, "X-Provider must be: free, openai, or gemini.")
    if provider in ("openai", "gemini") and not (x_api_key or "").strip():
        raise HTTPException(401, f"X-API-Key header required for provider '{provider}'.")

    paragraph       = req.paragraph.strip()
    web_search_used = False
    web_context     = ""
    sources         = []

    # ── Auto web search when no context is provided ──────────────────────────
    if not paragraph:
        try:
            web_context, sources = web_search_context(req.question, req.answer)
            if web_context:
                paragraph       = web_context
                web_search_used = True
        except Exception:
            pass  # fall back to pure world-knowledge mode

    try:
        detector = HallucinationDetector(provider=provider, api_key=(x_api_key or "").strip())
        result   = detector.detect(DetectionInput(
            paragraph=paragraph, question=req.question, answer=req.answer
        ))
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ["invalid_api_key","authentication","401","api key not"]):
            raise HTTPException(401, "Invalid API key.")
        if any(k in msg.lower() for k in ["quota","rate_limit","429","exceeded"]):
            raise HTTPException(429, "Rate limit or quota exceeded. Try again shortly.")
        raise HTTPException(500, msg)

    return DetectResponse(
        is_hallucinated       = result.is_hallucinated,
        confidence            = result.confidence,
        hallucination_types   = result.type_codes,
        hallucination_names   = [t.display_name for t in result.hallucination_types],
        hallucinated_elements = result.hallucinated_elements,
        explanation           = result.explanation,
        correct_answer        = result.correct_answer,
        provider              = provider,
        web_search_used       = web_search_used,
        web_context           = web_context if web_search_used else "",
        sources               = [SourceInfo(**s) for s in sources],
    )
