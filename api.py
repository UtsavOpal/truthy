"""
Truthy — Hallucination Detector API  v5.0
Free mode: Ollama local LLM (phi3:mini or any available model)
All modes: multi-source real-time web search when no context provided
Best-in-class free web search: Brave Search API > DuckDuckGo > Wikipedia
"""
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import pathlib, urllib.request, urllib.parse, json, re, os

from src.detector import HallucinationDetector
from src.models   import DetectionInput

app = FastAPI(
    title="Truthy — Hallucination Detector API", version="5.0.0",
    description=(
        "Detect LLM hallucinations. Provider via `X-Provider` header: `free`, `openai`, `gemini`.\n"
        "For `openai`/`gemini` pass key in `X-API-Key`.\n"
        "When no paragraph is provided, the API automatically fetches real-time web context.\n"
        "Free mode uses Ollama local LLM; falls back to heuristics if Ollama is unavailable."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = pathlib.Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════
# WEB SEARCH — Multi-source, real-time, best accuracy
# Priority: Brave Search API → DuckDuckGo → Wikipedia REST → Wikipedia Search
# ═══════════════════════════════════════════════════════════════

def _clean_query(question: str) -> str:
    """Build a clean, focused search query from the question."""
    q = question.strip()
    q = re.sub(r'^(what|who|when|where|why|how|which|is|are|was|were|did|do|does|tell me about)\s+',
               '', q, flags=re.IGNORECASE)
    return q[:200] if len(q) > 10 else question[:200]


def _brave_search(query: str, api_key: str) -> tuple[list[str], list[dict]]:
    """
    Brave Search API — best free real-time web search.
    Free tier: 2,000 queries/month. Sign up: https://api.search.brave.com
    Returns (context_parts, sources)
    """
    url = ("https://api.search.brave.com/res/v1/web/search?"
           + urllib.parse.urlencode({"q": query, "count": 5, "text_decorations": 0, "search_lang": "en"}))
    req = urllib.request.Request(url, headers={
        "Accept":               "application/json",
        "Accept-Encoding":      "gzip",
        "X-Subscription-Token": api_key,
        "User-Agent":           "TruthyDetector/5.0",
    })
    with urllib.request.urlopen(req, timeout=7) as r:
        data = json.loads(r.read())

    parts   = []
    sources = []
    results = data.get("web", {}).get("results", [])

    for item in results[:4]:
        title   = item.get("title", "")
        snippet = item.get("description", "")
        url_val = item.get("url", "")
        if snippet:
            parts.append(f"{title}: {snippet}")
            sources.append({"title": title, "url": url_val, "snippet": snippet[:200]})

    # Also grab featured snippet / infobox if present
    if data.get("query", {}).get("answer"):
        parts.insert(0, data["query"]["answer"])

    return parts, sources


def _duckduckgo_search(query: str) -> tuple[list[str], list[dict]]:
    """DuckDuckGo Instant Answer API — no key needed, limited to Wikipedia-style facts."""
    url = ("https://api.duckduckgo.com/?"
           + urllib.parse.urlencode({"q": query, "format": "json",
                                     "no_redirect": 1, "no_html": 1, "skip_disambig": 1}))
    req = urllib.request.Request(url, headers={"User-Agent": "TruthyDetector/5.0"})
    with urllib.request.urlopen(req, timeout=6) as r:
        ddg = json.loads(r.read())

    parts   = []
    sources = []

    if ddg.get("Answer"):
        parts.append(f"Direct answer: {ddg['Answer']}")

    if ddg.get("AbstractText"):
        parts.append(ddg["AbstractText"])
        sources.append({
            "title":   ddg.get("Heading", "Wikipedia"),
            "url":     ddg.get("AbstractURL", "https://en.wikipedia.org"),
            "snippet": ddg["AbstractText"][:200],
        })

    if ddg.get("Infobox") and isinstance(ddg["Infobox"], dict):
        facts = []
        for item in ddg["Infobox"].get("content", [])[:6]:
            if item.get("label") and item.get("value"):
                facts.append(f"{item['label']}: {item['value']}")
        if facts:
            parts.append("Key facts: " + "; ".join(facts))

    for rt in (ddg.get("RelatedTopics") or [])[:3]:
        if isinstance(rt, dict) and rt.get("Text"):
            parts.append(rt["Text"])
            if rt.get("FirstURL"):
                sources.append({
                    "title":   rt["Text"][:60],
                    "url":     rt["FirstURL"],
                    "snippet": rt["Text"][:200],
                })

    return parts, sources


def _wikipedia_rest(query: str) -> tuple[list[str], list[dict]]:
    """Wikipedia REST API — reliable, structured, always available."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query[:100])}"
    req = urllib.request.Request(url, headers={"User-Agent": "TruthyDetector/5.0"})
    with urllib.request.urlopen(req, timeout=7) as r:
        wiki = json.loads(r.read())

    if not wiki.get("extract"):
        return [], []

    extract = wiki["extract"]
    page_url = wiki.get("content_urls", {}).get("desktop", {}).get("page", "https://en.wikipedia.org")
    return (
        [extract[:800]],
        [{"title": wiki.get("title", "Wikipedia"), "url": page_url, "snippet": extract[:200]}]
    )


def _wikipedia_search(query: str) -> tuple[list[str], list[dict]]:
    """Wikipedia search API — broader fallback when exact page not found."""
    url = ("https://en.wikipedia.org/w/api.php?"
           + urllib.parse.urlencode({
               "action": "query", "list": "search",
               "srsearch": query[:100], "format": "json", "srlimit": 3,
           }))
    req = urllib.request.Request(url, headers={"User-Agent": "TruthyDetector/5.0"})
    with urllib.request.urlopen(req, timeout=6) as r:
        data = json.loads(r.read())

    results = data.get("query", {}).get("search", [])
    parts, sources = [], []
    for item in results[:2]:
        snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
        title   = item.get("title", "")
        if snippet:
            parts.append(f"{title}: {snippet}")
            sources.append({
                "title":   title,
                "url":     f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                "snippet": snippet[:200],
            })
    return parts, sources


def web_search_context(question: str, answer: str) -> tuple[str, list]:
    """
    Fetch real-time web context using best available source.
    Priority: Brave Search API (if key set) → DuckDuckGo → Wikipedia REST → Wikipedia Search
    Returns (context_paragraph, sources_list)
    """
    query   = _clean_query(question)
    parts   = []
    sources = []

    # 1. Brave Search (best accuracy, free tier 2k/month)
    brave_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if brave_key:
        try:
            p, s = _brave_search(query, brave_key)
            parts.extend(p); sources.extend(s)
        except Exception:
            pass

    # 2. DuckDuckGo (no key, good for Wikipedia-style facts)
    if not parts:
        try:
            p, s = _duckduckgo_search(query)
            parts.extend(p); sources.extend(s)
        except Exception:
            pass

    # 3. Wikipedia REST (direct page summary — very reliable)
    if len(" ".join(parts)) < 150:
        try:
            p, s = _wikipedia_rest(query)
            parts.extend(p); sources.extend(s)
        except Exception:
            pass

    # 4. Wikipedia Search (broad fallback)
    if not parts:
        try:
            p, s = _wikipedia_search(query)
            parts.extend(p); sources.extend(s)
        except Exception:
            pass

    context = "\n\n".join(parts).strip()
    return context, sources[:4]


# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

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
    web_search_used:       bool          = False
    web_context:           str           = ""
    sources:               list[SourceInfo] = []


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    p = pathlib.Path("templates/index.html")
    return HTMLResponse(p.read_text(encoding="utf-8")) if p.exists() else HTMLResponse("<h1>Not found</h1>", 404)

@app.get("/sitemap.xml", include_in_schema=False)
def sitemap():
    p = pathlib.Path("static/sitemap.xml")
    return FileResponse(str(p), media_type="application/xml") if p.exists() else HTMLResponse("Not found", 404)

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok", "version": "5.0.0",
        "providers": ["free", "openai", "gemini"],
        "free_mode": "ollama_with_heuristic_fallback",
        "web_search": "brave+duckduckgo+wikipedia",
        "brave_key_configured": bool(os.environ.get("BRAVE_API_KEY")),
    }

@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
def detect(
    req:        DetectRequest,
    x_api_key:  Optional[str] = Header(default="",     alias="X-API-Key"),
    x_provider: Optional[str] = Header(default="free", alias="X-Provider"),
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

    # Auto web search when no context provided (all providers)
    if not paragraph:
        try:
            web_context, sources = web_search_context(req.question, req.answer)
            if web_context:
                paragraph       = web_context
                web_search_used = True
        except Exception:
            pass  # fallback to world-knowledge mode

    try:
        detector = HallucinationDetector(
            provider=provider,
            api_key=(x_api_key or "").strip()
        )
        result = detector.detect(DetectionInput(
            paragraph=paragraph,
            question=req.question,
            answer=req.answer
        ))
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ["invalid_api_key", "authentication", "401", "api key"]):
            raise HTTPException(401, "Invalid API key.")
        if any(k in msg.lower() for k in ["quota", "rate_limit", "429", "exceeded"]):
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
