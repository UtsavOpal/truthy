"""
Truthy — Hallucination Detector API  v7.0
- Free mode  : Groq API (Llama 3.1 8B)
- All modes  : Multi-source real-time web search (Brave > DuckDuckGo > Wikipedia)
- Web context: Returned as clean structured JSON (title + facts + summary)
- Correct answer derived from web context when hallucination detected
- Language: passed via X-Language header; LLM responds in that language
"""
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import pathlib, urllib.request, urllib.parse, json, re, os

# Load .env file automatically (must come before reading env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.detector import HallucinationDetector
from src.models   import DetectionInput

app = FastAPI(title="Truthy — Hallucination Detector API", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = pathlib.Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════
# WEB SEARCH — Multi-source, real-time
# Priority: Brave Search API → DuckDuckGo → Wikipedia REST → Wikipedia Search
# Returns structured WebContext object instead of raw string
# ═══════════════════════════════════════════════════════════════

class WebContext:
    """Structured web context with title, key facts, and summary paragraphs."""
    def __init__(self):
        self.title:   str        = ""
        self.facts:   list[dict] = []   # [{label, value}, ...]
        self.summary: str        = ""
        self.sources: list[dict] = []   # [{title, url, snippet}, ...]

    def to_paragraph(self) -> str:
        """Convert to plain paragraph for the LLM detector."""
        parts = []
        if self.title:
            parts.append(f"Topic: {self.title}")
        if self.summary:
            parts.append(self.summary)
        if self.facts:
            facts_str = "; ".join(f"{f['label']}: {f['value']}" for f in self.facts[:8])
            parts.append(f"Key facts: {facts_str}")
        return "\n\n".join(parts).strip()

    def to_dict(self) -> dict:
        return {
            "title":   self.title,
            "facts":   self.facts,
            "summary": self.summary,
            "sources": self.sources,
        }


def _clean_query(question: str) -> str:
    q = question.strip()
    q = re.sub(
        r'^(what|who|when|where|why|how|which|is|are|was|were|did|do|does|tell me about)\s+',
        '', q, flags=re.IGNORECASE
    )
    return q[:200] if len(q) > 10 else question[:200]


def _brave_search(query: str, api_key: str, ctx: WebContext):
    url = ("https://api.search.brave.com/res/v1/web/search?"
           + urllib.parse.urlencode({"q": query, "count": 5, "text_decorations": 0, "search_lang": "en"}))
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": "TruthyDetector/7.0",
    })
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read())

    results = data.get("web", {}).get("results", [])
    summaries = []
    for item in results[:4]:
        title   = item.get("title", "")
        snippet = item.get("description", "")
        url_val = item.get("url", "")
        if snippet:
            summaries.append(f"{title}: {snippet}")
            ctx.sources.append({"title": title, "url": url_val, "snippet": snippet[:200]})

    if summaries:
        ctx.summary = " ".join(summaries[:3])
        if not ctx.title and results:
            ctx.title = results[0].get("title", "")


def _duckduckgo_search(query: str, ctx: WebContext):
    url = ("https://api.duckduckgo.com/?"
           + urllib.parse.urlencode({"q": query, "format": "json",
                                     "no_redirect": 1, "no_html": 1, "skip_disambig": 1}))
    req = urllib.request.Request(url, headers={"User-Agent": "TruthyDetector/7.0"})
    with urllib.request.urlopen(req, timeout=6) as r:
        ddg = json.loads(r.read())

    if ddg.get("Heading"):
        ctx.title = ddg["Heading"]

    if ddg.get("AbstractText"):
        ctx.summary = ddg["AbstractText"]
        ctx.sources.append({
            "title":   ddg.get("Heading", "Wikipedia"),
            "url":     ddg.get("AbstractURL", "https://en.wikipedia.org"),
            "snippet": ddg["AbstractText"][:200],
        })

    if ddg.get("Answer"):
        ctx.facts.insert(0, {"label": "Direct Answer", "value": ddg["Answer"]})

    if ddg.get("Infobox") and isinstance(ddg["Infobox"], dict):
        for item in ddg["Infobox"].get("content", [])[:10]:
            if item.get("label") and item.get("value"):
                ctx.facts.append({"label": item["label"], "value": str(item["value"])})

    for rt in (ddg.get("RelatedTopics") or [])[:3]:
        if isinstance(rt, dict) and rt.get("Text") and rt.get("FirstURL"):
            ctx.sources.append({
                "title":   rt["Text"][:60],
                "url":     rt["FirstURL"],
                "snippet": rt["Text"][:200],
            })


def _wikipedia_rest(query: str, ctx: WebContext):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query[:100])}"
    req = urllib.request.Request(url, headers={"User-Agent": "TruthyDetector/7.0"})
    with urllib.request.urlopen(req, timeout=7) as r:
        wiki = json.loads(r.read())

    if not wiki.get("extract"):
        return

    if not ctx.title:
        ctx.title = wiki.get("title", "")
    if not ctx.summary:
        ctx.summary = wiki["extract"][:800]

    page_url = wiki.get("content_urls", {}).get("desktop", {}).get("page", "https://en.wikipedia.org")
    if not any(s["url"] == page_url for s in ctx.sources):
        ctx.sources.append({
            "title":   wiki.get("title", "Wikipedia"),
            "url":     page_url,
            "snippet": wiki["extract"][:200],
        })


def _wikipedia_search(query: str, ctx: WebContext):
    url = ("https://en.wikipedia.org/w/api.php?"
           + urllib.parse.urlencode({
               "action": "query", "list": "search",
               "srsearch": query[:100], "format": "json", "srlimit": 3,
           }))
    req = urllib.request.Request(url, headers={"User-Agent": "TruthyDetector/7.0"})
    with urllib.request.urlopen(req, timeout=6) as r:
        data = json.loads(r.read())

    for item in data.get("query", {}).get("search", [])[:2]:
        snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
        title   = item.get("title", "")
        if snippet:
            if not ctx.summary:
                ctx.summary = f"{title}: {snippet}"
            ctx.sources.append({
                "title": title,
                "url":   f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                "snippet": snippet[:200],
            })


def _tavily_search(query: str, api_key: str, ctx: WebContext):
    """Tavily AI Search — 1000 free searches/month. Get key: https://app.tavily.com"""
    url = "https://api.tavily.com/search"
    payload = json.dumps({
        "query": query,
        "api_key": api_key,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "User-Agent": "TruthyDetector/7.0",
    })
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())

    if data.get("answer"):
        ctx.facts.insert(0, {"label": "Direct Answer", "value": data["answer"]})

    results = data.get("results", [])
    summaries = []
    for item in results[:4]:
        title   = item.get("title", "")
        snippet = item.get("content", "")[:300]
        url_val = item.get("url", "")
        if snippet:
            summaries.append(snippet)
            ctx.sources.append({"title": title, "url": url_val, "snippet": snippet[:200]})

    if summaries:
        if not ctx.summary:
            ctx.summary = " ".join(summaries[:2])
        if not ctx.title and results:
            ctx.title = results[0].get("title", "")



def web_search_context(question: str) -> WebContext:
    """
    Fetch real-time web context from multiple sources.
    Priority: Tavily → Brave → DuckDuckGo → Wikipedia REST → Wikipedia Search
    Returns structured WebContext with title, facts, summary, and sources.
    """
    query = _clean_query(question)
    ctx   = WebContext()

    # 1. Tavily AI Search (free tier: 1000/month — best quality, get key at app.tavily.com)
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            _tavily_search(query, tavily_key, ctx)
        except Exception:
            pass

    # 2. Brave Search (if configured)
    brave_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if brave_key and not ctx.summary:
        try:
            _brave_search(query, brave_key, ctx)
        except Exception:
            pass

    # 3. DuckDuckGo (always free, no key)
    if not ctx.summary:
        try:
            _duckduckgo_search(query, ctx)
        except Exception:
            pass

    # 4. Wikipedia REST (always free, no key)
    if len(ctx.summary) < 150:
        try:
            _wikipedia_rest(query, ctx)
        except Exception:
            pass

    # 5. Wikipedia Search fallback
    if not ctx.summary:
        try:
            _wikipedia_search(query, ctx)
        except Exception:
            pass

    ctx.sources = ctx.sources[:4]
    return ctx


# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

class DetectRequest(BaseModel):
    paragraph: str = Field(default="")
    question:  str = Field(...)
    answer:    str = Field(...)
    language:  str = Field(default="English")

class SourceInfo(BaseModel):
    title:   str
    url:     str
    snippet: str

class WebContextInfo(BaseModel):
    title:   str = ""
    facts:   list[dict] = []
    summary: str = ""

class DetectResponse(BaseModel):
    is_hallucinated:       bool
    confidence:            int
    hallucination_types:   list[str]
    hallucination_names:   list[str]
    hallucinated_elements: list[str]
    explanation:           str
    correct_answer:        str
    provider:              str
    language:              str          = "English"
    web_search_used:       bool         = False
    web_context_raw:       str          = ""
    web_context_structured: WebContextInfo = WebContextInfo()
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
        "status": "ok", "version": "7.0.0",
        "providers": ["free", "openai", "gemini"],
        "free_mode": "groq_llama3.1_8b",
        "web_search": "tavily+brave+duckduckgo+wikipedia",
        "tavily_configured": bool(os.environ.get("TAVILY_API_KEY")),
        "brave_configured":  bool(os.environ.get("BRAVE_API_KEY")),
        "groq_configured":   bool(os.environ.get("GROQ_API_KEY")),
    }

@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
def detect(
    req:        DetectRequest,
    x_api_key:  Optional[str] = Header(default="",     alias="X-API-Key"),
    x_provider: Optional[str] = Header(default="free", alias="X-Provider"),
    x_language: Optional[str] = Header(default="English", alias="X-Language"),
):
    provider = (x_provider or "free").strip().lower()
    language = (x_language or req.language or "English").strip()

    if provider not in ("free", "openai", "gemini"):
        raise HTTPException(400, "X-Provider must be: free, openai, or gemini.")
    if provider in ("openai", "gemini") and not (x_api_key or "").strip():
        raise HTTPException(401, f"X-API-Key required for provider '{provider}'.")

    paragraph       = req.paragraph.strip()
    web_search_used = False
    web_ctx_raw     = ""
    web_ctx_struct  = WebContextInfo()
    sources         = []

    # Auto web search when no context provided
    if not paragraph:
        try:
            wctx = web_search_context(req.question)
            para_from_web = wctx.to_paragraph()
            if para_from_web:
                paragraph       = para_from_web
                web_search_used = True
                web_ctx_raw     = para_from_web
                web_ctx_struct  = WebContextInfo(
                    title   = wctx.title,
                    facts   = wctx.facts,
                    summary = wctx.summary,
                )
                sources = wctx.sources
        except Exception:
            pass

    try:
        detector = HallucinationDetector(
            provider=provider,
            api_key=(x_api_key or "").strip(),
            language=language,
        )
        result = detector.detect(DetectionInput(
            paragraph=paragraph,
            question=req.question,
            answer=req.answer,
        ))
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ["invalid_api_key","authentication","401","api key"]):
            raise HTTPException(401, "Invalid API key.")
        if any(k in msg.lower() for k in ["quota","rate_limit","429","exceeded"]):
            raise HTTPException(429, "Rate limit exceeded. Try again shortly.")
        raise HTTPException(500, msg)

    return DetectResponse(
        is_hallucinated        = result.is_hallucinated,
        confidence             = result.confidence,
        hallucination_types    = result.type_codes,
        hallucination_names    = [t.display_name for t in result.hallucination_types],
        hallucinated_elements  = result.hallucinated_elements,
        explanation            = result.explanation,
        correct_answer         = result.correct_answer,
        provider               = provider,
        language               = language,
        web_search_used        = web_search_used,
        web_context_raw        = web_ctx_raw,
        web_context_structured = web_ctx_struct,
        sources                = [SourceInfo(**s) for s in sources],
    )
