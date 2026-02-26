# Hallucination Detector

Detect and classify hallucinations in LLM outputs — deployed as a **Web UI + REST API**.

**Live demo:** `https://hallucination-detector.onrender.com` *(after you deploy)*

---

## How it works

Users bring their **own OpenAI API key** — it is sent per-request and **never stored** on the server. This means zero API cost for you.

---

## Hallucination Taxonomy

| Type | Name | Description |
|------|------|-------------|
| **1A** | Out-of-Context Entity | Answer introduces an entity not in the paragraph |
| **1B** | Tuple Verification | Real entities, but incorrectly paired/linked |
| **2A** | Out-of-Context Intent | Correct entity, wrong verb/relationship |
| **3A** | Semantic Triple | Full subject–predicate–object triple is wrong |

---

## Local Development

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. (Optional) create .env for local testing
cp .env.example .env
# Edit .env — add OPENAI_API_KEY only if you want CLI mode

# 3. Start the server
uvicorn api:app --reload

# 4. Open in browser
#    Web UI  → http://localhost:8000
#    API docs → http://localhost:8000/docs
```

---

## Deploy to Render (Free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
# Create repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/hallucination-detector.git
git push -u origin main
```

### Step 2 — Create a Render Web Service

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repository
3. Render auto-detects `render.yaml` — just confirm these settings:

| Field | Value |
|-------|-------|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Instance Type | **Free** |

4. Click **Create Web Service** — done in ~2 minutes.

> ⚠️ No environment variables needed on Render — users supply their own keys.

---

## API Usage

### Endpoint

```
POST /detect
```

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `X-OpenAI-Key` | Yes | Your OpenAI API key (`sk-...`) |

### Request body

```json
{
  "paragraph": "Optional context paragraph...",
  "question":  "The question asked to the LLM",
  "answer":    "The LLM output to verify"
}
```

### Response

```json
{
  "is_hallucinated":       true,
  "confidence":            95,
  "hallucination_types":   ["1A"],
  "hallucination_names":   ["Entity – Out-of-Context Entity Hallucination"],
  "hallucinated_elements": ["Steven Spielberg"],
  "explanation":           "The paragraph states Christopher Nolan directed Inception...",
  "correct_answer":        "Inception was directed by Christopher Nolan."
}
```

### Python example

```python
import requests

response = requests.post(
    "https://hallucination-detector.onrender.com/detect",
    headers={"X-OpenAI-Key": "sk-your-key-here"},
    json={
        "paragraph": "Inception is a 2010 film directed by Christopher Nolan.",
        "question":  "Who directed Inception?",
        "answer":    "Inception was directed by Steven Spielberg."
    }
)
print(response.json())
```

### cURL example

```bash
curl -X POST https://hallucination-detector.onrender.com/detect \
  -H "Content-Type: application/json" \
  -H "X-OpenAI-Key: sk-your-key-here" \
  -d '{
    "paragraph": "Inception is a 2010 film directed by Christopher Nolan.",
    "question": "Who directed Inception?",
    "answer": "Inception was directed by Steven Spielberg."
  }'
```

---

## Project Structure

```
hallucination_detector/
├── api.py                  ← FastAPI app (Web UI + REST API)
├── main.py                 ← CLI entry point (local use)
├── requirements.txt
├── render.yaml             ← Render deployment config
├── Procfile                ← Alternative start command
├── .gitignore
├── templates/
│   └── index.html          ← Web UI (single-file, no build step)
├── sample_input.json
└── src/
    ├── detector.py         ← Core detection logic (per-request API key)
    ├── models.py           ← Data classes and hallucination taxonomy
    ├── display.py          ← CLI terminal output
    └── samples.py          ← 5 built-in test cases
```
