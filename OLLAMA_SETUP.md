# Setting Up Ollama for Free Mode

Truthy's free mode uses Ollama to run a local LLM on your machine.
This gives you real AI-powered detection — no API key needed, completely private.

## Step 1 — Install Ollama

**Windows / Mac:**
Download from https://ollama.com/download and install.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 2 — Pull a model (choose one)

Recommended for best accuracy on low-resource machines:
```bash
ollama pull phi3:mini        # Microsoft Phi-3 Mini — 2.3GB, excellent accuracy
```

Other options:
```bash
ollama pull qwen2:1.5b       # Alibaba Qwen2 1.5B — 1GB, very fast
ollama pull llama3.2:1b      # Meta Llama 3.2 1B — 1.3GB
ollama pull gemma2:2b        # Google Gemma2 2B — 1.6GB, very accurate
```

## Step 3 — Start Ollama

Ollama runs automatically after install. To start manually:
```bash
ollama serve
```

## Step 4 — Run Truthy

```bash
uvicorn api:app --reload
```

Truthy will automatically detect which Ollama models you have and use the best one.
It falls back to heuristics if Ollama is not running.

## Brave Search API (Optional but Recommended)

For the best real-time web search accuracy, get a free Brave Search API key:
1. Go to https://api.search.brave.com
2. Sign up — free tier gives 2,000 queries/month
3. Add to your `.env` file:
   ```
   BRAVE_API_KEY=your_key_here
   ```

Without Brave, Truthy uses DuckDuckGo + Wikipedia (works well for factual queries).
