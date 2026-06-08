# rag-tut

A small modular RAG service: ingest PDFs into Qdrant, retrieve/answer over them
with Ollama, exposed via a FastAPI API and a dark-mode Streamlit chat UI.
LangChain runs are traceable through LangSmith.

## Layout

```
app/
  config.py          # central settings from .env
  observability.py   # LangSmith tracing setup
  main.py            # FastAPI app + startup
  api/
    routes.py        # /ingest and /extract endpoints
    schemas.py       # request/response models
  services/
    rag.py           # ingestion + retrieval + answer generation
streamlit_app.py     # dark-mode chat interface
.streamlit/config.toml
ingest.py / extract.py / llm-judge.py   # original CLI scripts (still usable)
```

## Run

1. Start Qdrant: `docker compose up -d`
2. Start the API: `uv run uvicorn app.main:app --reload`
   - Docs at http://localhost:8000/docs
3. Start the chat UI: `uv run streamlit run streamlit_app.py`

## Endpoints

- `POST /ingest` — multipart upload of a PDF (`file`, optional `clean` flag).
- `POST /extract` — JSON `{ "query": str, "k": int, "generate": bool }`.
  With `generate: true` it returns a grounded answer plus source chunks.

## Observability (LangSmith)

Add to `.env` to capture LangChain traces:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your LangSmith key>
LANGSMITH_PROJECT=rag-tut
```

Startup logs whether tracing is active.
