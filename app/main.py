"""FastAPI application entrypoint.

Run with:
    uv run uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.observability import setup_observability

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAG-tut API",
    description="Modular RAG service: ingest PDFs and extract answers (LangChain + Qdrant + Ollama).",
    version="0.1.0",
)


@app.on_event("startup")
def _on_startup() -> None:
    app.state.tracing_enabled = setup_observability()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


app.include_router(router)
