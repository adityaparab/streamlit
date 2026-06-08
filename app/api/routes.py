"""API routes: separate endpoints for ingestion and extraction."""

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import ExtractRequest, ExtractResponse, IngestResponse
from app.config import settings
from app.services import rag

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_endpoint(
    file: UploadFile = File(..., description="PDF file to ingest."),
    clean: bool = Form(False, description="Wipe outputs/ and collection first."),
):
    """Ingest an uploaded PDF into the Qdrant vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(settings.uploads_dir, exist_ok=True)
    # Persist the upload to a stable path so PyPDFLoader can read it.
    dest = os.path.join(settings.uploads_dir, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        result = rag.ingest(dest, clean=clean)
    except Exception as exc:  # surface ingestion failures cleanly
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@router.post("/extract", response_model=ExtractResponse, tags=["extraction"])
async def extract_endpoint(req: ExtractRequest):
    """Retrieve relevant chunks for a query and optionally generate an answer."""
    try:
        if req.generate:
            answer_text, results = rag.answer(req.query, k=req.k)
        else:
            answer_text, results = None, rag.search(req.query, k=req.k)
    except RuntimeError as exc:  # collection missing, etc.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ExtractResponse(
        query=req.query,
        k=req.k,
        answer=answer_text,
        matches=rag.matches_to_dicts(results),
    )
