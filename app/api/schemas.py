"""Pydantic request/response models for the API."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    source: str
    chunks_ingested: int
    collection: str
    run_dir: str


class ExtractRequest(BaseModel):
    query: str = Field(..., description="The search query.")
    k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve.")
    generate: bool = Field(
        False, description="Also generate a grounded answer from the chunks."
    )


class Match(BaseModel):
    score: float
    source: str
    page: object | None = None
    chunk_index: object | None = None
    content: str


class ExtractResponse(BaseModel):
    query: str
    k: int
    answer: str | None = None
    matches: list[Match]
