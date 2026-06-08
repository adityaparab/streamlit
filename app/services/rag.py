"""Core RAG logic: ingestion into Qdrant and retrieval / answer generation.

This is the single source of truth used by the FastAPI routes. All LangChain
calls flow through here so observability (LangSmith) captures them uniformly.
"""

import os
import shutil
from datetime import datetime
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Cached Qdrant client (one connection per process)."""
    return QdrantClient(url=settings.qdrant_host)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_host
    )


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.generation_model, base_url=settings.ollama_host
    )


def _vector_store(create: bool = False, dim: int | None = None) -> QdrantVectorStore:
    """Return a vector store bound to the collection.

    If `create` is True, the collection is created (sized to `dim`) when absent.
    Otherwise a missing collection raises.
    """
    client = get_client()
    exists = client.collection_exists(settings.collection_name)

    if not exists:
        if not create:
            raise RuntimeError(
                f"Collection {settings.collection_name!r} does not exist — "
                "ingest a document first."
            )
        if dim is None:
            dim = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _cleanup() -> None:
    """Remove generated chunk files and drop the Qdrant collection."""
    if os.path.isdir(settings.outputs_dir):
        for entry in os.listdir(settings.outputs_dir):
            path = os.path.join(settings.outputs_dir, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

    client = get_client()
    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)


def _save_chunks_to_md(documents, run_dir: str) -> None:
    """Write each chunk (with metadata) to a Markdown file under `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)
    for i, doc in enumerate(documents):
        meta_lines = "\n".join(f"{k}: {v}" for k, v in doc.metadata.items())
        out_path = os.path.join(run_dir, f"chunk_{i:04d}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"---\n{meta_lines}\n---\n\n{doc.page_content}\n")


def ingest(file_path: str, clean: bool = False) -> dict:
    """Ingest the PDF at `file_path` into Qdrant and `outputs/`.

    Returns a summary dict with the chunk count and run directory.
    """
    if clean:
        _cleanup()

    pages = PyPDFLoader(file_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No text extracted from {file_path!r}")

    ingested_at = datetime.now()
    run_label = ingested_at.strftime("%Y-%m-%d %H-%M-%S")
    source_name = os.path.basename(file_path)
    for i, doc in enumerate(chunks):
        doc.metadata.update(
            {
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "ingested_at": ingested_at.isoformat(timespec="seconds"),
            }
        )

    run_dir = os.path.join(settings.outputs_dir, run_label)
    _save_chunks_to_md(chunks, run_dir)

    store = _vector_store(create=True)
    store.add_documents(chunks)

    return {
        "source": source_name,
        "chunks_ingested": len(chunks),
        "collection": settings.collection_name,
        "run_dir": run_dir,
    }


# --------------------------------------------------------------------------- #
# Retrieval / extraction
# --------------------------------------------------------------------------- #
def search(query: str, k: int = 4):
    """Return the top-`k` (Document, score) matches for `query`."""
    return _vector_store().similarity_search_with_score(query, k=k)


def answer(query: str, k: int = 4) -> tuple[str, list]:
    """Retrieve context for `query` and generate a grounded answer.

    Returns (answer_text, results) so callers can also surface the sources.
    """
    results = search(query, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    text = get_llm().invoke(prompt).content
    return text, results


def matches_to_dicts(results) -> list[dict]:
    """Serialize (Document, score) tuples for JSON responses."""
    out = []
    for doc, score in results:
        out.append(
            {
                "score": round(float(score), 4),
                "source": doc.metadata.get("source", "?"),
                "page": doc.metadata.get("page", "?"),
                "chunk_index": doc.metadata.get("chunk_index", "?"),
                "content": doc.page_content,
            }
        )
    return out
