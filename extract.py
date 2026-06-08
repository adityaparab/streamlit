"""Query the Qdrant embeddings created by `ingest.py`.

Embeds the query with the same Ollama embedding model used during ingestion,
performs a similarity search over the Qdrant collection, and optionally produces
an answer grounded in the retrieved chunks via the Ollama generation model.
"""

import argparse
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# Qdrant connection (matches the local Docker deployment).
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "book")

# Remote Ollama server and models — configured via `.env`.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "llama3")


def _vector_store() -> QdrantVectorStore:
    """Connect to the existing Qdrant collection for querying."""
    client = QdrantClient(url=QDRANT_HOST)
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection {COLLECTION_NAME!r} does not exist — run ingest.py first."
        )
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def search(query: str, k: int = 4):
    """Return the top-`k` (Document, score) matches for `query`."""
    store = _vector_store()
    return store.similarity_search_with_score(query, k=k)


def answer(query: str, k: int = 4) -> str:
    """Retrieve context for `query` and generate a grounded answer."""
    results = search(query, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    llm = ChatOllama(model=OLLAMA_GENERATION_MODEL, base_url=OLLAMA_HOST)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    return llm.invoke(prompt).content


def extract(query: str, k: int = 4, generate: bool = False):
    """Search the embeddings for `query`, printing matches (and an answer)."""
    results = search(query, k=k)

    print(f"\nTop {len(results)} matches for: {query!r}\n")
    for rank, (doc, score) in enumerate(results, start=1):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        idx = doc.metadata.get("chunk_index", "?")
        snippet = doc.page_content.replace("\n", " ").strip()[:300]
        print(f"[{rank}] score={score:.4f}  source={src} page={page} chunk={idx}")
        print(f"    {snippet}...\n")

    if generate:
        print("=" * 60)
        print("Answer:\n")
        print(answer(query, k=k))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Qdrant embeddings.")
    parser.add_argument("query", help="The search query.")
    parser.add_argument(
        "-k", type=int, default=4, help="Number of chunks to retrieve."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate an answer from the retrieved chunks via Ollama.",
    )
    args = parser.parse_args()
    extract(args.query, k=args.k, generate=args.generate)
