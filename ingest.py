"""Ingest a PDF into a locally hosted Qdrant database (LangChain pipeline).

Loads a PDF, splits it into overlapping chunks, embeds each chunk with an Ollama
model and stores them in Qdrant via LangChain. Each chunk is also written to a
timestamped Markdown file under the project's `outputs/` folder.
"""

import argparse
import os
import shutil
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Qdrant connection (matches the local Docker deployment).
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "book")

# Remote Ollama server and embedding model — configured via `.env`.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def cleanup(client: QdrantClient) -> None:
    """Remove all generated chunk files and drop the Qdrant collection."""
    if os.path.isdir(OUTPUTS_DIR):
        for entry in os.listdir(OUTPUTS_DIR):
            path = os.path.join(OUTPUTS_DIR, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        print(f"Cleared contents of {OUTPUTS_DIR!r}.")

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped Qdrant collection {COLLECTION_NAME!r}.")


def save_chunks_to_md(documents, run_dir: str) -> None:
    """Write each chunk to a Markdown file (with metadata) under `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)
    for i, doc in enumerate(documents):
        out_path = os.path.join(run_dir, f"chunk_{i:04d}.md")
        meta_lines = "\n".join(f"{k}: {v}" for k, v in doc.metadata.items())
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"---\n{meta_lines}\n---\n\n{doc.page_content}\n")


def ingest(file_path: str, clean: bool = False, early: bool = False) -> int:
    """Ingest the PDF at `file_path` into Qdrant and `outputs/`.

    If `clean` is True, existing outputs and the Qdrant collection are wiped
    before ingestion. Returns the number of chunks ingested.
    """
    client = QdrantClient(url=QDRANT_HOST)

    if clean:
        cleanup(client)

    if early:
        print("Early exit after cleanup (no ingestion).")
        return 0

    # Load and split the PDF into overlapping chunks.
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No text extracted from {file_path!r}")

    # Enrich each chunk with metadata describing its origin.
    ingested_at = datetime.now()
    run_label = ingested_at.strftime("%Y-%m-%d %H-%M-%S")
    source_name = os.path.basename(file_path)
    for i, doc in enumerate(chunks):
        doc.metadata.update(
            {
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "ingested_at": ingested_at.isoformat(timespec="seconds"),
            }
        )

    # Persist chunks to outputs/<human-readable-date-time>/.
    run_dir = os.path.join(OUTPUTS_DIR, run_label)
    save_chunks_to_md(chunks, run_dir)

    # Embeddings via Ollama.
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)

    # Ensure the collection exists, sized to the embedding dimension.
    if not client.collection_exists(COLLECTION_NAME):
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # Store the chunks in Qdrant through LangChain's vector store.
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks into '{COLLECTION_NAME}' ({run_dir}).")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join("files", "book.pdf"),
        help="Path to the PDF to ingest.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe outputs/ and the Qdrant collection before ingesting.",
    )
    parser.add_argument(
        "--early",
        action="store_true",
        help="Exit early after cleanup (no ingestion).",
    )
    args = parser.parse_args()
    ingest(args.file_path, clean=args.clean, early=args.early)
