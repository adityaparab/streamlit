"""Centralized configuration loaded from the environment / `.env`.

A single `settings` instance is imported across the app so every module shares
the same connection details and model names.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Qdrant (matches the local Docker deployment).
    qdrant_host: str = field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "http://localhost:6333")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "book")
    )

    # Remote Ollama server and models — configured via `.env`.
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        )
    )
    generation_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_GENERATION_MODEL", "llama3")
    )

    # Chunking.
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )

    # Filesystem.
    base_dir: str = field(
        default_factory=lambda: os.path.dirname(os.path.abspath(__file__))
    )

    @property
    def outputs_dir(self) -> str:
        # Project root is the parent of the `app/` package.
        root = os.path.dirname(self.base_dir)
        return os.path.join(root, "outputs")

    @property
    def uploads_dir(self) -> str:
        root = os.path.dirname(self.base_dir)
        return os.path.join(root, "uploads")


settings = Settings()
