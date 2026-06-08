"""LangChain observability via LangSmith tracing.

LangChain emits traces automatically when the standard LangSmith environment
variables are present. This module makes that opt-in explicit and logs whether
tracing is active, so it is obvious at startup whether runs are being captured.

To enable, set in `.env` (these are the names LangSmith reads natively):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=<your LangSmith API key>
    LANGSMITH_PROJECT=rag-tut                              # optional, default below
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com     # optional, default below
"""

import logging
import os

logger = logging.getLogger("rag-tut.observability")


def setup_observability() -> bool:
    """Configure LangSmith tracing from the environment.

    Returns True if tracing is enabled, False otherwise.
    """
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}

    if not tracing:
        logger.info(
            "LangChain tracing disabled. Set LANGSMITH_TRACING=true and "
            "LANGSMITH_API_KEY to enable LangSmith observability."
        )
        return False

    # Sensible defaults so a bare LANGSMITH_TRACING=true still works.
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", "rag-tut")

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning(
            "LANGSMITH_TRACING is set but LANGSMITH_API_KEY is missing — "
            "traces will not reach LangSmith."
        )
        return False

    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s).",
        os.environ["LANGSMITH_PROJECT"],
        os.environ["LANGSMITH_ENDPOINT"],
    )
    return True
