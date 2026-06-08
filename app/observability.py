"""LangChain observability via LangSmith tracing.

LangChain emits traces automatically when the standard LangSmith environment
variables are present. This module makes that opt-in explicit and logs whether
tracing is active, so it is obvious at startup whether runs are being captured.

To enable, set in `.env`:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your LangSmith API key>
    LANGCHAIN_PROJECT=rag-tut            # optional, defaults below
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com   # optional
"""

import logging
import os

logger = logging.getLogger("rag-tut.observability")


def setup_observability() -> bool:
    """Configure LangSmith tracing from the environment.

    Returns True if tracing is enabled, False otherwise.
    """
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes"}

    if not tracing:
        logger.info(
            "LangChain tracing disabled. Set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY to enable LangSmith observability."
        )
        return False

    # Sensible defaults so a bare LANGCHAIN_TRACING_V2=true still works.
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "rag-tut")

    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning(
            "LANGCHAIN_TRACING_V2 is set but LANGCHAIN_API_KEY is missing — "
            "traces will not reach LangSmith."
        )
        return False

    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s).",
        os.environ["LANGCHAIN_PROJECT"],
        os.environ["LANGCHAIN_ENDPOINT"],
    )
    return True
