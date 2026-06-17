"""Runtime configuration and environment loading."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


LLM_MODEL = "meta/llama-3.3-70b-instruct"
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.7
LLM_MAX_COMPLETION_TOKENS = 1024

EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_TRUNCATE = "END"
EMBEDDING_CHUNK_SIZE = 450
EMBEDDING_CHUNK_OVERLAP = 80

WEB_SEARCH_RESULTS = 3
MAX_QUERY_REWRITES = 2
MAX_LOCAL_CONTEXT_CHUNKS = 12
DEFAULT_RETRIEVAL_K = 12

ENV_KEYS = ("NVIDIA_API_KEY", "TAVILY_API_KEY", "USER_AGENT")


def load_environment() -> None:
    """Load .env values when available without overwriting existing env vars."""
    if load_dotenv is not None:
        load_dotenv()

    for key in ENV_KEYS:
        value = os.getenv(key)
        if value:
            os.environ[key] = value
