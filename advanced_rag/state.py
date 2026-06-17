"""LangGraph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class GraphState(TypedDict, total=False):
    question: str
    generation: Optional[str]
    web_search: Optional[str]
    documents: list[Any]
    local_documents: list[Any]
    retriever: Optional[object]
    rewrite_count: int
    trace: Annotated[list[str], operator.add]
