"""Compiled LangGraph application and public runners."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from advanced_rag.nodes import (
    decide_to_generate,
    generate,
    grade_documents,
    grade_generation_v_documents_and_question,
    retrieve,
    route_question,
    transform_query,
    web_search,
)
from advanced_rag.state import GraphState


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "DOCS_RELEVANT": "generate",
            "DOCS_IRRELEVANT": "websearch",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "transform_query",
            "useful": END,
            "not useful": "transform_query",
            "stop": END,
        },
    )
    workflow.add_edge("transform_query", "retrieve")

    return workflow.compile()


app = build_graph()


def _initial_state(question: str, retriever: Any = None, local_documents: list[Any] | None = None) -> dict[str, Any]:
    return {
        "question": question,
        "retriever": retriever,
        "local_documents": local_documents or [],
        "rewrite_count": 0,
        "trace": [],
    }


def run_rag_agent(question: str, retriever: Any = None, local_documents: list[Any] | None = None):
    """Run the RAG graph and return the final answer string."""
    final_state = app.invoke(_initial_state(question, retriever, local_documents))
    return final_state.get("generation")


def run_rag_agent_with_trace(
    question: str,
    retriever: Any = None,
    local_documents: list[Any] | None = None,
) -> dict[str, Any]:
    """Run the RAG graph and return the final answer plus node trace."""
    final_state = app.invoke(_initial_state(question, retriever, local_documents))
    trace = list(final_state.get("trace", []))

    if trace:
        first_step = trace[0]
        if first_step in {"retrieve", "retrieve_all_local_documents", "websearch"}:
            trace.insert(0, f"route_question -> {first_step}")
    elif retriever is None:
        trace = ["route_question -> websearch"]
    else:
        trace = ["route_question -> vectorstore"]

    return {
        "answer": final_state.get("generation"),
        "trace": trace,
        "final_state": final_state,
    }
