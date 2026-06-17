"""Backward-compatible imports for the packaged Advanced RAG implementation."""

from advanced_rag.graph import app, build_graph, run_rag_agent, run_rag_agent_with_trace
from advanced_rag.retrieval import setup_vectorstore

__all__ = [
    "app",
    "build_graph",
    "run_rag_agent",
    "run_rag_agent_with_trace",
    "setup_vectorstore",
]


if __name__ == "__main__":
    question = "What is the current weather in Hyderabad India?"
    print(f"Question: {question}")
    print(run_rag_agent(question))
