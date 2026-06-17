"""LangGraph node and edge decision functions."""

from __future__ import annotations

from typing import Any

from advanced_rag.chains import (
    answer_grader,
    hallucination_grader,
    question_rewriter,
    question_router,
    rag_chain,
    retrieval_grader,
    web_search_tool,
)
from advanced_rag.config import MAX_QUERY_REWRITES
from advanced_rag.config import MAX_LOCAL_CONTEXT_CHUNKS
from advanced_rag.retrieval import invoke_retriever
from advanced_rag.utils import (
    document_text,
    does_generation_answer_question,
    format_documents_for_prompt,
    get_binary_score,
    has_lexical_relevance,
    is_generation_grounded_in_context,
)


def route_question(state: dict[str, Any]) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(f"Question: {question}")
    has_local_kb = state.get("retriever") is not None

    try:
        source = question_router.invoke({"question": question, "has_local_kb": has_local_kb})
    except Exception as exc:
        print(f"Router failed: {exc}")
        source = None

    datasource = getattr(source, "datasource", None)
    if datasource not in {"websearch", "vectorstore"}:
        datasource = "vectorstore" if has_local_kb else "websearch"

    print(f"Route to: {datasource}")
    if datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"

    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"


def retrieve(state: dict[str, Any]) -> dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    retriever = state.get("retriever")
    local_documents = state.get("local_documents", []) or []

    if 0 < len(local_documents) <= MAX_LOCAL_CONTEXT_CHUNKS:
        print(f"Using all {len(local_documents)} local document chunk(s)")
        return {
            "documents": local_documents,
            "question": question,
            "trace": ["retrieve_all_local_documents"],
        }

    if retriever is None:
        print("No retriever provided - skipping vectorstore retrieval.")
        return {"documents": [], "question": question, "trace": ["retrieve"]}

    try:
        documents = invoke_retriever(retriever, question)
    except Exception as exc:
        print(f"Error while invoking retriever: {exc}")
        documents = []

    if documents:
        print(f"Retrieved {len(documents)} document(s)")
    else:
        print("No documents retrieved")

    return {"documents": documents, "question": question, "trace": ["retrieve"]}


def grade_documents(state: dict[str, Any]) -> dict[str, Any]:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", []) or []

    filtered_docs = []
    rejected_docs = []
    web_search = "No"
    local_context_mode = state.get("retriever") is not None or bool(state.get("local_documents"))

    for document in documents:
        page_text = document_text(document)
        try:
            score = retrieval_grader.invoke({"question": question, "document": page_text})
            grade = get_binary_score(score)
        except Exception as exc:
            print(f"Document grader failed: {exc}")
            grade = "no"

        if grade == "yes" or has_lexical_relevance(question, page_text):
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(document)
        elif local_context_mode:
            print("---GRADE: DOCUMENT RELEVANCE UNCERTAIN, KEEPING LOCAL CONTEXT---")
            filtered_docs.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            rejected_docs.append(document)
            web_search = "Yes"

    if not filtered_docs and rejected_docs and local_context_mode:
        print("---GRADE FALLBACK: KEEPING RETRIEVED LOCAL DOCUMENTS---")
        filtered_docs = rejected_docs
        web_search = "No"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
        "trace": ["grade_documents"],
    }


def generate(state: dict[str, Any]) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", []) or []
    context = format_documents_for_prompt(documents)

    generation = rag_chain.invoke({"context": context, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "trace": ["generate"],
    }


def grade_generation_v_documents_and_question(state: dict[str, Any]) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents", []) or []
    generation = state.get("generation", "")
    context = format_documents_for_prompt(documents)

    if is_generation_grounded_in_context(generation, context):
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        if does_generation_answer_question(question, generation):
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"

    try:
        score = hallucination_grader.invoke({"documents": context, "generation": generation})
        grade = get_binary_score(score)
    except Exception as exc:
        print(f"Hallucination grader failed: {exc}")
        grade = "no"

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        try:
            score2 = answer_grader.invoke({"question": question, "generation": generation})
            grade2 = get_binary_score(score2)
        except Exception as exc:
            print(f"Answer grader failed: {exc}")
            grade2 = "no"

        if grade2 == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"

        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        if state.get("rewrite_count", 0) >= MAX_QUERY_REWRITES:
            print("---DECISION: MAX QUERY REWRITES REACHED, STOP---")
            return "stop"
        return "not useful"

    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
    if state.get("rewrite_count", 0) >= MAX_QUERY_REWRITES:
        print("---DECISION: MAX QUERY REWRITES REACHED, STOP---")
        return "stop"
    return "not supported"


def web_search(state: dict[str, Any]) -> dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", []) or []

    try:
        docs = web_search_tool.invoke({"query": question})
    except Exception as exc:
        print(f"Web search tool failed: {exc}")
        return {"documents": documents, "question": question, "trace": ["websearch"]}

    normalized_contents = []
    for document in docs:
        if isinstance(document, dict) and "content" in document:
            normalized_contents.append(document["content"])
        elif hasattr(document, "page_content"):
            normalized_contents.append(document.page_content)
        elif isinstance(document, str):
            normalized_contents.append(document)
        else:
            normalized_contents.append(str(document))

    documents.append({"page_content": "\n".join(normalized_contents)})
    return {"documents": documents, "question": question, "trace": ["websearch"]}


def transform_query(state: dict[str, Any]) -> dict[str, Any]:
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", []) or []
    rewrite_count = state.get("rewrite_count", 0) + 1

    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question,
        "rewrite_count": rewrite_count,
        "trace": ["transform_query"],
    }


def decide_to_generate(state: dict[str, Any]) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    documents = state.get("documents", []) or []

    if documents:
        print("---DECISION: DOCUMENTS RELEVANT, GENERATE ANSWER---")
        return "DOCS_RELEVANT"

    print("---DECISION: NO RELEVANT DOCUMENTS, INCLUDE WEB SEARCH---")
    return "DOCS_IRRELEVANT"
