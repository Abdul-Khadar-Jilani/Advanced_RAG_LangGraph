"""Retriever and vectorstore helpers."""

from __future__ import annotations

from typing import Any, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from advanced_rag.chains import embeddings
from advanced_rag.config import EMBEDDING_CHUNK_OVERLAP, EMBEDDING_CHUNK_SIZE


def split_documents(documents: list[Any]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=EMBEDDING_CHUNK_SIZE,
        chunk_overlap=EMBEDDING_CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def setup_vectorstore(
    urls: Optional[list[str]] = None,
    docs: Optional[list[Any]] = None,
    embed_model: Any = None,
):
    """Create a FAISS retriever from URLs or preloaded Document objects."""
    loaded_docs = list(docs or [])
    if urls:
        for url in urls:
            loader = WebBaseLoader(url)
            loaded_docs.extend(loader.load())

    if not loaded_docs:
        return None

    doc_splits = split_documents(loaded_docs)
    vectorstore = FAISS.from_documents(doc_splits, embedding=embed_model or embeddings)
    return vectorstore.as_retriever()


def invoke_retriever(retriever: Any, question: str) -> list[Any]:
    """Invoke common retriever interfaces without coupling nodes to one implementation."""
    if retriever is None:
        return []
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)
    if callable(retriever):
        return retriever(question)
    raise TypeError("Retriever object has no known call method.")
