"""Document loading and vectorstore ingestion helpers for the UI."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS

from advanced_rag.chains import embeddings
from advanced_rag.retrieval import split_documents


LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def load_uploaded_documents(uploaded_files: list[Any]) -> tuple[list[Any], list[str]]:
    documents: list[Any] = []
    errors: list[str] = []

    if not uploaded_files:
        return documents, errors

    with TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            try:
                file_path = Path(temp_dir) / uploaded_file.name
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

                extension = file_path.suffix.lower()
                loader_class = LOADER_MAPPING.get(extension)
                if loader_class is None:
                    errors.append(f"Unsupported file type: {uploaded_file.name}")
                    continue

                loader = loader_class(str(file_path))
                documents.extend(loader.load())
            except Exception as exc:
                errors.append(f"Error processing {uploaded_file.name}: {exc}")

    return documents, errors


def load_url_documents(url_input: str) -> tuple[list[Any], list[str]]:
    documents: list[Any] = []
    errors: list[str] = []
    urls = [url.strip() for url in url_input.splitlines() if url.strip()]

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
        except Exception as exc:
            errors.append(f"Error loading URL {url}: {exc}")

    return documents, errors


def add_documents_to_vectorstore(vectorstore: Any, documents: list[Any]) -> tuple[Any, int, list[Any]]:
    split_docs = split_documents(documents)
    if not split_docs:
        return vectorstore, 0, []

    if vectorstore is None:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    else:
        vectorstore.add_documents(split_docs)

    return vectorstore, len(split_docs), split_docs
