"""Document text and lightweight grounding utilities."""

from __future__ import annotations

import re
from typing import Any


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def get_binary_score(result: Any) -> str:
    score = getattr(result, "binary_score", None)
    if score is None:
        return "no"
    return str(score).strip().lower()


def document_text(document: Any) -> str:
    """Return page text from LangChain Document objects or dict-like docs."""
    if document is None:
        return ""

    page_content = getattr(document, "page_content", None)
    if page_content is not None:
        return str(page_content)

    if isinstance(document, dict):
        return str(document.get("page_content", ""))

    return str(document)


def format_documents_for_prompt(documents: list[Any]) -> str:
    return "\n\n".join(document_text(document) for document in documents)


def query_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def has_lexical_relevance(question: str, document: str) -> bool:
    question_terms = query_terms(question)
    if not question_terms:
        return False

    document_terms = query_terms(document)
    overlap = question_terms & document_terms
    overlap_ratio = len(overlap) / len(question_terms)

    return len(overlap) >= 2 or overlap_ratio >= 0.4


def normalize_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def extract_grounding_values(text: str) -> set[str]:
    values = {value.lower() for value in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)}
    values.update(value.lower().rstrip(".,;:)") for value in re.findall(r"https?://\S+|www\.\S+", text))

    for raw_number in re.findall(r"\+?\d[\d\s(),.-]{3,}\d", text):
        normalized = normalize_digits(raw_number)
        if len(normalized) >= 4:
            values.add(normalized)
            if re.search(r"[.,]00$", raw_number.strip()) and len(normalized) > 2:
                values.add(normalized[:-2])

    return values


def are_values_supported(generation_values: set[str], context_values: set[str]) -> bool:
    for value in generation_values:
        if value in context_values:
            continue

        if value.isdigit() and any(
            context_value.isdigit()
            and (context_value.endswith(value) or value.endswith(context_value))
            for context_value in context_values
        ):
            continue

        return False

    return True


def is_generation_grounded_in_context(generation: str, context: str) -> bool:
    generation = (generation or "").strip()
    if not generation:
        return False
    if re.search(r"\b(i do not know|i don't know|not available|not found|cannot determine)\b", generation.lower()):
        return True

    generation_values = extract_grounding_values(generation)
    if generation_values:
        context_values = extract_grounding_values(context)
        return are_values_supported(generation_values, context_values)

    generation_terms = query_terms(generation)
    if not generation_terms:
        return False

    context_terms = query_terms(context)
    overlap_ratio = len(generation_terms & context_terms) / len(generation_terms)
    return overlap_ratio >= 0.75


def does_generation_answer_question(question: str, generation: str) -> bool:
    generation = (generation or "").strip()
    if not generation:
        return False

    if extract_grounding_values(generation):
        return True

    question_terms = query_terms(question)
    generation_terms = query_terms(generation)
    return bool(question_terms & generation_terms)
