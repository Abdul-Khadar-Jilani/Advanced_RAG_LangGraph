"""Prompt templates used by the RAG graph."""

from langchain_core.prompts import ChatPromptTemplate


router_prompt = ChatPromptTemplate.from_template(
    """You are an expert at routing a user question to either a vectorstore or a web search.

- has_local_kb = {has_local_kb}
- If has_local_kb is True, the vectorstore contains documents uploaded or loaded by the user.
- Use vectorstore for any question that may be answerable from those local documents.
- Use websearch only for information clearly unrelated to the local documents or for real-time/current events.

<question>
{question}
</question>

Classification (vectorstore or websearch):"""
)


retrieval_grader_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains facts, entities, terms, or context that could help answer the question, grade it as relevant.
For forms, tables, invoices, payslips, statements, reports, and similar documents, consider synonymous field labels relevant
even when the exact words differ.
This is not a strict test. The goal is only to filter clearly erroneous retrievals.

<question>
{question}
</question>

<document>
{document}
</document>

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
)


rag_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use only the retrieved context to answer the question.
For tables, forms, payslips, invoices, statements, and reports, interpret nearby labels and synonymous field names.
For example, a question may use a common phrase while the document uses a shorter label.
If the context does not contain the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

<question>
{question}
</question>

<context>
{context}
</context>

Answer:"""
)


hallucination_grader_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an answer is grounded in / supported by a set of facts.
Grade "yes" if the answer is directly supported by the facts, even when it uses a concise sentence,
minor rephrasing, or minor formatting differences for values.
Grade "no" only when the answer adds information that is not present in the facts.

<facts>
{documents}
</facts>

<answer>
{generation}
</answer>

Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the facts."""
)


answer_grader_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an answer is useful to resolve a question.
Grade "yes" if the answer provides the requested information or clearly says the information is not available.
Do not require extra explanation for simple factual questions.

<question>
{question}
</question>

<answer>
{generation}
</answer>

Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve the question."""
)


question_rewriter_prompt = ChatPromptTemplate.from_template(
    """You are a question re-writer that converts an input question to a better version optimized
for vectorstore retrieval. Preserve the user's intent and important entities.

<question>
{question}
</question>

Improved question:"""
)
