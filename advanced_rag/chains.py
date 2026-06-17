"""Model, tool, and chain construction."""

from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from advanced_rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_TRUNCATE,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    WEB_SEARCH_RESULTS,
    load_environment,
)
from advanced_rag.prompts import (
    answer_grader_prompt,
    hallucination_grader_prompt,
    question_rewriter_prompt,
    rag_prompt,
    retrieval_grader_prompt,
    router_prompt,
)
from advanced_rag.schemas import GradeAnswer, GradeDocuments, GradeHallucinations, RouteQuery


load_environment()

llm = ChatNVIDIA(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    top_p=LLM_TOP_P,
    max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
)

embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL, truncate=EMBEDDING_TRUNCATE)

web_search_tool = TavilySearchResults(k=WEB_SEARCH_RESULTS)

question_router = router_prompt | llm.with_structured_output(RouteQuery)
retrieval_grader = retrieval_grader_prompt | llm.with_structured_output(GradeDocuments)
rag_chain = rag_prompt | llm | StrOutputParser()
hallucination_grader = hallucination_grader_prompt | llm.with_structured_output(GradeHallucinations)
answer_grader = answer_grader_prompt | llm.with_structured_output(GradeAnswer)
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()
