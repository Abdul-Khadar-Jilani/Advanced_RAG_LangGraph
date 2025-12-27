"""
Agentic RAG Pipeline with LangGraph — UI-compatible version

Changes from original:
- No hardcoded/preloaded URLs or global retriever.
- run_rag_agent(question, retriever=None) accepts an external retriever.
- retrieve() reads retriever from state["retriever"].
- Preserves prompts, structured LLMs, graders, decision prints.
- State debug prints are present but commented out for quick enable.
"""

import os
from typing import Annotated, List, TypedDict, Optional
from pydantic import BaseModel, Field

# LangChain / LangGraph imports (keep as in your original)
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import dotenv
#import .env # Load environment variables from .env file
dotenv.load_dotenv()  # Load environment variables from .env file

# === Environment (replace keys before production) ===

#load api keys from .env file
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# === LLMs and embeddings (same as original) ===
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0.2,
    top_p=0.7,
    max_completion_tokens=1024,
)

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    truncate="END"
)

# web search tool (keeps same behavior for websearch)
web_search_tool = TavilySearchResults(k=3)

# === Pydantic structured outputs ===
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: str = Field(..., description="Given a user question choose to route it to web search or vectorstore.")

# === Graph state definition ===
class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    web_search: Optional[str]
    documents: List[dict]
    # retriever may be passed in from outside (UI)
    retriever: Optional[object]

# === (Optional) helper: setup_vectorstore() left here but NOT called by default ===
# You can keep this for programmatic vectorstore creation if you want,
# but it is not used by the app by default when running via UI.
def setup_vectorstore(urls: Optional[List[str]] = None, docs: Optional[List] = None, embed_model=None):
    """
    Helper to create a FAISS retriever from a list of URLs or preloaded Document objects.
    Not used by default in UI flow (UI should pass retriever).
    """
    if docs is None:
        docs = []
    if urls:
        for url in urls:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)
    embedder = embed_model or embeddings
    vectorstore = FAISS.from_documents(doc_splits, embedding=embedder)
    # return a retriever (user code/Streamlit should pass this retriever into run_rag_agent)
    return vectorstore.as_retriever()

# === Structured LLM chains (keep same wiring as original) ===
structured_llm_grader = llm.with_structured_output(GradeDocuments)
structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
structured_llm_router = llm.with_structured_output(RouteQuery)

# === Prompts (kept same as your original prompts - paste full templates if required) ===
router_prompt = ChatPromptTemplate.from_template(
    """You are an expert at routing a user question to either a vectorstore or a web search.

    - has_local_kb = {has_local_kb}
    - If has_local_kb is True, the vectorstore contains documents that the user has uploaded in this session (e.g., resumes, PDFs, notes, or other data).
    - Use the vectorstore for any questions that may be answered from this local knowledge base.
    - Use websearch only for information clearly unrelated to the local knowledge base or for real-time events.

    <question>
    {question}
    </question>

    Classification (vectorstore or websearch):"""
)



retrieval_grader_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

    <question>
    {question}
    </question>

    <document>
    {document}
    </document>

    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
)

rag_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
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
    Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in the facts.

    <facts>
    {documents}
    </facts>

    <answer>
    {generation}
    </answer>

    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in the facts."""
)

answer_grader_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an answer is useful to resolve a question.
    Give a binary 'yes' or 'no' score to indicate whether the answer is useful to resolve a question.

    <question>
    {question}
    </question>

    <answer>
    {generation}
    </answer>

    Give a binary score 'yes' or 'no' score to indicate whether the answer is useful to resolve a question."""
)

question_rewriter_prompt = ChatPromptTemplate.from_template(
    """You are a question re-writer that converts an input question to a better version that is optimized
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.

    <question>
    {question}
    </question>

    Improved question:"""
)

# === Chains composition (same as original wiring) ===
question_router = router_prompt | structured_llm_router
retrieval_grader = retrieval_grader_prompt | structured_llm_grader
rag_chain = rag_prompt | llm | StrOutputParser()
hallucination_grader = hallucination_grader_prompt | structured_llm_hallucination_grader
answer_grader = answer_grader_prompt | structured_llm_answer_grader
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()

# === Node functions (preserve prints and comment state dumps) ===
def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(f"Question: {question}")
    has_local_kb = state.get("retriever") is not None

    source = question_router.invoke({
        "question": question,
        "has_local_kb": has_local_kb
    })

    print(f"Route to: {source.datasource}")

    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"



def retrieve(state):
    """
    Retrieve documents from vectorstore — uses retriever passed in state['retriever'].
    """
    # print(f"[RETRIEVE] STATE: {state}")  # uncomment for debug
    print("---RETRIEVE---")
    question = state["question"]

    retriever = state.get("retriever")
    if retriever is None:
        print("⚠️ No retriever provided — skipping vectorstore retrieval.")
        return {"documents": [], "question": question}

    # Try common retriever interfaces robustly
    documents = []
    try:
        # If retriever is a LangGraph tool-like or wrapper with invoke()
        if hasattr(retriever, "invoke"):
            documents = retriever.invoke(question)
        # Standard LangChain retriever
        elif hasattr(retriever, "get_relevant_documents"):
            documents = retriever.get_relevant_documents(question)
        # Callable object
        elif callable(retriever):
            documents = retriever(question)
        else:
            print("⚠️ Retriever object has no known call method; returning empty documents.")
            documents = []
    except Exception as e:
        print(f"Error while invoking retriever: {e}")
        documents = []

    return {"documents": documents, "question": question}


def grade_documents(state):
    # print(f"[GRADE_DOCUMENTS] STATE: {state}")
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", []) or []

    filtered_docs = []
    web_search = "No"
    for d in documents:
        # d may be a Document-like object or a dict; try to get page_content
        page_text = getattr(d, "page_content", None) or d.get("page_content", "")
        score = retrieval_grader.invoke({"question": question, "document": page_text})
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def generate(state):
    # print(f"[GENERATE] STATE: {state}")
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", []) or []

    # Prepare context — join page_content where available
    context_texts = []
    for d in documents:
        content = getattr(d, "page_content", None) or d.get("page_content", "")
        context_texts.append(content)
    context = "\n\n".join(context_texts)

    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_generation_v_documents_and_question(state):
    # print(f"[GRADE_GENERATION] STATE: {state}")
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents", []) or []
    generation = state.get("generation", "")

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score2 = answer_grader.invoke({"question": question, "generation": generation})
        grade2 = score2.binary_score
        if grade2 == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    try:
        docs = web_search_tool.invoke({"query": question})
    except Exception as e:
        print(f"⚠️ Web search tool failed: {e}")
        return {"documents": documents, "question": question}

    # Normalize different formats into page_content strings
    normalized_contents = []
    for d in docs:
        if isinstance(d, dict) and "content" in d:
            normalized_contents.append(d["content"])
        elif hasattr(d, "page_content"):
            normalized_contents.append(d.page_content)
        elif isinstance(d, str):
            normalized_contents.append(d)
        else:
            normalized_contents.append(str(d))

    # Merge into single doc
    web_results_doc = {"page_content": "\n".join(normalized_contents)}
    documents.append(web_results_doc)

    return {"documents": documents, "question": question}


def transform_query(state):
    # print(f"[TRANSFORM_QUERY] STATE: {state}")
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", []) or []

    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    documents = state["documents"]

    # If we still have documents after grading, skip websearch
    if documents and len(documents) > 0:
        print("---DECISION: DOCUMENTS RELEVANT, GENERATE ANSWER---")
        return "DOCS_RELEVANT"
    else:
        print("---DECISION: NO RELEVANT DOCUMENTS, INCLUDE WEB SEARCH---")
        return "DOCS_IRRELEVANT"


# === Build the graph (node names same as original) ===
workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Conditional entry point uses the router function return strings
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
        "websearch": "websearch",
        "generate": "generate",
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
    },
)
workflow.add_edge("transform_query", "retrieve")

app = workflow.compile()

# === Runner ===
def run_rag_agent(question: str, retriever=None):
    """
    Run the RAG agent with an optional retriever (passed from UI).
    If retriever is None, the pipeline will route to websearch.
    """
    inputs = {"question": question, "retriever": retriever}
    # stream/monitoring is optional; here we just invoke for final result
    final_state = app.invoke(inputs)
    # generation key keeps compatibility with previous code
    return final_state.get("generation")

# === Example CLI test ===
if __name__ == "__main__":
    test_qs = [
        "What is the current weather in Hyderabad India?"
    ]

    for q in test_qs:
        print(f"\n{'='*50}")
        print(f"Question: {q}")
        print(f"{'='*50}")
        try:
            ans = run_rag_agent(q)
            print(f"\nFinal Answer: {ans}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*50)
    print("RAG Agent workflow completed!")
