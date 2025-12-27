
"""
Agentic RAG Pipeline with LangGraph, Llama 3.1, and NVIDIA NIM
Complete implementation combining multiple cells from the original notebook
"""

import os
import operator
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

# NVIDIA imports
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# LangGraph imports
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Set up environment variables
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")  # Replace with your actual API key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")  # Replace with your actual API key

# Initialize models and tools
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
web_search_tool = TavilySearchResults(k=3)

# Pydantic models for structured outputs
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: str = Field(
        ...,
        description="Given a user question choose to route it to web search or vectorstore.",
    )

# State definition
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]

def setup_vectorstore():
    """Set up the vector store with sample documents."""
    # Sample URLs - replace with your actual data sources
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Create vector store
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )

    return vectorstore.as_retriever()

# Initialize retriever
retriever = setup_vectorstore()

# Structured LLMs for different tasks
structured_llm_grader = llm.with_structured_output(GradeDocuments)
structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt templates
router_prompt = ChatPromptTemplate.from_template(
    """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use websearch.

    <question>
    {question}
    </question>

    Classification:"""
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

# Chain definitions
question_router = router_prompt | structured_llm_router
retrieval_grader = retrieval_grader_prompt | structured_llm_grader
rag_chain = rag_prompt | llm | StrOutputParser()
hallucination_grader = hallucination_grader_prompt | structured_llm_hallucination_grader
answer_grader = answer_grader_prompt | structured_llm_answer_grader
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()

# Node functions
def route_question(state):
    """
    Route question to web search or RAG.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(f"Question: {question}")

    source = question_router.invoke({"question": question})
    print(f"Route to: {source.datasource}")

    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def retrieve(state):
    """
    Retrieve documents from vectorstore.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
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
    """
    Generate answer using RAG on retrieved documents.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def web_search(state):
    """
    Web search based on the question.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = [{"page_content": web_results}]
    if documents is not None:
        documents.extend(web_results)
    else:
        documents = web_results

    return {"documents": documents, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search.
    """
    # import json
    # print("FULL STATE at generate:", json.dumps(state, indent=2, default=str))

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
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

# Compile
app = workflow.compile()

# Test the workflow
def run_rag_agent(question: str):
    """
    Run the RAG agent with a given question.
    """
    inputs = {"question": question}

    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         print(f"Finished running: {key}:")

    # Get final result
    final_state = app.invoke(inputs)
    return final_state["generation"]

# Example usage
if __name__ == "__main__":
    # Test questions
    test_questions = [

        "What is the current weather in Hyderabad India?"

    ]

    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")

        try:
            answer = run_rag_agent(question)
            print(f"\nFinal Answer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\n" + "="*50)
    print("RAG Agent workflow completed!")