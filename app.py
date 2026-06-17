import os
from pathlib import Path
from io import StringIO
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from tempfile import TemporaryDirectory


EMBEDDING_CHUNK_SIZE = 450
EMBEDDING_CHUNK_OVERLAP = 80


def _load_streamlit_secrets() -> None:
    for key in ("NVIDIA_API_KEY", "TAVILY_API_KEY", "USER_AGENT"):
        if os.getenv(key):
            continue
        try:
            if key in st.secrets:
                value = st.secrets[key]
                if value:
                    os.environ[key] = str(value)
        except Exception:
            continue


_load_streamlit_secrets()

from rag import run_rag_agent_with_trace  # Import the trace-aware runner from rag.py


STATIC_MERMAID = """
flowchart TD
        A[Start] --> B[route_question]
        B -->|vectorstore| C[retrieve]
        B -->|websearch| F[websearch]
        C --> D[grade_documents]
        D -->|DOCS_RELEVANT| E[generate]
        D -->|DOCS_IRRELEVANT| F
        F --> E
        E --> G[grade_generation_v_documents_and_question]
        G -->|useful| H[End]
        G -->|not supported / not useful| I[transform_query]
        I --> C
""".strip()


def render_mermaid(diagram: str, height: int = 520) -> None:
        components.html(
                f"""
                <html>
                    <head>
                        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                        <style>
                            body {{ margin: 0; padding: 0; background: transparent; }}
                            .mermaid {{ display: flex; justify-content: center; }}
                        </style>
                    </head>
                    <body>
                        <div class="mermaid">
{diagram}
                        </div>
                        <script>
                            mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', theme: 'neutral' }});
                        </script>
                    </body>
                </html>
                """,
                height=height,
                scrolling=True,
        )


def build_trace_mermaid(trace_steps):
        if not trace_steps:
                return "flowchart LR\n    A[Run the agent to see the latest trace]"

        lines = ["flowchart LR", '    A[Question]']
        previous_id = "A"

        for index, step in enumerate(trace_steps, start=1):
                node_id = f"N{index}"
                safe_label = step.replace('"', "'")
                lines.append(f'    {node_id}["{safe_label}"]')
                lines.append(f"    {previous_id} --> {node_id}")
                previous_id = node_id

        lines.append('    Z[Answer]')
        lines.append(f"    {previous_id} --> Z")
        return "\n".join(lines)

# -------------------------------
# Session State Init
# -------------------------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "docs" not in st.session_state:
    st.session_state["docs"] = []
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

# -------------------------------
# Sidebar - Knowledge Base
# -------------------------------
st.sidebar.header("📚 Knowledge Base Setup")

LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents (PDF, TXT, DOCX)", 
    type=["pdf", "txt", "docx"], 
    accept_multiple_files=True
)

url_input = st.sidebar.text_area(
    "Paste URLs (one per line)",
    placeholder="https://example.com/doc1\nhttps://example.com/doc2"
)

with st.spinner("Processing documents..."):
    new_docs = []
    # Process uploaded files
    if uploaded_files:
        with TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                try:
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    ext = Path(uploaded_file.name).suffix.lower()
                    if ext in LOADER_MAPPING:
                        loader_class = LOADER_MAPPING[ext]
                        loader = loader_class(str(file_path))
                        new_docs.extend(loader.load())
                except Exception as e:
                    st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")


        # Process URLs
        if url_input.strip():
            urls = [u.strip() for u in url_input.split("\n") if u.strip()]
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    new_docs.extend(loader.load())
                except Exception as e:
                    st.sidebar.error(f"Error loading URL {url}: {e}")


    if new_docs:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=EMBEDDING_CHUNK_SIZE,
            chunk_overlap=EMBEDDING_CHUNK_OVERLAP,
        )
        split_docs = splitter.split_documents(new_docs)
        try:
            embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

            if st.session_state["vectorstore"] is None:
                st.session_state["vectorstore"] = FAISS.from_documents(split_docs, embedder)
            else:
                st.session_state["vectorstore"].add_documents(split_docs)

            st.session_state["docs"].extend(new_docs)
            st.sidebar.success(f"Added {len(split_docs)} document chunks to KB")
        except Exception as exc:
            st.sidebar.error(f"Could not build the knowledge base: {exc}")
            st.sidebar.info("Set NVIDIA_API_KEY in Streamlit secrets or environment variables before uploading PDFs.")

# Clear KB button
if st.sidebar.button("Clear Knowledge Base"):
    st.session_state["vectorstore"] = None
    st.session_state["docs"] = []
    st.sidebar.warning("Knowledge Base cleared!")

# -------------------------------
# Main Chat Interface
# -------------------------------
st.title("🤖 Agentic RAG with LangGraph")
st.caption("Static graph overview plus the live last-run trace for the most recent question.")

overview_tab, trace_tab = st.tabs(["Graph Overview", "Last Run Trace"])

with overview_tab:
    st.subheader("Static LangGraph Flow")
    render_mermaid(STATIC_MERMAID, height=560)

question = st.text_input("Ask a question:")

if st.button("Run Agent") and question.strip():
    retriever = None
    if st.session_state["vectorstore"]:
        retriever = st.session_state["vectorstore"].as_retriever()

    run_result = run_rag_agent_with_trace(question, retriever)
    final_answer = run_result["answer"]

    st.session_state["last_run"] = {
        "question": question,
        "answer": final_answer,
        "trace": run_result["trace"],
    }
    st.session_state["chat_history"].append({"question": question, "answer": final_answer})

# -------------------------------
# Chat History Display
# -------------------------------
st.subheader("💬 Chat History")
for chat in st.session_state["chat_history"]:
    st.markdown(f"**Q:** {chat['question']}")
    st.markdown(f"**A:** {chat['answer']}")

with trace_tab:
    st.subheader("Latest Execution Path")
    last_run = st.session_state.get("last_run")

    if last_run:
        st.markdown(f"**Question:** {last_run['question']}")
        st.markdown(f"**Answer:** {last_run['answer']}")
        render_mermaid(build_trace_mermaid(last_run.get("trace", [])), height=420)

        with st.expander("Show node trace"):
            for step in last_run.get("trace", []):
                st.write(step)
    else:
        st.info("Run the agent once to display the live last-run trace.")

# -------------------------------
# Document Preview
# -------------------------------
if st.session_state["docs"]:
    with st.expander("📄 View Uploaded Documents"):
        for doc in st.session_state["docs"]:
            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
            st.text(doc.page_content[:500] + "...")
