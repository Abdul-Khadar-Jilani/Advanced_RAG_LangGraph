import os

import streamlit as st
import streamlit.components.v1 as components

from advanced_rag.config import DEFAULT_RETRIEVAL_K, ENV_KEYS


def _load_streamlit_secrets() -> None:
    for key in ENV_KEYS:
        if os.getenv(key):
            continue
        try:
            if key in st.secrets and st.secrets[key]:
                os.environ[key] = str(st.secrets[key])
        except Exception:
            continue


_load_streamlit_secrets()

from advanced_rag.graph import run_rag_agent_with_trace
from advanced_rag.ingestion import add_documents_to_vectorstore, load_uploaded_documents, load_url_documents


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
        G -->|max rewrites reached| H
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


def build_trace_mermaid(trace_steps: list[str]) -> str:
    if not trace_steps:
        return "flowchart LR\n    A[Run the agent to see the latest trace]"

    lines = ["flowchart LR", "    A[Question]"]
    previous_id = "A"

    for index, step in enumerate(trace_steps, start=1):
        node_id = f"N{index}"
        safe_label = step.replace('"', "'")
        lines.append(f'    {node_id}["{safe_label}"]')
        lines.append(f"    {previous_id} --> {node_id}")
        previous_id = node_id

    lines.append("    Z[Answer]")
    lines.append(f"    {previous_id} --> Z")
    return "\n".join(lines)


def init_session_state() -> None:
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "docs": [],
        "doc_chunks": [],
        "last_run": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

st.sidebar.header("Knowledge Base Setup")

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

url_input = st.sidebar.text_area(
    "Paste URLs (one per line)",
    placeholder="https://example.com/doc1\nhttps://example.com/doc2",
)

if st.sidebar.button("Index Knowledge Base", type="primary"):
    new_docs = []
    upload_docs, upload_errors = load_uploaded_documents(uploaded_files or [])
    url_docs, url_errors = load_url_documents(url_input)
    new_docs.extend(upload_docs)
    new_docs.extend(url_docs)

    for error in upload_errors + url_errors:
        st.sidebar.error(error)

    if not new_docs:
        st.sidebar.warning("No documents found to index.")
    else:
        with st.spinner("Indexing documents..."):
            try:
                vectorstore, chunk_count, split_docs = add_documents_to_vectorstore(
                    st.session_state["vectorstore"],
                    new_docs,
                )
                st.session_state["vectorstore"] = vectorstore
                st.session_state["docs"].extend(new_docs)
                st.session_state["doc_chunks"].extend(split_docs)
                st.sidebar.success(f"Added {chunk_count} document chunks to KB")
            except Exception as exc:
                st.sidebar.error(f"Could not build the knowledge base: {exc}")
                st.sidebar.info("Set NVIDIA_API_KEY in Streamlit secrets or environment variables before indexing.")

if st.sidebar.button("Clear Knowledge Base"):
    st.session_state["vectorstore"] = None
    st.session_state["docs"] = []
    st.session_state["doc_chunks"] = []
    st.sidebar.warning("Knowledge Base cleared.")

st.title("Agentic RAG with LangGraph")
st.caption("Document ingestion, routed retrieval, grading, grounded generation, and live graph trace.")

overview_tab, trace_tab = st.tabs(["Graph Overview", "Last Run Trace"])

with overview_tab:
    st.subheader("Static LangGraph Flow")
    render_mermaid(STATIC_MERMAID, height=560)

with st.form("question_form", clear_on_submit=False):
    question = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Run Agent")

if submitted and question.strip():
    retriever = None
    if st.session_state["vectorstore"]:
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": DEFAULT_RETRIEVAL_K})

    run_result = run_rag_agent_with_trace(
        question,
        retriever,
        local_documents=st.session_state["doc_chunks"],
    )
    final_answer = run_result["answer"]

    st.session_state["last_run"] = {
        "question": question,
        "answer": final_answer,
        "trace": run_result["trace"],
    }
    st.session_state["chat_history"].append({"question": question, "answer": final_answer})

st.subheader("Chat History")
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

if st.session_state["docs"]:
    with st.expander("View Uploaded Documents"):
        for doc in st.session_state["docs"]:
            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
            st.text(doc.page_content[:500] + "...")
