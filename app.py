import streamlit as st
from pathlib import Path
from io import StringIO
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from tempfile import TemporaryDirectory
from rag import run_rag_agent  # Import your main function from rag.py

# -------------------------------
# Session State Init
# -------------------------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "docs" not in st.session_state:
    st.session_state["docs"] = []

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(new_docs)
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
        
        if st.session_state["vectorstore"] is None:
            st.session_state["vectorstore"] = FAISS.from_documents(split_docs, embedder)
        else:
            st.session_state["vectorstore"].add_documents(split_docs)
        
        st.session_state["docs"].extend(new_docs)
        st.sidebar.success(f"Added {len(split_docs)} document chunks to KB")

# Clear KB button
if st.sidebar.button("Clear Knowledge Base"):
    st.session_state["vectorstore"] = None
    st.session_state["docs"] = []
    st.sidebar.warning("Knowledge Base cleared!")

# -------------------------------
# Main Chat Interface
# -------------------------------
st.title("🤖 Agentic RAG with LangGraph")

question = st.text_input("Ask a question:")

if st.button("Run Agent") and question.strip():
    retriever = None
    if st.session_state["vectorstore"]:
        retriever = st.session_state["vectorstore"].as_retriever()

    # Run your existing LangGraph agent
    # NOTE: If your run_rag_agent requires retriever, modify it accordingly
    final_answer = run_rag_agent(question, retriever)

    st.session_state["chat_history"].append({"question": question, "answer": final_answer})

# -------------------------------
# Chat History Display
# -------------------------------
st.subheader("💬 Chat History")
for chat in st.session_state["chat_history"]:
    st.markdown(f"**Q:** {chat['question']}")
    st.markdown(f"**A:** {chat['answer']}")

# -------------------------------
# Document Preview
# -------------------------------
if st.session_state["docs"]:
    with st.expander("📄 View Uploaded Documents"):
        for doc in st.session_state["docs"]:
            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
            st.text(doc.page_content[:500] + "...")

