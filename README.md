# Advanced RAG with LangGraph

Document-agnostic Retrieval-Augmented Generation using LangGraph, NVIDIA NIM models, FAISS, Streamlit, and layered answer validation.

The goal of this project is to go beyond basic "retrieve then generate" RAG by adding routing, document relevance grading, groundedness checks, answer usefulness grading, query rewriting, web fallback, and trace visibility.

## Highlights

- Upload PDFs, TXT files, DOCX files, or load URLs.
- Route questions to local vectorstore or web search.
- Grade retrieved chunks before generation.
- Use all local chunks directly when the uploaded knowledge base is small.
- Generate answers only from retrieved context.
- Check groundedness and answer usefulness.
- Rewrite queries when retrieval/generation is weak.
- Stop rewrite loops after a configured limit.
- Show the latest LangGraph execution trace in Streamlit.

## Architecture

```text
Start
  -> route_question
      -> vectorstore -> retrieve -> grade_documents
      -> websearch
  -> generate
  -> grade_generation_v_documents_and_question
      -> useful -> End
      -> not supported / not useful -> transform_query -> retrieve
      -> max rewrites reached -> End
```

See [docs/architecture.md](docs/architecture.md) for module-level details.

## Project Structure

```text
advanced_rag/
  advanced_rag/
    config.py
    prompts.py
    schemas.py
    state.py
    chains.py
    retrieval.py
    ingestion.py
    utils.py
    nodes.py
    graph.py
  archive/
    langgraph_rag.py
  docs/
    architecture.md
  evals/
    rag_eval_dataset.jsonl
    run_eval.py
  tests/
    test_utils.py
  app.py
  rag.py
  langgraph.json
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```bash
NVIDIA_API_KEY=your_nvidia_api_key
TAVILY_API_KEY=your_tavily_api_key
USER_AGENT=advanced-rag-langgraph/0.1
```

## Run

```bash
streamlit run app.py
```

Then:

1. Upload documents or paste URLs.
2. Click `Index Knowledge Base`.
3. Ask questions.
4. Inspect the graph overview and latest trace.

For small uploaded documents, the graph sends all indexed chunks into the local context instead of relying on vector similarity alone. For larger document sets, it retrieves a wider candidate set and still keeps local context from being discarded just because a grader is uncertain.

## Direct Python Usage

```python
from rag import run_rag_agent_with_trace

result = run_rag_agent_with_trace("What does the document say about refund policy?", retriever)
print(result["answer"])
print(result["trace"])
```

## Tests

```bash
python -m unittest discover tests
```

## Local Evaluation

For now this project skips LangSmith and includes a small local eval runner:

```bash
python evals/run_eval.py path/to/document.txt
```

## Notes

- The active graph is in `advanced_rag/graph.py`.
- `rag.py` is a compatibility wrapper for older imports.
- `archive/langgraph_rag.py` is retained only as historical reference.
- `pyproject.toml` and `uv.lock` are intentionally ignored in this repo at the moment.
