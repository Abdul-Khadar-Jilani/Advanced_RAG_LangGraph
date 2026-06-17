# Architecture

This project is a document-agnostic Agentic RAG application. The graph can answer from any user-loaded document collection, not only resumes.

## Runtime Flow

1. `route_question`
   - Routes to local vectorstore when uploaded documents may answer the question.
   - Routes to web search for clearly external or current information.

2. `retrieve`
   - Invokes the provided retriever interface.
   - Supports LangChain retrievers, callable retrievers, and invoke-style retrievers.
   - Uses all local chunks when the uploaded knowledge base is small enough to fit the configured context budget.

3. `grade_documents`
   - Uses an LLM relevance grader.
   - Keeps local document context available for generation even when relevance is uncertain.
   - Treats grading as a quality signal, not a hard blocker for uploaded documents.

4. `generate`
   - Generates a concise answer using only retrieved context.

5. `grade_generation_v_documents_and_question`
   - Checks answer groundedness against retrieved context.
   - Uses exact value support for numbers, URLs, and emails.
   - Uses an LLM groundedness grader as the semantic fallback.

6. `transform_query`
   - Rewrites the question for retrieval when generated answers are not grounded or useful.
   - Stops after a configured rewrite limit to avoid infinite loops.

## Package Layout

```text
advanced_rag/
  config.py      Runtime constants and env loading
  prompts.py     Prompt templates
  schemas.py     Structured output schemas
  state.py       LangGraph state
  chains.py      Models, tools, and composed chains
  retrieval.py   Retriever and vectorstore helpers
  ingestion.py   UI document loading and indexing helpers
  utils.py       Document text and grounding utilities
  nodes.py       LangGraph nodes and routing decisions
  graph.py       Graph compilation and public runners
```

## Design Principles

- Keep ingestion separate from runtime RAG.
- Keep the UI thin.
- Answer from uploaded documents first when local context is available.
- Treat retrieved documents as data, never instructions.
- Prefer LLM graders for semantic judgments.
- Use deterministic checks only for exact support that machines verify well, such as numeric values, URLs, and emails.
- Keep the graph observable through trace output.
