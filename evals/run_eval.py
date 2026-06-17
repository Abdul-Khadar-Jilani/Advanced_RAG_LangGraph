"""Small local evaluation runner for document-grounded QA.

Usage:
    python evals/run_eval.py path/to/document.txt

The script intentionally avoids LangSmith for now. It builds a temporary
vectorstore from the supplied text file and checks whether expected substrings
appear in answers for the examples in rag_eval_dataset.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.documents import Document

from advanced_rag.graph import run_rag_agent_with_trace
from advanced_rag.retrieval import setup_vectorstore


DATASET_PATH = Path(__file__).with_name("rag_eval_dataset.jsonl")


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    document_path = Path(sys.argv[1])
    document = Document(
        page_content=document_path.read_text(encoding="utf-8"),
        metadata={"source": str(document_path)},
    )
    retriever = setup_vectorstore(docs=[document])

    failures = 0
    for example in load_dataset():
        result = run_rag_agent_with_trace(example["question"], retriever)
        answer = result["answer"] or ""
        expected = [item.lower() for item in example.get("answer_contains", [])]
        passed = all(item in answer.lower() for item in expected)
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {example['question']}")
        print(f"  answer: {answer}")
        print(f"  trace: {' -> '.join(result['trace'])}")
        if not passed:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
