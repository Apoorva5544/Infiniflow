"""
Infiniflow Evaluation Harness
=============================
Runs the real retrieval pipeline over a deterministic corpus and reports
citation-grade metrics:

- context_precision@5  (retrieval: fraction of top-5 chunks containing the
  evidence needed to answer the question) — computed with and without the
  cross-encoder reranker so you can measure its impact directly.
- faithfulness         (generation, needs GROQ_API_KEY): 0–1 estimate of how
  well the answer stays grounded in the retrieved context.

Usage:
    python -m evals.evaluate              # offline retrieval metrics
    python -m evals.evaluate --json       # machine-readable output

Exit status is non-zero if any metric falls below CI_* thresholds, so the
suite can be wired into CI to catch retrieval regressions.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from evals.corpus import CORPUS, QUESTIONS

# Fail CI below these floors (believable baseline for a deterministic corpus).
CI_CONTEXT_PRECISION_OK = 0.2
CI_RECALL_OK = 0.75
CI_FAITHFULNESS_OK = 0.65

_SOURCE_RE = re.compile(r"\b[a-zA-Z0-9_.-]+\.(?:pdf|txt|md|docx)\b")


def build_chunks() -> List[Document]:
    chunks = []
    for item in CORPUS:
        doc = Document(
            page_content=item["text"],
            metadata={"source": item["source"], "page": item["page"]},
        )
        chunks.append(doc)
    return chunks


def significant_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())}


def token_overlap(text_a: str, text_b: str) -> float:
    a, b = significant_tokens(text_a), significant_tokens(text_b)
    if not a:
        return 0.0
    return len(a & b) / len(a)


def precision_at_k(
    retrieved: List[Document], ground_truth: str, positive_threshold: float = 0.4
) -> float:
    """Fraction of top-k chunks that carry evidence for the ground truth."""
    top = retrieved[:5]
    if not top:
        return 0.0
    hits = sum(
        1
        for d in top
        if token_overlap(ground_truth, d.page_content) >= positive_threshold
    )
    return hits / len(top)


def recall_at_k(retrieved: List[Document], expected_sources: List[str]) -> float:
    """Fraction of expected sources present anywhere in the top-k retrieval."""
    top = retrieved[:5]
    if not expected_sources:
        return 0.0
    found = {d.metadata.get("source", "?") for d in top}
    hits = sum(1 for s in expected_sources if s in found)
    return hits / len(expected_sources)


def mrr_at_k(retrieved: List[Document], expected_sources: List[str]) -> float:
    """Reciprocal rank of the first chunk that belongs to an expected source."""
    for rank, d in enumerate(retrieved[:5], start=1):
        if d.metadata.get("source", "?") in expected_sources:
            return 1.0 / rank
    return 0.0


def retrieve(vs, chunks: List[Document], query: str, rerank: bool) -> List[Document]:
    from rag_engine import get_hybrid_retriever

    retriever = get_hybrid_retriever(vs, chunks=chunks)
    docs = retriever.invoke(query)
    if rerank:
        from ai_engine.reranker import CrossEncoderReranker

        docs = CrossEncoderReranker(top_k=5).compress_documents(docs, query)
    return docs


def evaluate(retrieval_only: bool = False) -> Dict:
    from rag_engine import _embeddings, CHROMA_PATH as _ROOT
    from langchain_community.vectorstores import Chroma

    temp_dir = tempfile.mkdtemp(prefix="infiniflow_eval_")
    chunks = build_chunks()
    vs = Chroma.from_documents(
        chunks, _embeddings(), persist_directory=os.path.join(temp_dir, "db")
    )

    try:
        rows = []
        for q in QUESTIONS:
            base = retrieve(vs, chunks, q["question"], rerank=False)
            reranked = retrieve(vs, chunks, q["question"], rerank=True)
            base_sources = {d.metadata.get("source", "?") for d in base}
            rows.append(
                {
                    "question": q["question"],
                    "p_at_5_hybrid": precision_at_k(base, q["ground_truth"]),
                    "p_at_5_hybrid_reranked": precision_at_k(
                        reranked, q["ground_truth"]
                    ),
                    "recall_at_5_hybrid": recall_at_k(base, q["expected_sources"]),
                    "recall_at_5_hybrid_reranked": recall_at_k(
                        reranked, q["expected_sources"]
                    ),
                    "mrr_at_5_hybrid": mrr_at_k(base, q["expected_sources"]),
                    "mrr_at_5_hybrid_reranked": mrr_at_k(
                        reranked, q["expected_sources"]
                    ),
                    "sources_found": base_sources.issuperset(q["expected_sources"]),
                }
            )

        report = {
            "metric": "context_precision@5",
            "hybrid": {"n": len(rows), "mean": _mean(r["p_at_5_hybrid"] for r in rows)},
            "hybrid_reranked": {
                "n": len(rows),
                "mean": _mean(r["p_at_5_hybrid_reranked"] for r in rows),
            },
            "recall_at_5_hybrid": {
                "n": len(rows),
                "mean": _mean(r["recall_at_5_hybrid"] for r in rows),
            },
            "recall_at_5_hybrid_reranked": {
                "n": len(rows),
                "mean": _mean(r["recall_at_5_hybrid_reranked"] for r in rows),
            },
            "mrr_at_5_hybrid": {
                "n": len(rows),
                "mean": _mean(r["mrr_at_5_hybrid"] for r in rows),
            },
            "mrr_at_5_hybrid_reranked": {
                "n": len(rows),
                "mean": _mean(r["mrr_at_5_hybrid_reranked"] for r in rows),
            },
            "reranker_improvement": None,
            "queries": rows,
            "faithfulness": None,
        }
        base_mean = report["hybrid"]["mean"]
        reranked_mean = report["hybrid_reranked"]["mean"]
        if base_mean > 0:
            report["reranker_improvement"] = round(reranked_mean - base_mean, 4)

        if not retrieval_only and os.getenv("GROQ_API_KEY"):
            report["faithfulness"] = _evaluate_faithfulness()
        return report
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _evaluate_faithfulness() -> Optional[Dict]:
    """LLM-as-judge faithfulness over generated answers (needs API key)."""
    from langchain_groq import ChatGroq
    from rag_engine import _embeddings, get_hybrid_retriever
    from ai_engine.reranker import CrossEncoderReranker
    from langchain_community.vectorstores import Chroma
    import tempfile, shutil

    llm = ChatGroq(
        temperature=0,
        model_name="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip("\"' "),
        max_tokens=int(os.getenv("MAX_LLM_OUTPUT_TOKENS", "900")),
    )
    temp_dir = tempfile.mkdtemp(prefix="infiniflow_eval_")
    chunks = build_chunks()
    try:
        vs = Chroma.from_documents(
            chunks, _embeddings(), persist_directory=os.path.join(temp_dir, "db")
        )
        retriever = get_hybrid_retriever(vs, chunks=chunks)
        scores = []
        for q in QUESTIONS:
            docs = CrossEncoderReranker(top_k=5).compress_documents(
                retriever.invoke(q["question"]), q["question"]
            )
            context_text = "\n".join(d.page_content for d in docs)
            prompt = (
                "You are an evaluator. Using ONLY the provided context, score the "
                "faithfulness of the answer on a scale from 0 to 1, where 1 means "
                "every claim in the answer is supported by the context and 0 means "
                "the answer is hallucinated. Output only the number.\n\n"
                f"CONTEXT:\n{context_text}\n\nANSWER:\n{_answer(llm, q['question'], context_text)}"
            )
            try:
                score = float(llm.invoke(prompt).content.strip())
                scores.append(min(max(score, 0.0), 1.0))
            except Exception:
                continue
        return {"n": len(scores), "mean": _mean(scores)} if scores else None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _answer(llm, question: str, context_text: str) -> str:
    prompt = (
        "You are an Elite Research AI. Answer using ONLY the provided context. "
        "If it is not in the context, say so. Keep it short.\n\n"
        f"CONTEXT:\n{context_text}\n\nQuestion: {question}"
    )
    try:
        return llm.invoke(prompt).content
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Infiniflow evaluation harness")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip faithfulness (LLM) metrics even if a key is present",
    )
    args = parser.parse_args()

    report = evaluate(retrieval_only=args.no_generation)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Infiniflow Evaluation Report")
        print("============================")
        for r in report["queries"]:
            print(f'- {r["question"]}')
            print(f'    precision@5 (hybrid):        {r["p_at_5_hybrid"]:.2f}')
            print(f'    precision@5 (hybrid+rerank): {r["p_at_5_hybrid_reranked"]:.2f}')
            print(
                f'    recall@5  (hybrid+rerank):   {r["recall_at_5_hybrid_reranked"]:.2f}'
            )
            print(
                f'    mrr@5     (hybrid+rerank):   {r["mrr_at_5_hybrid_reranked"]:.2f}'
            )
            print(f'    expected sources found:      {r["sources_found"]}')
        h, hr = report["hybrid"]["mean"], report["hybrid_reranked"]["mean"]
        print(f"\nMean context_precision@5 hybrid:       {h:.2f}")
        print(f"Mean context_precision@5 hybrid+rerank: {hr:.2f}")
        imp = report.get("reranker_improvement")
        if imp is not None:
            print(f"Reranker impact (delta):                {imp:+.2f}")
        print(
            f'Mean recall@5 (hybrid+rerank):          {report["recall_at_5_hybrid_reranked"]["mean"]:.2f}'
        )
        print(
            f'Mean MRR@5 (hybrid+rerank):             {report["mrr_at_5_hybrid_reranked"]["mean"]:.2f}'
        )
        if report.get("faithfulness"):
            print(
                f'Mean faithfulness (LLM-as-judge):       {report["faithfulness"]["mean"]:.2f}'
            )

    base = report["hybrid"]["mean"]
    reranked_mean = report["hybrid_reranked"]["mean"]
    recall_ok = report["recall_at_5_hybrid_reranked"]["mean"] >= CI_RECALL_OK
    ok = (
        base >= CI_CONTEXT_PRECISION_OK
        and reranked_mean >= CI_CONTEXT_PRECISION_OK
        and recall_ok
    )
    if report.get("faithfulness"):
        ok = ok and report["faithfulness"]["mean"] >= CI_FAITHFULNESS_OK
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
