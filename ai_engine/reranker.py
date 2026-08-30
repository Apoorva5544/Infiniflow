"""
Cross-Encoder Reranker
======================
Re-scores the candidate documents retrieved by the hybrid retriever
(vector + BM25) with a cross-encoder, then keeps the top_k most relevant.

Design notes:
- The cross-encoder is loaded lazily (first use) so importing this module
  never blocks startup or requires a GPU.
- If the model or its dependencies are unavailable (no network, first-run
  in CI, memory constraints), the system degrades gracefully to deterministic
  top-k truncation instead of crashing the query pipeline.
- Integrates with LangChain as a `BaseDocumentCompressor`, so it can be
  dropped straight into a `ContextualCompressionRetriever`.
"""

from typing import List, Optional
import math
import os

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level lazy state: (model, failed)
_model = None
_unavailable_reason: Optional[str] = None


def _is_unavailable() -> bool:
    global _model, _unavailable_reason
    if _model is not None:
        return False
    if _unavailable_reason is not None:
        return True
    if os.getenv("INFINIFLOW_SKIP_RERANKER", "").lower() in {"1", "true", "yes"}:
        _unavailable_reason = "disabled via INFINIFLOW_SKIP_RERANKER"
        return True
    try:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(DEFAULT_MODEL)
        return False
    except Exception as e:  # pragma: no cover - depends on environment
        _unavailable_reason = str(e)
        print(
            f"[Reranker] Cross-encoder unavailable ({e}); falling back to top-k truncation."
        )
        return True


class CrossEncoderReranker(BaseDocumentCompressor):
    """Rerank documents with a cross-encoder; degree to top-k truncation if unavailable."""

    top_k: int = 5
    max_chars: int = 512

    def compress_documents(
        self, documents: List[Document], query: str, **kwargs
    ) -> List[Document]:
        if not documents:
            return []
        if _is_unavailable():
            return _attach_rank(documents[: self.top_k], ranking="truncation")

        pairs = [[str(query), d.page_content[: self.max_chars]] for d in documents]
        try:
            scores = _model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            kept = [d for d, _ in ranked[: self.top_k]]
            for i, (d, s) in enumerate(ranked):
                d.metadata["rerank_score"] = round(_sigmoid(float(s)), 4)
                d.metadata["rerank_rank"] = i
            return kept
        except Exception as e:  # pragma: no cover - model.predict can fail per-GPU
            print(f"[Reranker] predict failed ({e}); falling back to top-k truncation.")
            return _attach_rank(documents[: self.top_k], ranking="truncation")


def _sigmoid(x: float) -> float:
    """Monotonic normalization of cross-encoder logits into (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _attach_rank(documents: List[Document], ranking: str) -> List[Document]:
    for i, d in enumerate(documents):
        d.metadata["rerank_rank"] = i
        d.metadata["rerank_score"] = None
        d.metadata["rerank_mode"] = ranking
    return documents


def rerank_documents(query, documents, top_k: int = 5):
    """Module-level helper. Accepts Documents (or objects with page_content/metadata)."""
    return CrossEncoderReranker(top_k=top_k).compress_documents(
        list(documents), str(query)
    )
