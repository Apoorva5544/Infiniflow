"""
Tests for the cross-encoder reranker (graceful-fallback behaviours).
Behaves deterministically and never downloads the model in CI.
"""

import pytest
from langchain_core.documents import Document

from ai_engine.reranker import CrossEncoderReranker, rerank_documents


@pytest.fixture(autouse=True)
def force_fallback(monkeypatch):
    """Simulate an environment where the cross-encoder cannot be loaded."""
    monkeypatch.setattr("ai_engine.reranker._is_unavailable", lambda: True)
    yield


def _docs(n=8):
    return [
        Document(
            page_content=f"relevant passage number {i} about pandas habitat",
            metadata={"source": "a.pdf", "page": i},
        )
        for i in range(n)
    ]


class TestCrossEncoderReranker:

    def test_truncates_to_top_k(self):
        docs = _docs(8)
        out = CrossEncoderReranker(top_k=3).compress_documents(docs, "pandas")
        assert len(out) == 3

    def test_keeps_all_when_fewer_than_top_k(self):
        docs = _docs(2)
        out = CrossEncoderReranker(top_k=5).compress_documents(docs, "pandas")
        assert len(out) == 2

    def test_empty_input(self):
        assert CrossEncoderReranker().compress_documents([], "x") == []

    def test_attaches_ranking_metadata(self):
        docs = _docs(4)
        out = CrossEncoderReranker(top_k=4).compress_documents(docs, "pandas")
        for i, d in enumerate(out):
            assert d.metadata["rerank_rank"] == i
            assert d.metadata["rerank_mode"] == "truncation"

    def test_preserves_original_documents(self):
        docs = _docs(3)
        out = rerank_documents("pandas", docs, top_k=2)
        assert out == docs[:2]
