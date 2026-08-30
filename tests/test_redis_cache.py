"""
Tests for the Redis-backed semantic cache. Uses fakeredis so the suite runs
without a live Redis instance; embeddings are disabled so exact-match paths.
"""

import pytest
from ai_engine import semantic_cache as sc
from ai_engine.semantic_cache import RedisSemanticCache


@pytest.fixture
def cache(monkeypatch):
    import fakeredis

    monkeypatch.setattr(sc, "_get_embeddings", lambda: None)
    return RedisSemanticCache(redis_client=fakeredis.FakeRedis(decode_responses=True))


def test_backend_label(cache):
    stats = cache.get_stats()
    assert stats["backend"] == "redis"
    assert stats["semantic_search_available"] is False


def test_exact_match(cache):
    query = "What is machine learning?"
    response = {"answer": "ML is a subset of AI", "sources": ["doc1.pdf"]}
    cache.set(query, response)
    cached = cache.get(query)
    assert cached is not None
    assert cached["answer"] == response["answer"]


def test_workspace_isolation(cache):
    query = "What is AI?"
    cache.set(query, {"answer": "Answer for workspace 1"}, workspace_id=1)
    cache.set(query, {"answer": "Answer for workspace 2"}, workspace_id=2)
    assert cache.get(query, workspace_id=1)["answer"] == "Answer for workspace 1"
    assert cache.get(query, workspace_id=2)["answer"] == "Answer for workspace 2"


def test_invalidate_workspace(cache):
    query = "What is AI?"
    cache.set(query, {"answer": "AI is artificial intelligence"}, workspace_id=1)
    cache.invalidate(workspace_id=1)
    assert cache.get(query, workspace_id=1) is None


def test_stats_and_top_queries(cache):
    for i in range(3):
        cache.set(f"Query {i}", {"answer": f"Answer {i}"}, workspace_id=7)
        for _ in range(i + 1):
            cache.get(f"Query {i}", workspace_id=7)

    stats = cache.get_stats()
    assert stats["size"] == 3
    assert stats["hit_count"] >= 3

    top = cache.get_top_queries(limit=2)
    assert len(top) <= 2
    assert top[0]["query"] == "Query 2"  # hit 3x


def test_get_semantic_cache_prefers_redis(monkeypatch):
    import fakeredis

    monkeypatch.setattr(
        sc, "_redis_from_url", lambda: fakeredis.FakeRedis(decode_responses=True)
    )
    # Reset the module singleton before asserting backend selection.
    monkeypatch.setattr(sc, "_global_cache", None)
    assert isinstance(sc.get_semantic_cache(), RedisSemanticCache)
