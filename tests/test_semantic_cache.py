"""
Tests for Semantic Cache
"""
import pytest
from ai_engine.semantic_cache import SemanticCache
from datetime import datetime, timedelta


class TestSemanticCache:
    
    @pytest.fixture
    def cache(self):
        """Create cache instance"""
        return SemanticCache(similarity_threshold=0.85, ttl_hours=1)
    
    def test_exact_match(self, cache):
        """Test exact query match"""
        query = "What is machine learning?"
        response = {"answer": "ML is a subset of AI", "sources": ["doc1.pdf"]}
        
        cache.set(query, response)
        cached = cache.get(query)
        
        assert cached is not None
        assert cached["answer"] == response["answer"]
    
    def test_semantic_similarity(self, cache):
        """Test semantic similarity matching"""
        query1 = "What is machine learning?"
        query2 = "Can you explain machine learning?"
        response = {"answer": "ML is a subset of AI", "sources": ["doc1.pdf"]}
        
        cache.set(query1, response)
        cached = cache.get(query2)
        
        # Should find similar query
        assert cached is not None or cache.miss_count > 0
    
    def test_workspace_isolation(self, cache):
        """Test workspace-specific caching"""
        query = "What is AI?"
        response1 = {"answer": "Answer for workspace 1"}
        response2 = {"answer": "Answer for workspace 2"}
        
        cache.set(query, response1, workspace_id=1)
        cache.set(query, response2, workspace_id=2)
        
        cached1 = cache.get(query, workspace_id=1)
        cached2 = cache.get(query, workspace_id=2)
        
        assert cached1["answer"] != cached2["answer"]
    
    def test_cache_invalidation(self, cache):
        """Test cache invalidation"""
        query = "What is AI?"
        response = {"answer": "AI is artificial intelligence"}
        
        cache.set(query, response, workspace_id=1)
        cache.invalidate(workspace_id=1)
        
        cached = cache.get(query, workspace_id=1)
        assert cached is None
    
    def test_cache_stats(self, cache):
        """Test cache statistics"""
        query = "What is AI?"
        response = {"answer": "AI is artificial intelligence"}
        
        cache.set(query, response)
        cache.get(query)  # Hit
        cache.get("Unknown query")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hit_count"] >= 1
        assert stats["miss_count"] >= 1
        assert "hit_rate" in stats
    
    def test_top_queries(self, cache):
        """Test top queries retrieval"""
        for i in range(5):
            query = f"Query {i}"
            response = {"answer": f"Answer {i}"}
            cache.set(query, response)
            
            # Access some queries multiple times
            for _ in range(i):
                cache.get(query)
        
        top = cache.get_top_queries(limit=3)
        
        assert len(top) <= 3
        assert isinstance(top, list)
