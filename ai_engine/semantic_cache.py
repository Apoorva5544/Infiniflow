"""
Semantic Caching System
- Cache similar queries using embeddings
- Reduce API costs and latency
- Smart cache invalidation
- Gracefully degrades if embeddings are unavailable
"""
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

# Lazy import — torch/HuggingFace may be unavailable in some envs
_embeddings = None
_np = None
_cosine_similarity = None


def _get_embeddings():
    global _embeddings, _np, _cosine_similarity
    if _embeddings is not None:
        return _embeddings
    try:
        import numpy as _numpy
        from sklearn.metrics.pairwise import cosine_similarity as _cs
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _np = _numpy
        _cosine_similarity = _cs
    except Exception as e:
        print(f"[SemanticCache] Embedding model unavailable ({e}). Cache will use exact-match only.")
        _embeddings = False  # Sentinel: tried and failed
    return _embeddings


class SemanticCache:
    """
    Semantic cache that matches similar queries.
    Falls back to exact-match if embedding model cannot be loaded.
    """

    def __init__(self, similarity_threshold: float = 0.85, ttl_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.utcnow() - timestamp > self.ttl

    def get(self, query: str, workspace_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        # Exact match first (no embedding needed)
        exact_key = self._generate_key(query)
        if exact_key in self.cache:
            entry = self.cache[exact_key]
            if not self._is_expired(entry["timestamp"]):
                if workspace_id is None or entry.get("workspace_id") == workspace_id:
                    self.hit_count += 1
                    entry["hit_count"] += 1
                    entry["last_accessed"] = datetime.utcnow()
                    return entry["response"]

        # Semantic similarity search (optional)
        emb = _get_embeddings()
        if not emb:
            self.miss_count += 1
            return None

        try:
            query_embedding = emb.embed_query(query)
            best_match = None
            best_similarity = 0.0

            for entry in self.cache.values():
                if self._is_expired(entry["timestamp"]):
                    continue
                if workspace_id and entry.get("workspace_id") != workspace_id:
                    continue
                cached_emb = entry.get("embedding")
                if cached_emb is None:
                    continue
                similarity = _cosine_similarity([query_embedding], [cached_emb])[0][0]
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = entry

            if best_match:
                self.hit_count += 1
                best_match["hit_count"] += 1
                best_match["last_accessed"] = datetime.utcnow()
                return best_match["response"]
        except Exception as e:
            print(f"[SemanticCache] Similarity search error: {e}")

        self.miss_count += 1
        return None

    def set(self, query: str, response: Dict[str, Any], workspace_id: Optional[int] = None):
        key = self._generate_key(query)
        embedding = None
        emb = _get_embeddings()
        if emb:
            try:
                embedding = emb.embed_query(query)
            except Exception:
                pass

        self.cache[key] = {
            "query": query,
            "response": response,
            "embedding": embedding,
            "workspace_id": workspace_id,
            "timestamp": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "hit_count": 0,
        }

    def invalidate(self, workspace_id: Optional[int] = None):
        if workspace_id is None:
            self.cache.clear()
        else:
            to_remove = [k for k, v in self.cache.items() if v.get("workspace_id") == workspace_id]
            for k in to_remove:
                del self.cache[k]

    def cleanup_expired(self):
        to_remove = [k for k, v in self.cache.items() if self._is_expired(v["timestamp"])]
        for k in to_remove:
            del self.cache[k]

    def get_stats(self) -> Dict[str, Any]:
        total = self.hit_count + self.miss_count
        return {
            "size": len(self.cache),
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total * 100, 2) if total else 0,
            "total_requests": total,
            "semantic_search_available": bool(_get_embeddings()),
        }

    def get_top_queries(self, limit: int = 10) -> list:
        sorted_entries = sorted(self.cache.values(), key=lambda x: x["hit_count"], reverse=True)
        return [
            {
                "query": e["query"],
                "hit_count": e["hit_count"],
                "last_accessed": e["last_accessed"].isoformat(),
            }
            for e in sorted_entries[:limit]
        ]


# Global cache instance
_global_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = SemanticCache()
    return _global_cache
