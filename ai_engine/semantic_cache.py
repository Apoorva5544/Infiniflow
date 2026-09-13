"""
Semantic Caching System
- Cache similar queries using embeddings
- Reduce API costs and latency
- Smart cache invalidation
- Gracefully degrades if embeddings are unavailable
- Production backend: Redis (REDIS_URL / UPSTASH_REDIS_REST_URL) with an
  automatic fallback to a process-local in-memory cache when Redis is absent.
"""

import hashlib
import json
import os
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
    # HuggingFaceEmbeddings pulls in torch + sentence-transformers (~400MB+ RSS).
    # On a 512MB free-tier instance that single load OOMs the webservice, so
    # semantic *similarity* matching is opt-in. Exact-match caching (which needs
    # no model) stays enabled by default via ENABLE_SEMANTIC_SIMILARITY=0.
    if (
        os.getenv("ENABLE_SEMANTIC_SIMILARITY", "false").strip().lower()
        not in {"1", "true", "yes"}
    ):
        _embeddings = False
        return False
    try:
        import numpy as _numpy
        from sklearn.metrics.pairwise import cosine_similarity as _cs
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _np = _numpy
        _cosine_similarity = _cs
    except Exception as e:
        print(
            f"[SemanticCache] Embedding model unavailable ({e}). Cache will use exact-match only."
        )
        _embeddings = False  # Sentinel: tried and failed
    return _embeddings


class SemanticCache:
    """
    Semantic cache that matches similar queries.
    Falls back to exact-match if embedding model cannot be loaded.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        ttl_hours: int = 24,
        max_entries: int = 256,
    ):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        # Bounded cache: entries each carry the full response text (and possibly
        # an embedding), so an unbounded dict is a slow memory leak. Evict the
        # least-recently-accessed entry once full.
        self.max_entries = max_entries
        self.hit_count = 0
        self.miss_count = 0

    def _generate_key(self, query: str, workspace_id: Optional[int] = None) -> str:
        # Include workspace_id so the same question asked in different
        # workspaces never collides on a single cache entry.
        return hashlib.sha256(f"{workspace_id or ''}:{query}".encode()).hexdigest()

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.utcnow() - timestamp > self.ttl

    def get(
        self, query: str, workspace_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        # Exact match first (no embedding needed)
        exact_key = self._generate_key(query, workspace_id)
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
                if (
                    similarity > best_similarity
                    and similarity >= self.similarity_threshold
                ):
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

    def set(
        self, query: str, response: Dict[str, Any], workspace_id: Optional[int] = None
    ):
        key = self._generate_key(query, workspace_id)
        embedding = None
        emb = _get_embeddings()
        if emb:
            try:
                embedding = emb.embed_query(query)
            except Exception:
                pass

        if len(self.cache) >= self.max_entries:
            oldest_key = min(
                self.cache, key=lambda k: self.cache[k].get("last_accessed")
            )
            del self.cache[oldest_key]

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
            to_remove = [
                k
                for k, v in self.cache.items()
                if v.get("workspace_id") == workspace_id
            ]
            for k in to_remove:
                del self.cache[k]

    def cleanup_expired(self):
        to_remove = [
            k for k, v in self.cache.items() if self._is_expired(v["timestamp"])
        ]
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
        sorted_entries = sorted(
            self.cache.values(), key=lambda x: x["hit_count"], reverse=True
        )
        return [
            {
                "query": e["query"],
                "hit_count": e["hit_count"],
                "last_accessed": e["last_accessed"].isoformat(),
            }
            for e in sorted_entries[:limit]
        ]


# Global cache instance
_global_cache = None

_CACHE_PREFIX = "infiniflow:cache"


def _redis_from_url():
    """Creates a Redis client from REDIS_URL (or Upstash REST URL).

    Returns None (and logs) when Redis is not configured or unreachable so the
    calling code falls back to the in-memory cache transparently.
    """
    url = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
    if not url:
        return None
    try:
        import redis as _r

        client = _r.Redis.from_url(
            url,
            socket_timeout=3,
            socket_connect_timeout=3,
            health_check_interval=5,
            decode_responses=True,
        )
        client.ping()
        return client
    except Exception as e:
        print(f"[SemanticCache] Redis unavailable ({e}); using in-memory cache.")
        return None


def _cache_namespace(workspace_id: Optional[int]) -> str:
    return f"{_CACHE_PREFIX}:ws:{workspace_id or 'global'}"


class RedisSemanticCache:
    """
    Redis-backed semantic cache. Keys are scoped per workspace, entries carry a
    TTL handled natively by Redis, and semantic matching still happens in-process
    (no RediSearch module required) by scanning the workspace keyspace.
    """

    def __init__(
        self,
        redis_client: Any = None,
        similarity_threshold: float = 0.85,
        ttl_hours: int = 24,
    ):
        self.r = redis_client if redis_client is not None else _redis_from_url()
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.hit_count = 0
        self.miss_count = 0
        if self.r is None:
            print("[SemanticCache] No Redis client available; reads/writes are no-ops.")

    def __bool__(self):
        return self.r is not None

    def _generate_key(self, query: str, workspace_id: Optional[int] = None) -> str:
        return (
            f"{_cache_namespace(workspace_id)}:"
            + hashlib.sha256(f"{workspace_id or ''}:{query}".encode()).hexdigest()
        )

    def _serialize(self, entry: Dict[str, Any]) -> str:
        wire = dict(entry)
        for field in ("timestamp", "last_accessed"):
            if isinstance(wire.get(field), datetime):
                wire[field] = wire[field].isoformat()
        return json.dumps(wire)

    def _deserialize(self, raw: str) -> Optional[Dict[str, Any]]:
        entry = json.loads(raw)
        for field in ("timestamp", "last_accessed"):
            if isinstance(entry.get(field), str):
                entry[field] = datetime.fromisoformat(entry[field])
        return entry

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.utcnow() - timestamp > self.ttl

    def _scan_keys(self, workspace_id: Optional[int] = None):
        if self.r is None:
            return
        if workspace_id is None:
            pattern = f"{_CACHE_PREFIX}:*"
        else:
            pattern = f"{_cache_namespace(workspace_id)}:*"
        yield from self.r.scan_iter(match=pattern, count=100)

    def get(
        self, query: str, workspace_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        if self.r is None:
            return None

        # Exact match first (no embedding needed)
        key = self._generate_key(query, workspace_id)
        raw = self.r.get(key)
        if raw:
            try:
                entry = self._deserialize(raw)
                if not self._is_expired(entry["timestamp"]):
                    self.hit_count += 1
                    entry["hit_count"] += 1
                    entry["last_accessed"] = datetime.utcnow()
                    self.r.setex(
                        key, int(self.ttl.total_seconds()), self._serialize(entry)
                    )
                    return entry["response"]
                self.r.delete(key)
            except Exception as e:
                print(f"[SemanticCache] Redis exact-match error: {e}")

        # Semantic similarity search (optional)
        emb = _get_embeddings()
        if not emb:
            self.miss_count += 1
            return None

        try:
            query_embedding = emb.embed_query(query)
            best_match = None
            best_similarity = 0.0

            for k in self._scan_keys(workspace_id):
                raw_e = self.r.get(k)
                if not raw_e:
                    continue
                entry = self._deserialize(raw_e)
                if self._is_expired(entry["timestamp"]):
                    self.r.delete(k)
                    continue
                cached_emb = entry.get("embedding")
                if cached_emb is None:
                    continue
                similarity = _cosine_similarity([query_embedding], [cached_emb])[0][0]
                if (
                    similarity > best_similarity
                    and similarity >= self.similarity_threshold
                ):
                    best_similarity = similarity
                    best_match = (k, entry)

            if best_match:
                key, entry = best_match
                self.hit_count += 1
                entry["hit_count"] += 1
                entry["last_accessed"] = datetime.utcnow()
                self.r.setex(key, int(self.ttl.total_seconds()), self._serialize(entry))
                return entry["response"]
        except Exception as e:
            print(f"[SemanticCache] Redis similarity search error: {e}")

        self.miss_count += 1
        return None

    def set(
        self, query: str, response: Dict[str, Any], workspace_id: Optional[int] = None
    ):
        if self.r is None:
            return
        embedding = None
        emb = _get_embeddings()
        if emb:
            try:
                embedding = emb.embed_query(query)
            except Exception:
                pass

        entry = {
            "query": query,
            "response": response,
            "embedding": embedding,
            "workspace_id": workspace_id,
            "timestamp": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "hit_count": 0,
        }
        key = self._generate_key(query, workspace_id)
        self.r.setex(key, int(self.ttl.total_seconds()), self._serialize(entry))

    def invalidate(self, workspace_id: Optional[int] = None):
        if self.r is None:
            return
        for k in self._scan_keys(workspace_id):
            self.r.delete(k)
        if workspace_id is None:
            self.hit_count = 0
            self.miss_count = 0

    def cleanup_expired(self):
        """Redis evicts expired keys natively (TTL); kept for interface parity."""
        if self.r is None:
            return
        for k in self._scan_keys(None):
            if self.r.ttl(k) == -1:
                entry = self.r.get(k)
                if entry and self._is_expired(self._deserialize(entry)["timestamp"]):
                    self.r.delete(k)

    def get_stats(self) -> Dict[str, Any]:
        size = 0
        if self.r is not None:
            size = sum(1 for _ in self._scan_keys(None))
        total = self.hit_count + self.miss_count
        return {
            "backend": "redis",
            "size": size,
            "total_entries": size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total * 100, 2) if total else 0,
            "total_requests": total,
            "semantic_search_available": bool(_get_embeddings()),
        }

    def get_top_queries(self, limit: int = 10) -> list:
        if self.r is None:
            return []
        entries = []
        for k in self._scan_keys(None):
            raw_e = self.r.get(k)
            if not raw_e:
                continue
            entry = self._deserialize(raw_e)
            entries.append(entry)
        entries.sort(key=lambda x: x.get("hit_count", 0), reverse=True)
        return [
            {
                "query": e["query"],
                "hit_count": e["hit_count"],
                "last_accessed": e.get("last_accessed", e.get("timestamp")).isoformat(),
            }
            for e in entries[:limit]
        ]


def get_semantic_cache():
    """Returns a Redis-backed cache when Redis is configured, else in-memory."""
    global _global_cache
    if _global_cache is None:
        client = _redis_from_url()
        if client is not None:
            _global_cache = RedisSemanticCache(redis_client=client)
        else:
            _global_cache = SemanticCache()
    return _global_cache
