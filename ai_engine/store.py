"""
Vector Store Facade
Backends (select via VECTOR_STORE env var):
  - "chroma"   (default): local ChromaDB persistent collections — zero infra,
    ideal for development, single-process workloads and `streamlit run app.py`.
  - "pgvector" : PostgreSQL + pgvector ANN — production grade, concurrent writes,
    horizontally scalable, survives process restarts without local state.

The public functions expose the same signatures as the legacy Chroma helpers so
the rest of the platform (FastAPI backend + Streamlit app) is backend-agnostic.
"""

import os
import shutil
from threading import Lock
from typing import List

from dotenv import load_dotenv

# Load .env before reading VECTOR_STORE — import order must not decide the
# backend (rag_engine's load_dotenv runs after this module is imported).
load_dotenv(override=True)

# In-process vector-store cache. Reconnecting / re-counting the store on every
# request is wasteful (and slow on free-tier CPU) — reuse the loaded object
# and invalidate it the moment the collection is written to or deleted.
_vector_store_cache: dict = {}
_vector_store_lock = Lock()
_MAX_CACHED_STORES = 8


def _store_type() -> str:
    st = os.getenv("VECTOR_STORE", "chroma").strip().lower()
    if st in {"", "chroma"}:
        # Render free-tier instances have NO persistent disk: a local chroma_db
        # is wiped on every deploy/restart (the docs "disappear" and queries
        # fail with "no documents"). Fail loudly instead of silently dropping
        # data. Set VECTOR_STORE=pgvector + DATABASE_URL (e.g. Neon) in prod.
        is_prod = os.getenv("ENVIRONMENT", "").strip().lower() in {
            "production",
            "prod",
        }
        allow_ephemeral = os.getenv(
            "ALLOW_EPHEMERAL_CHROMA", "false"
        ).strip().lower() in {"1", "true", "yes"}
        if is_prod and not allow_ephemeral:
            raise RuntimeError(
                "VECTOR_STORE is unset/'chroma' in a production environment — "
                "Chroma's local disk store is wiped on every restart and loses "
                "ingested documents. Set VECTOR_STORE=pgvector and DATABASE_URL "
                "(Neon), or set ALLOW_EPHEMERAL_CHROMA=1 to override."
            )
    return st


def _embeddings():
    # Deferred import: rag_engine re-exports these store functions, so importing
    # it at module scope would create a circular import.
    from rag_engine import _embeddings as _e

    return _e()


# ── Chroma backend (default / local) ─────────────────────────────────────────


def _chroma_path() -> str:
    return os.getenv("CHROMA_PATH", "./chroma_db")


def _chroma_create_vector_store(chunks, collection_name: str = "default"):
    from langchain_community.vectorstores import Chroma

    persist_dir = os.path.join(_chroma_path(), collection_name)
    return Chroma.from_documents(
        documents=chunks, embedding=_embeddings(), persist_directory=persist_dir
    )


def _chroma_get_vector_store(collection_name: str = "default"):
    from langchain_community.vectorstores import Chroma

    persist_dir = os.path.join(_chroma_path(), collection_name)
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=_embeddings())
    return None


def _chroma_delete_collection(collection_name: str) -> bool:
    persist_dir = os.path.join(_chroma_path(), collection_name)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        return True
    return False


def _chroma_list_collections() -> List[str]:
    path = _chroma_path()
    if not os.path.exists(path):
        return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


# ── PostgreSQL + pgvector backend (production) ───────────────────────────────


def _pg_connection_string() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _pg_engine():
    from sqlalchemy import create_engine

    return create_engine(_pg_connection_string(), pool_pre_ping=True, pool_recycle=1800)


def _pg_connect():
    db_url = _pg_connection_string()
    if not db_url.startswith("postgres"):
        raise RuntimeError(
            "VECTOR_STORE=pgvector requires DATABASE_URL pointing at PostgreSQL"
        )
    return db_url


def _pg_get_vector_store(collection_name: str):
    from langchain_community.vectorstores import PGVector
    from sqlalchemy import text

    db_url = _pg_connect()
    # A collection with zero rows is treated as "no knowledge layer yet",
    # mirroring Chroma's None-when-missing behaviour.
    engine = _pg_engine()
    with engine.connect() as conn:
        count = (
            conn.execute(
                text(
                    """
                SELECT COUNT(*) FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :name
                """
                ),
                {"name": collection_name},
            ).scalar()
            or 0
        )
    if not count:
        return None
    return PGVector(
        connection_string=db_url,
        embedding_function=_embeddings(),
        collection_name=collection_name,
        use_jsonb=True,
    )


def _pg_create_vector_store(chunks, collection_name: str = "default"):
    from langchain_community.vectorstores import PGVector

    db_url = _pg_connect()
    # Adds (never replaces) — incremental multi-file uploads behave like Chroma.
    return PGVector.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        connection_string=db_url,
        collection_name=collection_name,
        use_jsonb=True,
    )


def _pg_delete_collection(collection_name: str) -> bool:
    from sqlalchemy import text

    db_url = _pg_connect()
    engine = _pg_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :name
                )
                """
            ),
            {"name": collection_name},
        )
        conn.execute(
            text("DELETE FROM langchain_pg_collection WHERE name = :name"),
            {"name": collection_name},
        )
    return True


def _pg_list_collections() -> List[str]:
    from sqlalchemy import text

    _pg_connect()
    engine = _pg_engine()
    with engine.connect() as conn:
        return list(
            conn.execute(
                text("SELECT name FROM langchain_pg_collection ORDER BY name")
            ).scalars()
        )


def stored_documents(collection_name: str) -> List:
    """
    Recovers the persisted Documents of a pgvector collection straight from the
    database. Needed because PGVector only keeps an in-process docstore while
    *adding* documents — after a worker restart the store is empty and BM25
    would otherwise silently degrade to vector-only.
    """
    from langchain_core.documents import Document
    from sqlalchemy import text

    _pg_connect()
    engine = _pg_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :name
                """
            ),
            {"name": collection_name},
        ).all()
    return [
        Document(page_content=doc or "", metadata=(meta or {})) for doc, meta in rows
    ]


# ── Public facade ────────────────────────────────────────────────────────────


def create_vector_store(chunks, collection_name: str = "default"):
    """Creates/updates a named vector store (appends chunks)."""
    if _store_type() == "pgvector":
        store = _pg_create_vector_store(chunks, collection_name)
    else:
        store = _chroma_create_vector_store(chunks, collection_name)
    with _vector_store_lock:
        _vector_store_cache.pop(f"{_store_type()}:{collection_name}", None)
    return store


def get_vector_store(collection_name: str = "default"):
    """Returns the named vector store, or None if it has no documents.

    Results are cached in-process and evicted on write/delete so re-opening
    the store (and re-querying Postgres) doesn't happen on every request.
    """
    key = f"{_store_type()}:{collection_name}"
    with _vector_store_lock:
        if key in _vector_store_cache and _vector_store_cache[key] is not None:
            return _vector_store_cache[key]

    if _store_type() == "pgvector":
        try:
            store = _pg_get_vector_store(collection_name)
        except Exception as e:
            print(f"[store] pgvector load failed for {collection_name}: {e}")
            store = None
    else:
        store = _chroma_get_vector_store(collection_name)

    with _vector_store_lock:
        if store is not None:
            _vector_store_cache[key] = store
            if len(_vector_store_cache) > _MAX_CACHED_STORES:
                _vector_store_cache.pop(next(iter(_vector_store_cache)))
        return store


def delete_collection(collection_name: str) -> bool:
    """Deletes a named collection."""
    if _store_type() == "pgvector":
        ok = _pg_delete_collection(collection_name)
    else:
        ok = _chroma_delete_collection(collection_name)
    with _vector_store_lock:
        _vector_store_cache.pop(f"{_store_type()}:{collection_name}", None)
    return ok


def list_collections() -> List[str]:
    """Lists available knowledge base collections."""
    if _store_type() == "pgvector":
        return _pg_list_collections()
    return _chroma_list_collections()
