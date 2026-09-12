"""
Production-Grade FastAPI Backend
- Real JWT authentication on all protected routes
- Async document ingestion via Celery
- Hybrid RAG with ChromaDB + BM25 (60/40 EnsembleRetriever)
- History-aware query reformulation
- Rate limiting, CORS, structured logging
"""

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import List, Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel, EmailStr
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os, shutil, tempfile, time, logging, json, asyncio

from . import models, auth
from .database import engine, get_db
from .config import settings
from .analytics import AnalyticsEngine

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import (
    process_document,
    create_vector_store,
    get_vector_store,
    get_hybrid_retriever,
    get_qa_chain,
    list_collections,
    delete_collection,
)
from ai_engine.semantic_cache import get_semantic_cache
from ai_engine.reranker import CrossEncoderReranker

# ── Bootstrap ──────────────────────────────────────────────────────────────────
models.Base.metadata.create_all(bind=engine)

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

semantic_cache = get_semantic_cache()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade RAG platform — Hybrid Retriever (ChromaDB + BM25), JWT Auth, Celery async ingestion",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str


class UserLogin(BaseModel):
    email: str
    password: str


class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    llm_model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0


class QueryRequest(BaseModel):
    question: str
    chat_history: List[Dict] = []
    strategy: str = "auto"
    use_cache: bool = True


class AgentTaskRequest(BaseModel):
    task: str
    task_type: str = "auto"


# ── Citation / SSE helpers ─────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = (
    "You are an Elite Research AI. Answer the question using ONLY the provided CONTEXT. "
    "If the answer isn't in the CONTEXT, state that clearly. "
    "Cite the source of every factual claim inline using the numbered markers "
    "like [1], [2] where the number refers to the source list in the CONTEXT. "
    "Structure your response with clear headings or bullet points if appropriate.\n\n"
    "CONTEXT:\n{context}"
)

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference context "
    "in the chat history, formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just reformulate it if "
    "needed and otherwise return it as is."
)


def _sse(event: str, data: dict) -> str:
    """Encode a single Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _to_history(chat_history: List[Dict]) -> List:
    """Convert [{role, content}] request history into LangChain messages."""
    out = []
    for m in chat_history:
        role = str(m.get("role") or m.get("type") or "").lower()
        content = m.get("content", "")
        if role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
    return out


def _build_citations(docs) -> List[Dict[str, Any]]:
    """Deduplicated, chunk-level citations with source, page, snippet and score."""
    citations: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page")
        key = f"{source}:{page}:{doc.page_content[:60]}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": source,
                "page": page,
                "chunk_text": doc.page_content[:500],
                "relevance_score": meta.get("rerank_score"),
            }
        )
    return citations


def _numbered_context(docs) -> str:
    """Format reranked docs as a numbered source list the LLM can cite as [1], [2]."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        loc = f" (page {meta.get('page')})" if meta.get("page") is not None else ""
        parts.append(
            f"[{i}] Source: {meta.get('source', 'Unknown')}{loc}: {doc.page_content}"
        )
    return "\n\n".join(parts)


# ── AUTH ───────────────────────────────────────────────────────────────────────


@app.post("/api/v1/auth/signup", tags=["Authentication"])
@limiter.limit("5/minute")
async def signup(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    if db.query(models.User).filter(models.User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        email=user.email,
        hashed_password=auth.get_password_hash(user.password),
        full_name=user.full_name,
        role=models.UserRole.USER.value,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"New user: {user.email}")
    return {"message": "User created successfully", "user_id": new_user.id}


@app.post("/api/v1/auth/login", tags=["Authentication"])
@limiter.limit("10/minute")
async def login(
    request: Request, credentials: UserLogin, db: Session = Depends(get_db)
):
    """Login and receive a JWT access token."""
    user = db.query(models.User).filter(models.User.email == credentials.email).first()
    if not user or not auth.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")

    user.last_login = datetime.utcnow()
    db.commit()

    token = auth.create_access_token(data={"sub": user.email, "id": user.id})
    logger.info(f"Login: {credentials.email}")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


@app.get("/api/v1/auth/me", tags=["Authentication"])
async def get_me(current_user: models.User = Depends(auth.get_current_user)):
    """Return the authenticated user's profile."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "total_queries": current_user.total_queries,
        "total_documents": current_user.total_documents,
        "created_at": current_user.created_at.isoformat(),
    }


# ── WORKSPACES ─────────────────────────────────────────────────────────────────


@app.post("/api/v1/workspaces", tags=["Workspaces"])
@limiter.limit("20/minute")
async def create_workspace(
    request: Request,
    workspace: WorkspaceCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Create a new isolated Knowledge Layer (workspace)."""
    new_ws = models.Workspace(
        name=workspace.name,
        description=workspace.description,
        owner_id=current_user.id,
        llm_model=workspace.llm_model,
        temperature=workspace.temperature,
    )
    db.add(new_ws)
    db.commit()
    db.refresh(new_ws)

    if settings.VECTOR_STORE == "chroma":
        os.makedirs(
            os.path.join(settings.CHROMA_PATH, f"ws_{new_ws.id}"), exist_ok=True
        )
    logger.info(
        f"Workspace created: {workspace.name} (ID: {new_ws.id}) by {current_user.email}"
    )
    return {
        "id": new_ws.id,
        "name": new_ws.name,
        "description": new_ws.description,
        "created_at": new_ws.created_at.isoformat(),
    }


@app.get("/api/v1/workspaces", tags=["Workspaces"])
async def list_workspaces(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """List all workspaces owned by the authenticated user."""
    workspaces = (
        db.query(models.Workspace)
        .filter(
            models.Workspace.owner_id == current_user.id,
            models.Workspace.is_active == True,
        )
        .all()
    )
    return [
        {
            "id": ws.id,
            "name": ws.name,
            "description": ws.description,
            "total_documents": ws.total_documents,
            "total_queries": ws.total_queries,
            "llm_model": ws.llm_model,
            "created_at": ws.created_at.isoformat(),
        }
        for ws in workspaces
    ]


@app.get("/api/v1/workspaces/{workspace_id}", tags=["Workspaces"])
async def get_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Get workspace details including document list."""
    ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    documents = (
        db.query(models.Document)
        .filter(models.Document.workspace_id == workspace_id)
        .all()
    )
    return {
        "id": ws.id,
        "name": ws.name,
        "description": ws.description,
        "llm_model": ws.llm_model,
        "temperature": ws.temperature,
        "total_documents": ws.total_documents,
        "total_queries": ws.total_queries,
        "created_at": ws.created_at.isoformat(),
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "chunk_count": d.chunk_count,
                "status": d.status,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in documents
        ],
    }


@app.delete("/api/v1/workspaces/{workspace_id}", tags=["Workspaces"])
async def delete_workspace(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Soft-delete a workspace and remove its vector store."""
    ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    ws.is_active = False
    db.commit()

    # Remove ChromaDB collection
    try:
        delete_collection(f"ws_{workspace_id}")
    except Exception as e:
        logger.warning(f"Could not delete vector store for ws_{workspace_id}: {e}")

    logger.info(f"Workspace {workspace_id} deleted by {current_user.email}")
    return {"message": "Workspace deleted successfully"}


# ── DOCUMENTS ─────────────────────────────────────────────────────────────────


@app.post("/api/v1/workspaces/{workspace_id}/upload", tags=["Documents"])
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    workspace_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Upload and synchronously process a PDF into the workspace Knowledge Layer."""
    ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    content = await file.read()
    file_size = len(content)

    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Persist temp file
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    # Insert DB record as "processing"
    doc = models.Document(
        filename=file.filename,
        file_path=temp_path,
        file_type=file.content_type or "application/pdf",
        file_size=file_size,
        workspace_id=workspace_id,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        start = time.time()
        chunks = process_document(temp_path)
        collection_name = f"ws_{workspace_id}"
        create_vector_store(chunks, collection_name)
        elapsed = time.time() - start

        doc.chunk_count = len(chunks)
        doc.processing_time = elapsed
        doc.status = "processed"
        ws.total_documents += 1
        db.commit()

        semantic_cache.invalidate(workspace_id=workspace_id)
        logger.info(
            f"Ingested: {file.filename} → {len(chunks)} chunks ({elapsed:.2f}s)"
        )

        return {
            "message": f"Successfully ingested {file.filename}",
            "document_id": doc.id,
            "chunks": len(chunks),
            "processing_time_seconds": round(elapsed, 2),
        }

    except Exception as e:
        doc.status = "failed"
        db.commit()
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Delete a document record (does not remove from vector store)."""
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    ws = (
        db.query(models.Workspace)
        .filter(models.Workspace.id == doc.workspace_id)
        .first()
    )
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if ws.total_documents > 0:
        ws.total_documents -= 1
    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}


# ── QUERYING ───────────────────────────────────────────────────────────────────


@app.post("/api/v1/workspaces/{workspace_id}/query", tags=["Query"])
@limiter.limit("30/minute")
async def query_workspace(
    request: Request,
    workspace_id: int,
    query_req: QueryRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """
    Query a Knowledge Layer using the Hybrid Retriever (ChromaDB 60% + BM25 40%)
    with history-aware query reformulation.
    """
    ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Semantic cache check
    if query_req.use_cache:
        cached = semantic_cache.get(query_req.question, workspace_id)
        if cached:
            logger.info(f"Cache hit: {query_req.question[:50]}")
            return {**cached, "cached": True, "cache_stats": semantic_cache.get_stats()}

    start = time.time()

    try:
        collection_name = f"ws_{workspace_id}"
        vs = get_vector_store(collection_name)
        if not vs:
            raise HTTPException(
                status_code=404, detail="No documents in this Knowledge Layer"
            )

        # Advanced RAG: "auto" = direct retrieval, no extra LLM round-trips
        strategy = query_req.strategy if query_req.strategy != "auto" else "simple"
        logger.info(f"Query started [{strategy}]: {query_req.question[:50]}")

        # Hybrid retriever (60/40 vector/BM25) + history-aware chain
        retriever = get_hybrid_retriever(vs)
        chain = get_qa_chain(vs, retriever)
        response = chain.invoke(
            {
                "input": query_req.question,
                "chat_history": query_req.chat_history,
            }
        )
        documents = response.get("context", []) or []

        latency = (time.time() - start) * 1000

        # Local relevance scoring (keyword overlap) — avoids 5 extra LLM calls
        q_terms = {w for w in query_req.question.lower().split() if len(w) > 2}
        if documents and q_terms:
            overlaps = [
                len(q_terms & set(doc.page_content.lower().split())) / len(q_terms)
                for doc in documents[:5]
            ]
            relevance_score = min(10.0, (sum(overlaps) / len(overlaps)) * 10)
        else:
            relevance_score = 0.0

        sources = list(
            set(
                [
                    doc.metadata.get("source", "Unknown")
                    for doc in response.get("context", [])
                ]
            )
        )

        result = {
            "answer": response["answer"],
            "sources": sources,
            "citations": _build_citations(response.get("context", [])),
            "strategy_used": strategy,
            "documents_retrieved": len(documents),
            "latency_ms": round(latency, 2),
            "relevance_score": round(relevance_score, 2),
            "reranker": os.getenv("ENABLE_RERANKER", "true").lower()
            in {"1", "true", "yes"},
            "cached": False,
        }

        # Log to DB
        query_log = models.QueryLog(
            user_id=current_user.id,
            workspace_id=workspace_id,
            query_text=query_req.question,
            response_text=response["answer"],
            status=models.QueryStatus.COMPLETED.value,
            latency_ms=latency,
            tokens_used=len(response["answer"].split()) * 2,
            documents_retrieved=len(documents),
            model_used=ws.llm_model,
            retrieval_score=relevance_score,
            sources=sources,
        )
        db.add(query_log)
        ws.total_queries += 1
        current_user.total_queries += 1
        db.commit()

        if query_req.use_cache:
            semantic_cache.set(query_req.question, result, workspace_id)

        logger.info(
            f"Query OK [{strategy}]: {query_req.question[:50]} ({latency:.1f}ms)"
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Query error: {e}")
        fail_log = models.QueryLog(
            user_id=current_user.id,
            workspace_id=workspace_id,
            query_text=query_req.question,
            response_text=str(e),
            status=models.QueryStatus.FAILED.value,
            latency_ms=latency,
            model_used=ws.llm_model,
        )
        db.add(fail_log)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


# ── STREAMING QUERY ──────────────────────────────────────────────────────────


@app.post("/api/v1/workspaces/{workspace_id}/query/stream", tags=["Query"])
@limiter.limit("30/minute")
async def query_workspace_stream(
    request: Request,
    workspace_id: int,
    query_req: QueryRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Stream a RAG answer with SSE. Retrieval + rerank happen first, then the
    sources are pushed as an event, followed by streamed LLM tokens."""
    ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if ws.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Semantic cache: replay sources + answer as an instant stream
    if query_req.use_cache:
        cached = semantic_cache.get(query_req.question, workspace_id)
        if cached:

            async def cached_stream() -> AsyncGenerator[str, None]:
                yield _sse(
                    "sources",
                    {
                        "type": "sources",
                        "citations": cached.get("citations", []),
                        "strategy_used": cached.get("strategy_used", "simple"),
                        "cached": True,
                    },
                )
                yield _sse("token", {"type": "token", "content": cached["answer"]})
                yield _sse(
                    "done",
                    {
                        "type": "done",
                        "cached": True,
                        "latency_ms": 0,
                        "cache_stats": semantic_cache.get_stats(),
                    },
                )

            return StreamingResponse(cached_stream(), media_type="text/event-stream")

    start = time.time()
    llm_key = os.getenv("GROQ_API_KEY", "").strip("\"' ")
    history = _to_history(query_req.chat_history)
    use_rerank = os.getenv("ENABLE_RERANKER", "true").lower() in {"1", "true", "yes"}
    reranker_top_k = int(os.getenv("RERANKER_TOP_K", "5"))
    collection_name = f"ws_{workspace_id}"

    # Stream generation. The first byte is sent IMMEDIATELY, then retrieval,
    # reformulation and generation run inline. Keeping data flowing means
    # Render's proxy never applies its whole-response timeout to this route —
    # the #1 cause of the 502s on the sync endpoint during free-tier cold starts.
    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse("start", {"type": "start"})

        # 2. Retrieval (inside the stream, so first-byte latency is ~zero)
        try:
            vs = get_vector_store(collection_name)
            if not vs:
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "detail": "No documents in this Knowledge Layer",
                    },
                )
                return

            llm = ChatGroq(
                temperature=ws.temperature or 0.0,
                model_name=ws.llm_model,
                groq_api_key=llm_key,
            )

            # History-aware reformulation (standalone question)
            standalone = query_req.question
            if history:
                try:
                    resp = await asyncio.to_thread(
                        llm.invoke,
                        [
                            SystemMessage(CONTEXTUALIZE_SYSTEM_PROMPT),
                            *history,
                            HumanMessage(query_req.question),
                        ],
                    )
                    standalone = (resp.content or "").strip() or query_req.question
                except Exception as e:
                    logger.warning(f"Reformulation failed, using raw question: {e}")

            retriever = get_hybrid_retriever(vs)
            docs = await asyncio.to_thread(retriever.invoke, standalone)
            if use_rerank:
                reranked = await asyncio.to_thread(
                    CrossEncoderReranker(top_k=reranker_top_k).compress_documents,
                    docs,
                    standalone,
                )
            else:
                reranked = []
                for i, d in enumerate(docs[:reranker_top_k]):
                    d.metadata["rerank_rank"] = i
                    d.metadata["rerank_score"] = None
                    reranked.append(d)
            if not reranked:
                reranked = docs[:reranker_top_k]

            citations = _build_citations(reranked)
            context_block = _numbered_context(reranked)
            messages = [
                SystemMessage(QA_SYSTEM_PROMPT.format(context=context_block)),
                *history,
                HumanMessage(query_req.question),
            ]

            yield _sse(
                "sources",
                {
                    "type": "sources",
                    "citations": citations,
                    "strategy_used": "hybrid" if not use_rerank else "hybrid+rerank",
                    "documents_retrieved": len(reranked),
                    "cached": False,
                },
            )
        except HTTPException as e:
            logger.error(f"Retrieval failed for streaming query: {e.detail}")
            yield _sse("error", {"type": "error", "detail": e.detail})
            return
        except Exception as e:
            logger.error(f"Retrieval failed for streaming query: {e}")
            yield _sse(
                "error", {"type": "error", "detail": f"Retrieval failed: {str(e)}"}
            )
            return

        # 3. Stream generation
        answer_parts: List[str] = []
        error_detail = None
        try:
            async for chunk in llm.astream(messages):
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    yield _sse("token", {"type": "token", "content": content})
        except Exception as e:
            error_detail = str(e)
            logger.error(f"Streaming error: {e}")
            yield _sse("error", {"type": "error", "detail": error_detail})

        latency_ms = (time.time() - start) * 1000
        answer = "".join(answer_parts)
        status_val = (
            models.QueryStatus.FAILED.value
            if error_detail
            else models.QueryStatus.COMPLETED.value
        )

        log_row = models.QueryLog(
            user_id=current_user.id,
            workspace_id=workspace_id,
            query_text=query_req.question,
            response_text=(answer or error_detail or "")[:5000],
            status=status_val,
            latency_ms=latency_ms,
            tokens_used=len(answer.split()) * 2,
            documents_retrieved=len(reranked),
            model_used=ws.llm_model,
            retrieval_score=(
                round(citations[0]["relevance_score"], 2)
                if citations and citations[0]["relevance_score"] is not None
                else None
            ),
            sources=[c["source"] for c in citations],
        )
        db.add(log_row)
        ws.total_queries += 1
        current_user.total_queries += 1
        db.commit()

        if query_req.use_cache and not error_detail:
            semantic_cache.set(
                query_req.question,
                {
                    "answer": answer,
                    "sources": [c["source"] for c in citations],
                    "citations": citations,
                    "strategy_used": "hybrid" if not use_rerank else "hybrid+rerank",
                },
                workspace_id,
            )

        yield _sse(
            "done",
            {
                "type": "done",
                "latency_ms": round(latency_ms, 2),
                "cached": False,
                "error": error_detail,
            },
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── ANALYTICS ─────────────────────────────────────────────────────────────────


@app.get("/api/v1/workspaces/{workspace_id}/analytics", tags=["Analytics"])
async def get_workspace_analytics(
    workspace_id: int,
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return AnalyticsEngine.get_workspace_analytics(db, workspace_id, days)


@app.get("/api/v1/workspaces/{workspace_id}/insights", tags=["Analytics"])
async def get_query_insights(
    workspace_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return AnalyticsEngine.get_query_insights(db, workspace_id)


@app.get("/api/v1/analytics/platform", tags=["Analytics"])
async def get_platform_analytics(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return AnalyticsEngine.get_platform_analytics(db, days)


# ── CACHE ──────────────────────────────────────────────────────────────────────


@app.get("/api/v1/cache/stats", tags=["System"])
async def get_cache_stats(current_user: models.User = Depends(auth.get_current_user)):
    return semantic_cache.get_stats()


@app.get("/api/v1/cache/top-queries", tags=["System"])
async def get_top_queries(
    limit: int = 10,
    current_user: models.User = Depends(auth.get_current_user),
):
    return semantic_cache.get_top_queries(limit)


# ── HEALTH ─────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "vector_store": settings.VECTOR_STORE,
        "cache_backend": type(semantic_cache).__name__,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Frontend (single-container deployment) ─────────────────────────────────────
# Mount the built React app last so all API routes registered above take
# priority. Runs only when the Vite `dist` output is present (i.e. the
# multi-stage Docker image), so local API-only runs are unaffected.
_SPA_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist"
)
if os.path.isdir(_SPA_DIST):
    app.mount("/", StaticFiles(directory=_SPA_DIST, html=True), name="frontend")
else:
    logger.info(
        "frontend/dist not found — API-only mode (use the Vite dev server for the UI)"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main_v2:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
