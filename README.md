# Infiniflow — Production RAG Platform with Agentic Retrieval

> Multi-tenant RAG with hybrid search (vector + BM25), cross-encoder reranking,
> agentic retrieval strategies, real-time SSE streaming, source-level citations,
> and a built-in evaluation harness.

Infiniflow turns uploaded documents into queryable **knowledge layers** inside
isolated **workspaces**, protected by JWT-auth, and answers questions with
citations you can trace back to the exact chunk, page, and document.

---

## ✨ What makes this different

| Feature | Basic RAG tutorial | Infiniflow |
|---|---|---|
| Retrieval | Vector only | Hybrid (vector + BM25, 60/40 ensemble) |
| Precision | Top-k only | Cross-encoder **reranker** after retrieval |
| Reasoning | One-shot | Adaptive strategies: routing, multi-query, HyDE, decomposition |
| Citations | None | Chunk-level attribution (source, page, snippet, score) |
| Streaming | No | SSE with sources-first, then token stream |
| Caching | In-memory | Semantic cache (exact + embedding match) — **Redis-backed** with TTL & invalidation |
| Storage | SQLite | **PostgreSQL + pgvector** (HNSW ANN) with SQLite fallback for dev |
| Auth / tenancy | Optional | JWT on every route + per-user workspace isolation |
| Metrics | Guesswork | `evals/` harness with CI gates (context precision, recall, MRR, faithfulness) |
| Frontend | API curl | React (Vite) chat with streaming answers + expandable citation cards |

## 🏗️ Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                    Client                     │
                        │      React chat (SSE) · Streamlit analytics   │
                        └───────────────┬──────────────────────────────┘
                                        │ JWT Bearer
                                        ▼
                       ┌──────────────────────────────────────────────┐
                       │        FastAPI (backend/main_v2.py)          │
                       │  auth · workspaces · upload · query · stream │
└───────┬───────────────────────┬──────────────┘
                                │                       │
               ┌────────────────▼───────────────┐  ┌─────▼─────────────────────┐
               │         SQLAlchemy / SQLite    │  │  Semantic cache           │
               │  users · workspaces · docs ·   │  │  Redis (REDIS_URL) with   │
               │  query_logs · api_usage        │  │  in-memory fallback       │
               │  PostgreSQL (+ pgvector) in    │  └───────────────────────────┘
               │  prod · connection pooling     │
               └──────────────────────────────┘
                                │
                 ┌─────────────▼──────────────────────────┐
                 │             Retrieval layer             │
                 │  ChromaDB (dev) OR pgvector (prod) ANN  │
                 │  BM25 keyword search                    │
                 │  EnsembleRetriever (0.6 + 0.4)          │
                 │  Cross-encoder reranker (top-k → 5)     │
                 └─────────────┬──────────────────────────┘
                               │ context + citations
                               ▼
                       ┌──────────────────────────────┐
                       │  LLM generation (Groq)      │
                       │  history-aware reformulation│
                       │  numbered source citations  │
                       └──────────────────────────────┘
```

Ingestion path: `upload → PyPDFLoader → RecursiveCharacterTextSplitter → ONNX
MiniLM embeddings → ChromaDB (dev) or pgvector (prod) collection per workspace
→ cache invalidated`.

Query path: `reformulate → hybrid retrieve → rerank → generate → stream
(sources event first, then tokens)`.

## 🗄️ Storage backends

Everything storage-related is env-switchable, so dev stays zero-infra while
production scales:

| Concern | Dev / default | Production |
|---|---|---|
| Metadata (users, docs, logs) | SQLite | **PostgreSQL** via `DATABASE_URL` (pooled, `pool_pre_ping`) |
| Embeddings | ChromaDB local dir | **pgvector** via `VECTOR_STORE=pgvector` (HNSW ANN on Neon/Postgres) |
| Semantic cache | In-process dict | **Redis** via `REDIS_URL` (Upstash) with automatic fallback |

The vector-store facade (`ai_engine/store.py`) exposes identical functions for
both backends, and `get_hybrid_retriever` rebuilds BM25 from whatever store is
active — swap `VECTOR_STORE=pgvector` and the existing upload/query/stream
endpoints work unchanged. Embeddings stay 384-dim (`all-MiniLM-L6-v2` via
ONNX) in both.

## 📊 Evaluation

`evals/evaluate.py` runs the real retrieval pipeline over a deterministic
corpus and reports citation-grade metrics — and **fails CI** when they regress:

- `context_precision@5` — fraction of top-5 chunks that carry the answer evidence
- `recall@5` — fraction of expected sources found in the top-5
- `MRR@5` — how early the first relevant source ranks (shows reranker value)
- `faithfulness` — LLM-as-judge groundedness of the generated answer
  (enabled when `GROQ_API_KEY` is set)

```bash
python -m evals.evaluate                # human-readable report
python -m evals.evaluate --json --no-generation   # CI-friendly, offline
```

Every push runs this in GitHub Actions (see `.github/workflows/ci.yml`), so
retrieval quality is measured, not assumed.

## 🚀 Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)

### 1. Backend

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your GROQ_API_KEY and a long JWT_SECRET
uvicorn backend.main_v2:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/api/docs>

### 2. Frontend

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

### 3. Analytics dashboard

Infiniflow ships an **admin analytics dashboard** (Streamlit) for query
insights, cache stats, and usage trends:

```bash
streamlit run app.py
```

### 4. Docker

```bash
docker compose up --build
```

Up brings the API, Celery worker, Redis, and the analytics dashboard.

### 5. Render (production blueprint)

A `render.yaml` blueprint deploys four connected services: the FastAPI
backend (pgvector on Neon), a Celery worker, the Streamlit analytics dashboard,
and the React UI (built with `VITE_API_URL` pointing at the API).

1. Create an **Env Group** named `infiniflow` with the secrets:
   `GROQ_API_KEY`, `JWT_SECRET`, `DATABASE_URL` (Neon), `REDIS_URL`.
2. In the Render dashboard: **New → Blueprint** → point at this repo.
   Render will propose the four services from `render.yaml`.

The blueprint assumes default URLs (`<service>-onrender.com`); update
`VITE_API_URL` and `CORS_ORIGINS` in `render.yaml` if you rename services.
Tables bootstrap automatically on boot (`create_all`) for a fresh Neon DB.

## 🔍 Querying — API overview

| Method | Route | Description |
|---|---|---|
| POST | `/api/v1/auth/signup` | Register |
| POST | `/api/v1/auth/login` | Login → JWT |
| GET | `/api/v1/auth/me` | Current user |
| POST | `/api/v1/workspaces` | Create workspace |
| GET | `/api/v1/workspaces` | List workspaces |
| POST | `/api/v1/workspaces/{id}/upload` | Ingest a document |
| POST | `/api/v1/workspaces/{id}/query` | Non-streaming RAG answer |
| POST | `/api/v1/workspaces/{id}/query/stream` | **SSE**: sources event → token stream |
| GET | `/api/v1/workspaces/{id}/analytics` | Usage analytics |

Every query returns structured `citations`:

```json
{
  "answer": "The capital of France is Paris, located on the Seine [1].",
  "citations": [
    {
      "source": "europe_guide.pdf",
      "page": 4,
      "chunk_text": "Paris, the capital of France, is located on the River Seine...",
      "relevance_score": 0.94
    }
  ],
  "strategy_used": "hybrid+rerank",
  "latency_ms": 412
}
```

## 🧠 Reranking

Hybrid retrieval returns ~10 candidates; the cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores them and keeps the top 5
that actually answer the question. Scoring is sigmoid-normalized into (0, 1)
so it displays cleanly next to citations.

- Enable/disable: `ENABLE_RERANKER=true` (`.env`)
- Top-k: `RERANKER_TOP_K=5`
- If the model can't be loaded (no network, first run in CI), the pipeline
  **degrades gracefully** to top-k truncation instead of failing.

## 📈 Roadmap

The platform is architected so these slot in without rework:

1. **Multi-provider LLM router** — Groq primary with OpenAI/Anthropic/Sarvam
   fallbacks (`ai_engine/` already isolates providers).
2. **RAGAS** — plug `ragas` into `evals/evaluate.py` for reference-free
   faithfulness / answer relevancy on live traffic.
3. **GraphRAG mode** — entity-extraction pass at ingestion feeding a knowledge
   graph for multi-hop relationship questions.
4. **RedisVL / RediSearch ANN** — move the in-process semantic matching into
   Redis-native KNN so the cache scales to millions of entries per worker.
5. **Async ingestion** — promote the Celery worker to the default upload path
   (it already supports it) so large PDFs never block the request.

## Environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (LLM generation) |
| `JWT_SECRET` | Secret for signing JWTs — long random string in prod |
| `DATABASE_URL` | SQLAlchemy URL. SQLite by default; set a `postgresql://` URL (e.g. Neon) for pooling + pgvector |
| `VECTOR_STORE` | `chroma` (default) or `pgvector` (requires Postgres `DATABASE_URL`) |
| `REDIS_URL` | Redis connection string (Upstash `rediss://…`) — enables Redis-backed cache |
| `UPSTASH_REDIS_REST_URL` | Alternative Redis endpoint for the cache backend |
| `CHROMA_PATH` | Directory for ChromaDB collections |
| `ENABLE_RERANKER` | Toggle cross-encoder reranking (default `true`) |
| `RERANKER_TOP_K` | Reranker output size (default `5`) |
| `REDIS_HOST` / `REDIS_PORT` | Redis (Celery broker) |
| `INFINIFLOW_SKIP_RERANKER` | Force reranker fallback (used in CI) |

## Repository layout

```
├── backend/             FastAPI API: auth, workspaces, documents, query, stream
├── ai_engine/           reranker, adaptive retrieval, semantic cache, agents
├── evals/               deterministic corpus + evaluation harness
├── frontend/            React + Vite + Tailwind chat UI
├── tests/               pytest suites
├── rag_engine.py        retrieval / generation core
└── app.py               admin analytics dashboard (Streamlit)
```

## License

MIT