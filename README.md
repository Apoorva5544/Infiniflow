# Infiniflow

A RAG (Retrieval-Augmented Generation) platform that lets you build knowledge bases from your documents and query them using a hybrid search engine. It ships with a Streamlit analytical dashboard and a production-grade REST API backed by FastAPI.

## What it does

- Upload PDFs into isolated **workspaces** (knowledge bases)
- Query them using a **hybrid retriever** — ChromaDB vector search combined with BM25 keyword search (60/40 weighted ensemble)
- Supports **history-aware query reformulation** so follow-up questions work naturally
- Semantic caching layer to avoid redundant LLM calls
- JWT authentication on all API routes
- React frontend for the full user-facing experience
- Streamlit interface for analytical / internal use

## Stack

| Layer | Tech |
|---|---|
| LLM / Embeddings | Groq API (Llama 3), HuggingFace embeddings |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| API | FastAPI, SQLAlchemy, SQLite |
| Auth | JWT (python-jose) |
| Frontend | React + Vite + TailwindCSS |
| Analytical UI | Streamlit |

## Project Structure

```
infiniflow/
├── app.py                  # Streamlit analytical dashboard
├── rag_engine.py           # Core RAG logic — chunking, embeddings, retrieval chains
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── backend/                # FastAPI REST API
│   ├── main_v2.py          # API routes (auth, workspaces, documents, querying)
│   ├── models.py           # SQLAlchemy ORM models
│   ├── auth.py             # JWT utilities
│   ├── config.py           # Settings (env-driven)
│   ├── database.py         # DB session + engine setup
│   └── analytics.py        # Query analytics engine
│
├── ai_engine/              # Advanced RAG modules
│   ├── advanced_rag.py     # Query routing + adaptive retrieval + evaluator
│   ├── semantic_cache.py   # In-memory semantic cache
│   └── agents.py           # Agentic query handling
│
├── frontend/               # React + Vite SPA
│   ├── src/
│   │   ├── pages/          # Dashboard, Login, Workspace views
│   │   └── api/            # Axios API layer
│   └── tailwind.config.js
│
└── tests/
    ├── test_advanced_rag.py
    └── test_semantic_cache.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)

### 1. Clone and set up Python environment

```bash
git clone https://github.com/Apoorva5544/Infiniflow.git
cd Infiniflow

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and a JWT_SECRET_KEY
```

### 3. Run the API

```bash
uvicorn backend.main_v2:app --reload --port 8000
```

API docs will be at `http://localhost:8000/api/docs`

### 4. Run the Streamlit dashboard (optional)

```bash
streamlit run app.py
```

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

### Docker (optional)

```bash
docker-compose up --build
```

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |
| `JWT_SECRET_KEY` | Secret for signing JWTs (generate a random string) |
| `CHROMA_PATH` | Path to store ChromaDB data (default: `./chroma_db`) |
| `DATABASE_URL` | SQLite path (default: `sqlite:///./backend/database.db`) |

See `.env.example` for all options.

## API Overview

| Method | Route | Description |
|---|---|---|
| POST | `/api/v1/auth/signup` | Register |
| POST | `/api/v1/auth/login` | Login → JWT |
| GET | `/api/v1/auth/me` | Current user |
| POST | `/api/v1/workspaces` | Create workspace |
| GET | `/api/v1/workspaces` | List workspaces |
| POST | `/api/v1/workspaces/{id}/upload` | Upload PDF |
| POST | `/api/v1/workspaces/{id}/query` | Query workspace |
| GET | `/api/v1/workspaces/{id}/analytics` | Usage analytics |

Full interactive docs at `/api/docs` when the server is running.

## License

MIT
