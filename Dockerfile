# Multi-stage build: React frontend + FastAPI backend in one deployable image.

# ── Stage 1: Build the React frontend ─────────────────────────────────────────
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend (serves the built UI as StaticFiles) ──────────────
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: build-essential for source-built wheels, curl for Docker health.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — psycopg2-binary (Postgres/Neon driver) is pinned in requirements.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ ./backend/
COPY ai_engine/ ./ai_engine/
COPY rag_engine.py ./
COPY app.py ./

# Built frontend static assets
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

# Render sets PORT; docker compose uses the default 8000.
CMD ["sh", "-c", "uvicorn backend.main_v2:app --host 0.0.0.0 --port ${PORT:-8000}"]