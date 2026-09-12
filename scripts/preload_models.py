"""Pre-download ML models at Docker build time.

First-use model downloads (from HuggingFace / chromadb) can stall a free-tier
cold start for tens of seconds and blow Render's request timeout. Downloading
them into the image at build time turns those stalls into instant local loads.

Everything here is best-effort: a missing model logs a SKIP and the app still
boots — the model is simply downloaded lazily on first use later.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _try(label: str, fn) -> None:
    try:
        fn()
        print(f"[preload] OK  {label}")
    except Exception as e:  # pragma: no cover - build-time resilience
        print(f"[preload] SKIP {label}: {e}")


def _preload_chroma_onnx() -> None:
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    # Instantiating downloads the MiniLM ONNX model used by the vector store.
    ONNXMiniLM_L6_V2()


def _preload_hf_minilm() -> None:
    from langchain_huggingface import HuggingFaceEmbeddings

    # all-MiniLM-L6-v2 used by the semantic cache.
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _preload_reranker() -> None:
    from sentence_transformers import CrossEncoder

    # Only used when ENABLE_RERANKER=true; harmless to have on disk.
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def main() -> None:
    _try("chromadb ONNX MiniLM (embeddings)", _preload_chroma_onnx)
    _try("sentence-transformers all-MiniLM-L6-v2 (semantic cache)", _preload_hf_minilm)
    _try("cross-encoder ms-marco-MiniLM-L-6-v2 (reranker)", _preload_reranker)


if __name__ == "__main__":
    main()