import os
import re
import time
import zlib
from threading import Lock

from ai_engine import store as _store
from ai_engine.reranker import CrossEncoderReranker
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfReadError, PdfStreamError

# In-process hybrid-retriever cache. Building BM25 requires pulling every
# document out of the store and re-indexing it — doing that per request is a
# big waste on free-tier CPU. Keyed by vector-store object identity: the store
# cache in ai_engine.store hands out a NEW object after every upload/delete, so
# a stale key can never survive a write and a reused object hits the cache.
_retriever_cache: dict = {}
_retriever_lock = Lock()
_MAX_RETRIEVERS = 6


class _OnnxMiniLMEmbeddings:
    """Lightweight ONNX MiniLM embeddings (~50MB RAM) instead of PyTorch (~400MB)."""

    def __init__(self):
        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self._ef(list(texts))]

    def embed_query(self, text):
        return list(map(float, self._ef([text])[0]))


def _embeddings():
    return _OnnxMiniLMEmbeddings()


class UnreadablePdfError(Exception):
    """Raised when a PDF cannot be parsed (corrupt, truncated, or not a PDF)."""


def _assert_valid_pdf(file_path):
    """Raise UnreadablePdfError when the file is not a PDF at all."""
    with open(file_path, "rb") as f:
        header = f.read(1024)
    if not header.startswith(b"%PDF-"):
        name = os.path.basename(file_path)
        raise UnreadablePdfError(
            f"'{name}' is not a valid PDF file (missing %PDF header)."
        )


def _zlib_decompress_tolerant(data):
    """Tolerant FlateDecode decompression for possibly-truncated streams.

    pypdf refuses anything it cannot fully decode ("Stream has ended
    unexpectedly"); here we salvage whatever plaintext the deflate stream
    contains before it breaks. Tries zlib-wrapped, then raw deflate.
    """
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(data)
            try:
                out += d.flush()
            except Exception:
                pass
            if out:
                return out
        except Exception:
            try:
                d = zlib.decompressobj(wbits)
                return d.decompress(data)
            except Exception:
                continue
    return b""


_TEXT_UNESCAPES = {
    "n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
    "(": "(", ")": ")", "\\": "\\",
}


def _unescape_pdf_string(s):
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(s):
            break
        nxt = s[i]
        if nxt in _TEXT_UNESCAPES:
            out.append(_TEXT_UNESCAPES[nxt])
            i += 1
        elif nxt.isdigit():
            oct_digits = nxt
            i += 1
            while i < len(s) and len(oct_digits) < 3 and s[i].isdigit():
                oct_digits += s[i]
                i += 1
            out.append(chr(int(oct_digits, 8) & 0xFF))
        else:
            out.append(nxt)
            i += 1
    return "".join(out)


def _text_from_content(content):
    """Pull (…) Tj/TJ strings out of a decompressed content stream."""
    s = content.decode("latin-1", errors="ignore")
    parts = []
    for m in re.finditer(r"\((?:[^()\\]|\\.)*\)", s):
        parts.append(_unescape_pdf_string(m.group(0)[1:-1]))
    return " ".join(p for p in parts if p.strip())


def _extract_text_from_pdf_bytes(raw):
    """Last-resort recovery for truncated/corrupt PDFs.

    Decompresses each FlateDecode stream (best-effort) and extracts the text
    show operators (Tj / TJ), so a PDF pypdf rejects can still be ingested.
    """
    chunks = []
    for m in re.finditer(rb"stream\r?\n", raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        data = raw[start:] if end < 0 else raw[start:end]
        dec = _zlib_decompress_tolerant(data)
        if not dec:
            continue
        txt = _text_from_content(dec)
        if txt:
            chunks.append(txt)
    return "\n".join(chunks)


def _recover_documents(file_path, cause):
    """Salvage text from a PDF pypdf refused, or raise UnreadablePdfError."""
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        raw = f.read()

    text = _extract_text_from_pdf_bytes(raw)
    if text.strip():
        print(
            f"Recovered {len(text)} chars from unreadable PDF '{file_name}' "
            f"(pypdf: {cause})"
        )
        from langchain_core.documents import Document

        return [Document(page_content=text, metadata={"source": file_name})]

    raise UnreadablePdfError(
        f"'{file_name}' appears corrupted or truncated and no text could be "
        f"recovered ({cause}). Please re-export or re-download the file and retry."
    )


def process_document(file_path):
    """Loads a PDF and splits it into smaller chunks with metadata."""
    _assert_valid_pdf(file_path)

    file_name = os.path.basename(file_path)
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except (PdfReadError, PdfStreamError) as e:
        documents = _recover_documents(file_path, e)
    except Exception as e:
        if "Stream has ended unexpectedly" in str(e):
            documents = _recover_documents(file_path, e)
        else:
            raise

    # Add source filename to metadata for citations
    for doc in documents:
        doc.metadata["source"] = file_name

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv(override=True)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


def list_collections():
    """Returns a list of available knowledge base collections."""
    return _store.list_collections()


def create_vector_store(chunks, collection_name="default"):
    """Creates/Updates a named Chroma/pgvector vector database (appends chunks)."""
    store = _store.create_vector_store(chunks, collection_name)
    _invalidate_retrievers()
    return store


def get_vector_store(collection_name="default"):
    """Retrieves a specific vector store by name, or None when empty."""
    return _store.get_vector_store(collection_name)


def delete_collection(collection_name):
    """Deletes a named collection and drops its cached retriever."""
    try:
        return _store.delete_collection(collection_name)
    finally:
        _invalidate_retrievers()


def _invalidate_retrievers():
    """Drop cached hybrid retrievers (keyed by store object identity)."""
    with _retriever_lock:
        _retriever_cache.clear()


def _stored_documents(vector_store):
    """Recovers the documents persisted in a vector store for BM25 rebuild.

    Works for Chroma-style stores (.get()), PGVector stores (in-process
    docstore, or straight from Postgres across restarts) and any store class
    that keeps a langchain docstore (PGVector.docstore._dict) instead.
    """
    from langchain_core.documents import Document

    if hasattr(vector_store, "get"):
        data = vector_store.get()
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        if docs:
            return [
                Document(page_content=d, metadata=m or {}) for d, m in zip(docs, metas)
            ]
        return []

    ds = getattr(vector_store, "docstore", None)
    if ds is not None:
        backing = getattr(ds, "_dict", None)
        if isinstance(backing, dict):
            return [d for d in backing.values() if isinstance(d, Document)]

    if getattr(vector_store, "__class__", None).__name__ == "PGVector":
        collection_name = getattr(vector_store, "collection_name", None)
        if collection_name:
            try:
                return _store.stored_documents(collection_name)
            except Exception as e:
                print(f"Warning: Could not recover pgvector documents for BM25: {e}")
    return []


def get_hybrid_retriever(vector_store, chunks=None):
    """Creates a Hybrid Retriever combining Vector and BM25 search.

    The expensive part (pulling all documents out of the store + building the
    BM25 index) is memoized per vector-store object. The store cache hands out
    a new object whenever documents are added, so uploads always see fresh data.
    """
    key = id(vector_store)
    with _retriever_lock:
        cached = _retriever_cache.get(key)
    if cached is not None:
        return cached

    retriever = _build_hybrid_retriever(vector_store, chunks)
    with _retriever_lock:
        _retriever_cache[key] = retriever
        if len(_retriever_cache) > _MAX_RETRIEVERS:
            _retriever_cache.pop(next(iter(_retriever_cache)))
    return retriever


def _build_hybrid_retriever(vector_store, chunks=None):
    # 1. Vector Retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. BM25 Retriever (Keyword Search)
    if chunks is None:
        try:
            chunks = _stored_documents(vector_store)
        except Exception as e:
            print(f"Warning: Error fetching documents for BM25: {e}")
            return vector_retriever

    if not chunks:
        return vector_retriever

    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5

        # 3. Ensemble (Hybrid)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],  # More weight to semantic search
        )
        return ensemble_retriever
    except Exception as e:
        print(f"Warning: Could not initialize BM25Retriever: {e}")
        return vector_retriever


def get_qa_chain(vector_store, retriever=None):
    """Creates a conversational RAG chain with Enterprise-grade features."""
    api_key = os.getenv("GROQ_API_KEY", "").strip("\"' ")
    llm = ChatGroq(
        temperature=0, model_name="qwen/qwen3.8-27b", groq_api_key=api_key
    )

    if retriever is None:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 10}
        )  # Fetch more for reranking

    # Optional cross-encoder reranking: compress the hybrid candidates down to the
    # top_k that actually answer the question. Degree is graceful — the compressor
    # falls back to top-k truncation if the cross-encoder cannot be loaded.
    if str(os.getenv("ENABLE_RERANKER", "true")).lower() in {"1", "true", "yes"}:
        try:
            top_k = int(os.getenv("RERANKER_TOP_K", "5"))
            retriever = ContextualCompressionRetriever(
                base_compressor=CrossEncoderReranker(top_k=top_k),
                base_retriever=retriever,
            )
        except Exception as e:
            print(f"Warning: Reranker init failed ({e}); using raw hybrid retriever.")

    # Contextualize question logic
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question logic
    system_prompt = (
        "You are an Elite Research AI. Answer the question using ONLY the provided context. "
        "If the answer isn't in the context, state that clearly. "
        "Structure your response with clear headings or bullet points if appropriate. "
        "Cite the source of every factual claim inline using [filename.pdf] or [filename.pdf, page N].\n\n"
        "CONTEXT:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def rerank_documents(query, documents, top_k=5):
    """
    Reranks documents using a Cross-Encoder for precision.
    Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` when available; otherwise
    degrades to top-k truncation so the pipeline never breaks.
    """
    return CrossEncoderReranker(top_k=top_k).compress_documents(
        list(documents), str(query)
    )
