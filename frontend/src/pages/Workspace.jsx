import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api';
import toast from 'react-hot-toast';

// ── Icons ────────────────────────────────────────────────────────────────────
const ArrowLeft = () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
    </svg>
);
const Send = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
    </svg>
);
const Upload = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
    </svg>
);
const File = () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
);
const Spinner = ({ size = 4 }) => (
    <svg className={`animate-spin w-${size} h-${size}`} fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
);

// ── Message bubble ────────────────────────────────────────────────────────────
function Message({ msg }) {
    const isUser = msg.type === 'user';

    if (isUser) {
        return (
            <div className="flex justify-end mb-4 animate-fade-in">
                <div className="max-w-[70%] px-4 py-3 rounded-2xl rounded-tr-sm bg-brand-600 text-white text-sm">
                    {msg.text}
                </div>
            </div>
        );
    }

    if (msg.type === 'error') {
        return (
            <div className="flex mb-4 animate-fade-in">
                <div className="max-w-[80%] px-4 py-3 rounded-2xl rounded-tl-sm bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                    ⚠️ {msg.text}
                </div>
            </div>
        );
    }

    if (msg.type === 'system') {
        return (
            <div className="flex justify-center mb-4 animate-fade-in">
                <span className="badge bg-brand-500/10 text-brand-400 text-xs">✓ {msg.text}</span>
            </div>
        );
    }

    // AI answer
    return (
        <div className="flex mb-6 animate-fade-in">
            <div className="w-7 h-7 rounded-full bg-brand-600/20 border border-brand-500/30 flex items-center justify-center text-brand-400 text-xs font-bold shrink-0 mr-3 mt-0.5">
                AI
            </div>
            <div className="max-w-[80%] space-y-3">
                <div className="px-4 py-3 rounded-2xl rounded-tl-sm glass text-sm ai-answer">
                    <div dangerouslySetInnerHTML={{ __html: formatAnswer(msg.text) }} />
                </div>

                {/* Metadata row */}
                {(msg.sources?.length > 0 || msg.latency_ms || msg.strategy_used) && (
                    <div className="flex flex-wrap gap-2">
                        {msg.strategy_used && (
                            <span className="badge bg-purple-500/10 text-purple-400">
                                🧠 {msg.strategy_used}
                            </span>
                        )}
                        {msg.latency_ms && (
                            <span className="badge bg-blue-500/10 text-blue-400">
                                ⚡ {msg.latency_ms}ms
                            </span>
                        )}
                        {msg.relevance_score !== undefined && (
                            <span className="badge bg-amber-500/10 text-amber-400">
                                📊 {(msg.relevance_score * 100).toFixed(0)}% relevance
                            </span>
                        )}
                        {msg.cached && (
                            <span className="badge bg-teal-500/10 text-teal-400">⚡ cached</span>
                        )}
                    </div>
                )}

                {/* Sources */}
                {msg.sources?.length > 0 && (
                    <div>
                        <p className="text-xs text-gray-600 mb-1.5">Sources</p>
                        <div className="flex flex-wrap gap-1.5">
                            {msg.sources.map((s, i) => (
                                <span key={i} className="badge bg-white/[0.04] text-gray-400 border border-white/[0.08]">
                                    📄 {s}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

function formatAnswer(text) {
    if (!text) return '';
    // Simple markdown-like rendering
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/^#{1,3} (.+)$/gm, '<h3 class="font-semibold text-white mt-2 mb-1">$1</h3>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/`([^`]+)`/g, '<code class="font-mono text-xs bg-white/[0.08] px-1.5 py-0.5 rounded text-brand-300">$1</code>')
        .replace(/\n\n/g, '</p><p class="mb-3">')
        .replace(/\n/g, '<br/>');
}

// ── Document list item ─────────────────────────────────────────────────────────
function DocItem({ doc }) {
    const statusColor = {
        processed: 'text-brand-400 bg-brand-500/10',
        processing: 'text-amber-400 bg-amber-500/10',
        failed: 'text-red-400 bg-red-500/10',
    }[doc.status] || 'text-gray-400 bg-gray-500/10';

    return (
        <div className="flex items-start gap-2 p-2.5 rounded-xl hover:bg-white/[0.03] transition-colors group">
            <span className="text-gray-500 mt-0.5 shrink-0"><File /></span>
            <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-300 truncate font-medium">{doc.filename}</p>
                <div className="flex items-center gap-2 mt-1">
                    <span className={`badge text-[10px] ${statusColor}`}>{doc.status}</span>
                    {doc.chunk_count > 0 && (
                        <span className="text-[10px] text-gray-600">{doc.chunk_count} chunks</span>
                    )}
                </div>
            </div>
        </div>
    );
}

// ── Main Workspace ─────────────────────────────────────────────────────────────
export default function Workspace() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [workspace, setWorkspace] = useState(null);
    const [messages, setMessages] = useState([]);
    const [question, setQuestion] = useState('');
    const [querying, setQuerying] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);
    const bottomRef = useRef(null);
    const fileInputRef = useRef(null);

    const fetchWorkspace = useCallback(async () => {
        try {
            const resp = await api.get(`/api/v1/workspaces/${id}`);
            setWorkspace(resp.data);
        } catch {
            toast.error('Failed to load workspace');
        }
    }, [id]);

    useEffect(() => { fetchWorkspace(); }, [fetchWorkspace]);
    useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.name.match(/\.(pdf|docx|txt|md)$/i)) {
            toast.error('Supported: PDF, DOCX, TXT, MD');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        setUploading(true);

        const toastId = toast.loading(`Ingesting ${file.name}...`);
        try {
            const resp = await api.post(`/api/v1/workspaces/${id}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            toast.success(
                `Ingested! ${resp.data.chunks} chunks in ${resp.data.processing_time_seconds}s`,
                { id: toastId }
            );
            setMessages(prev => [...prev, {
                type: 'system',
                text: `${file.name} ingested — ${resp.data.chunks} chunks created`,
            }]);
            fetchWorkspace(); // refresh doc list
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Ingestion failed', { id: toastId });
        } finally {
            setUploading(false);
            fileInputRef.current.value = '';
        }
    };

    const handleQuery = async (e) => {
        e.preventDefault();
        const q = question.trim();
        if (!q || querying) return;

        setQuestion('');
        setMessages(prev => [...prev, { type: 'user', text: q }]);
        setQuerying(true);

        try {
            const resp = await api.post(`/api/v1/workspaces/${id}/query`, {
                question: q,
                chat_history: chatHistory,
                strategy: 'auto',
                use_cache: true,
            });

            const aiMsg = {
                type: 'ai',
                text: resp.data.answer,
                sources: resp.data.sources || [],
                strategy_used: resp.data.strategy_used,
                latency_ms: resp.data.latency_ms,
                relevance_score: resp.data.relevance_score,
                cached: resp.data.cached,
            };
            setMessages(prev => [...prev, aiMsg]);

            // Keep rolling chat history (last 10 turns)
            setChatHistory(prev => [
                ...prev.slice(-10),
                { role: 'human', content: q },
                { role: 'ai', content: resp.data.answer },
            ]);
        } catch (err) {
            setMessages(prev => [...prev, {
                type: 'error',
                text: err.response?.data?.detail || 'Query failed. Make sure documents are ingested first.',
            }]);
        } finally {
            setQuerying(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-950 overflow-hidden">

            {/* ── Left sidebar: doc list ── */}
            <aside className="w-64 shrink-0 border-r border-white/[0.06] flex flex-col bg-gray-950">
                <div className="p-4 border-b border-white/[0.06]">
                    <button
                        onClick={() => navigate('/')}
                        className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors text-sm mb-4"
                    >
                        <ArrowLeft /> Back
                    </button>
                    {workspace && (
                        <div>
                            <h2 className="text-white font-semibold text-sm truncate">{workspace.name}</h2>
                            <p className="text-gray-500 text-xs mt-0.5">{workspace.llm_model}</p>
                        </div>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-3">
                    <div className="flex items-center justify-between mb-2">
                        <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">Documents</p>
                        <span className="text-xs text-gray-600">{workspace?.documents?.length || 0}</span>
                    </div>

                    {workspace?.documents?.length === 0 ? (
                        <p className="text-xs text-gray-600 text-center py-6">No documents yet</p>
                    ) : (
                        <div className="space-y-0.5">
                            {workspace?.documents?.map(doc => <DocItem key={doc.id} doc={doc} />)}
                        </div>
                    )}
                </div>

                <div className="p-4 border-t border-white/[0.06]">
                    <label className={`btn-primary w-full justify-center cursor-pointer ${uploading ? 'opacity-60 pointer-events-none' : ''}`}>
                        {uploading ? <><Spinner /> Ingesting...</> : <><Upload /> Ingest Document</>}
                        <input ref={fileInputRef} type="file" hidden accept=".pdf,.docx,.txt,.md" onChange={handleUpload} disabled={uploading} />
                    </label>
                </div>
            </aside>

            {/* ── Chat area ── */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Chat header */}
                <div className="px-6 py-4 border-b border-white/[0.06] flex items-center gap-3">
                    <div className="flex-1">
                        <h1 className="text-white font-semibold text-sm">
                            {workspace?.name || 'Loading...'}
                        </h1>
                        <p className="text-gray-500 text-xs">
                            Hybrid Retriever · ChromaDB (60%) + BM25 (40%) · History-aware reformulation
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <span className="badge bg-brand-500/10 text-brand-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse-slow" />
                            Groq · Llama 3.1
                        </span>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-6 py-6">
                    {messages.length === 0 && (
                        <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
                            <div className="w-16 h-16 rounded-2xl bg-brand-600/10 border border-brand-500/20 flex items-center justify-center text-brand-400 mb-4">
                                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                                </svg>
                            </div>
                            <h3 className="text-white font-semibold mb-2">Ready to query</h3>
                            <p className="text-gray-500 text-sm max-w-xs">
                                Upload documents to this Knowledge Layer, then ask anything. The hybrid retriever will find the most relevant context.
                            </p>
                        </div>
                    )}

                    {messages.map((msg, i) => <Message key={i} msg={msg} />)}

                    {querying && (
                        <div className="flex mb-4 animate-fade-in">
                            <div className="w-7 h-7 rounded-full bg-brand-600/20 border border-brand-500/30 flex items-center justify-center text-brand-400 text-xs font-bold shrink-0 mr-3">
                                AI
                            </div>
                            <div className="px-4 py-3 rounded-2xl glass flex items-center gap-2 text-gray-400 text-sm">
                                <Spinner />
                                Retrieving & synthesizing...
                            </div>
                        </div>
                    )}

                    <div ref={bottomRef} />
                </div>

                {/* Input */}
                <div className="px-6 pb-6">
                    <form onSubmit={handleQuery} className="relative">
                        <input
                            value={question}
                            onChange={e => setQuestion(e.target.value)}
                            placeholder="Ask anything about your documents..."
                            disabled={querying}
                            className="input-field pr-12 py-4 rounded-2xl"
                            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleQuery(e); } }}
                        />
                        <button
                            type="submit"
                            disabled={querying || !question.trim()}
                            className="absolute right-3 top-1/2 -translate-y-1/2 w-9 h-9 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center text-white transition-all"
                        >
                            {querying ? <Spinner size={4} /> : <Send />}
                        </button>
                    </form>
                    <p className="text-center text-xs text-gray-700 mt-2">
                        EnsembleRetriever · BM25 + ChromaDB Semantic · History-aware query reformulation
                    </p>
                </div>
            </div>
        </div>
    );
}
