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
                <div className="max-w-[70%] px-4 py-3 rounded-2xl rounded-tr-sm bg-emerald-600 text-white text-sm shadow-sm">
                    {msg.text}
                </div>
            </div>
        );
    }

    if (msg.type === 'error') {
        return (
            <div className="flex mb-4 animate-fade-in">
                <div className="max-w-[80%] px-4 py-3 rounded-2xl rounded-tl-sm bg-red-50 border border-red-200 text-red-600 text-sm">
                    ⚠️ {msg.text}
                </div>
            </div>
        );
    }

    if (msg.type === 'system') {
        return (
            <div className="flex justify-center mb-4 animate-fade-in">
                <span className="badge bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs">✓ {msg.text}</span>
            </div>
        );
    }

    // AI answer
    return (
        <div className="flex mb-6 animate-fade-in">
            <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-500 text-xs font-bold shrink-0 mr-3 mt-0.5">
                AI
            </div>
            <div className="max-w-[80%] space-y-3">
                <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-white border border-slate-200 shadow-sm text-sm ai-answer">
                    <div dangerouslySetInnerHTML={{ __html: formatAnswer(msg.text) }} />
                    {!msg.text && <span className="text-slate-400 animate-pulse">Thinking…</span>}
                </div>

                {/* Metadata row */}
                {(msg.sources?.length > 0 || msg.latency_ms || msg.strategy_used) && (
                    <div className="flex flex-wrap gap-2">
                        {msg.strategy_used && (
                            <span className="badge bg-teal-50 text-teal-700 border border-teal-200">
                                🧠 {msg.strategy_used}
                            </span>
                        )}
                        {msg.latency_ms && (
                            <span className="badge bg-blue-50 text-blue-700 border border-blue-200">
                                ⚡ {msg.latency_ms}ms
                            </span>
                        )}
                        {msg.relevance_score !== undefined && (
                            <span className="badge bg-amber-50 text-amber-700 border border-amber-200">
                                📊 {(msg.relevance_score * 100).toFixed(0)}% relevance
                            </span>
                        )}
                        {msg.cached && (
                            <span className="badge bg-emerald-50 text-emerald-700 border border-emerald-200">⚡ cached</span>
                        )}
                    </div>
                )}

                {/* Citations */}
                {msg.citations?.length > 0 && (
                    <div>
                        <p className="text-xs text-slate-500 mb-1.5">Citations</p>
                        <div className="space-y-1.5">
                            {msg.citations.map((c, i) => (
                                <details key={i} className="rounded-xl bg-white border border-slate-200 px-3 py-2 group open:shadow-sm">
                                    <summary className="cursor-pointer list-none flex items-center justify-between gap-2 text-xs">
                                        <span className="flex items-center gap-1.5 min-w-0">
                                            <span>📄</span>
                                            <span className="text-slate-700 truncate">{c.source}</span>
                                            {c.page != null && <span className="text-slate-400 shrink-0">· p.{c.page}</span>}
                                        </span>
                                        <span className="flex items-center gap-2 shrink-0">
                                            {typeof c.relevance_score === 'number' && (
                                                <span className="text-emerald-600 font-semibold">{(c.relevance_score * 100).toFixed(0)}%</span>
                                            )}
                                            <span className="text-slate-400">▾</span>
                                        </span>
                                    </summary>
                                    <p className="text-[11px] text-slate-500 mt-1.5 leading-relaxed">{c.chunk_text}</p>
                                </details>
                            ))}
                        </div>
                    </div>
                )}

                {/* Sources (legacy list, shown only when structured citations are absent) */}
                {!msg.citations?.length && msg.sources?.length > 0 && (
                    <div>
                        <p className="text-xs text-slate-500 mb-1.5">Sources</p>
                        <div className="flex flex-wrap gap-1.5">
                            {msg.sources.map((s, i) => (
                                <span key={i} className="badge bg-slate-50 text-slate-600 border border-slate-200">
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
        .replace(/\[(\d+)\]/g, '<sup class="text-emerald-600 font-semibold">[$1]</sup>')
        .replace(/^#{1,3} (.+)$/gm, '<h3 class="font-semibold text-slate-900 mt-2 mb-1">$1</h3>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/`([^`]+)`/g, '<code class="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded text-emerald-700">$1</code>')
        .replace(/\n\n/g, '</p><p class="mb-3">')
        .replace(/\n/g, '<br/>');
}

// ── SSE streaming helper ──────────────────────────────────────────────────────
async function streamQuery(workspaceId, payload, handlers) {
    const base = api.defaults.baseURL || '';
    const token = localStorage.getItem('token');
    const resp = await fetch(`${base}/api/v1/workspaces/${workspaceId}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`Stream request failed (${resp.status})`);
    if (!resp.body) throw new Error('Streaming not supported by this browser');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
            const frame = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            const dataLine = frame.split('\n').find(l => l.startsWith('data: '));
            if (!dataLine) continue;
            let data;
            try {
                data = JSON.parse(dataLine.slice(6));
            } catch {
                continue;
            }
            if (data.type === 'sources') handlers.onSources?.(data);
            else if (data.type === 'token') handlers.onToken?.(data.content);
            else if (data.type === 'done') handlers.onDone?.(data);
            else if (data.type === 'error') handlers.onError?.(data.detail);
        }
    }
}

// ── Document list item ─────────────────────────────────────────────────────────
function DocItem({ doc }) {
    const statusColor = {
        processed: 'text-emerald-700 bg-emerald-50 border-emerald-200',
        processing: 'text-amber-700 bg-amber-50 border-amber-200',
        failed: 'text-red-700 bg-red-50 border-red-200',
    }[doc.status] || 'text-slate-600 bg-slate-50 border-slate-200';

    return (
        <div className="flex items-start gap-2 p-2.5 rounded-lg hover:bg-slate-50 transition-colors group">
            <span className="text-slate-400 mt-0.5 shrink-0"><File /></span>
            <div className="min-w-0 flex-1">
                <p className="text-xs text-slate-800 truncate font-medium">{doc.filename}</p>
                <div className="flex items-center gap-2 mt-1">
                    <span className={`badge text-[10px] border ${statusColor}`}>{doc.status}</span>
                    {doc.chunk_count > 0 && (
                        <span className="text-[10px] text-slate-400">{doc.chunk_count} chunks</span>
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
    const answerRef = useRef('');

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
        setQuerying(true);
        answerRef.current = '';

        const aiMsg = { id: Date.now() + 1, type: 'ai', text: '', sources: [], citations: [], strategy_used: 'hybrid+rerank', latency_ms: null, relevance_score: undefined, cached: false };
        setMessages(prev => [...prev, { id: Date.now(), type: 'user', text: q }, aiMsg]);

        const patch = (updates) => setMessages(prev =>
            prev.map(m => m.id === aiMsg.id ? { ...m, ...updates } : m)
        );

        const payload = { question: q, chat_history: chatHistory, strategy: 'auto', use_cache: true };

        // 1) Try streaming (SSE)
        try {
            await streamQuery(id, payload, {
                onSources: (s) => patch({
                    citations: s.citations || [],
                    strategy_used: s.strategy_used || aiMsg.strategy_used,
                    cached: s.cached,
                }),
                onToken: (tok) => {
                    answerRef.current += tok;
                    patch({ text: answerRef.current });
                },
                onDone: (d) => patch({ latency_ms: d.latency_ms, cached: d.cached }),
                onError: (detail) => {
                    patch({ type: 'error', text: detail || 'Streaming failed.' });
                    answerRef.current = '';
                },
            });
        } catch {
            // 2) Fallback: non-streaming endpoint (also used with suppressed sources)
            try {
                const resp = await api.post(`/api/v1/workspaces/${id}/query`, payload);
                const d = resp.data;
                answerRef.current = d.answer;
                patch({
                    text: d.answer,
                    sources: d.sources || [],
                    citations: d.citations || [],
                    strategy_used: d.strategy_used || 'simple',
                    latency_ms: d.latency_ms,
                    relevance_score: d.relevance_score,
                    cached: d.cached,
                });
            } catch (err) {
                const detail = err.response?.data?.detail;
                const status = err.response?.status;
                let msg;
                if (detail) {
                    msg = detail;
                } else if (status) {
                    msg = `Query failed (HTTP ${status}). The service may be restarting — try again in a moment.`;
                } else {
                    msg = 'Cannot reach the server. The service is likely restarting (e.g. after an OOM event) — try again in a moment.';
                }
                patch({ type: 'error', text: msg });
            }
        }

        // Rolling chat history (last 10 turns)
        setChatHistory(prev => [
            ...prev.slice(-10),
            { role: 'human', content: q },
            { role: 'ai', content: answerRef.current },
        ]);
        setQuerying(false);
    };

    return (
        <div className="flex h-screen bg-white overflow-hidden">
            {/* ── Left sidebar: doc list ── */}
            <aside className="w-64 shrink-0 border-r border-slate-200/80 flex flex-col bg-white">
                <div className="p-4 border-b border-slate-200/80">
                    <button
                        onClick={() => navigate('/workspaces')}
                        className="flex items-center gap-2 text-slate-500 hover:text-slate-900 transition-colors text-sm mb-4"
                    >
                        <ArrowLeft /> Back
                    </button>
                    {workspace && (
                        <div>
                            <h2 className="font-headline-md text-slate-900 font-bold text-sm truncate">{workspace.name}</h2>
                            <p className="font-label-mono-xs text-slate-500 text-xs mt-0.5">{workspace.llm_model}</p>
                        </div>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-3">
                    <div className="flex items-center justify-between mb-2">
                        <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">Documents</p>
                        <span className="text-xs text-slate-400">{workspace?.documents?.length || 0}</span>
                    </div>

                    {workspace?.documents?.length === 0 ? (
                        <p className="text-xs text-slate-400 text-center py-6">No documents yet</p>
                    ) : (
                        <div className="space-y-0.5">
                            {workspace?.documents?.map(doc => <DocItem key={doc.id} doc={doc} />)}
                        </div>
                    )}
                </div>

                <div className="p-4 border-t border-slate-200/80">
                    <label className={`btn-primary w-full justify-center cursor-pointer ${uploading ? 'opacity-60 pointer-events-none' : ''}`}>
                        {uploading ? <><Spinner /> Ingesting...</> : <><Upload /> Ingest Document</>}
                        <input ref={fileInputRef} type="file" hidden accept=".pdf,.docx,.txt,.md" onChange={handleUpload} disabled={uploading} />
                    </label>
                </div>
            </aside>

            {/* ── Chat area ── */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Chat header */}
                <div className="px-6 py-4 border-b border-slate-200/80 flex items-center gap-3">
                    <div className="flex-1">
                        <h1 className="font-headline-md text-slate-900 font-bold text-sm">
                            {workspace?.name || 'Loading...'}
                        </h1>
                        <p className="text-slate-500 text-xs">
                            Hybrid Retrieval · Vector + BM25 + Cross-encoder rerank · Streaming with citations
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <span className="badge bg-emerald-50 text-emerald-700 border border-emerald-200">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-slow" />
                            {workspace?.llm_model || 'Groq'}
                        </span>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto px-6 py-6 bg-slate-50/50">
                    {messages.length === 0 && (
                        <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
                            <div className="w-16 h-16 rounded-2xl bg-emerald-50 border border-emerald-200/70 flex items-center justify-center text-emerald-600 mb-4">
                                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                                </svg>
                            </div>
                            <h3 className="font-headline-lg text-slate-900 font-bold mb-2">Ready to query</h3>
                            <p className="text-slate-500 text-sm max-w-xs">
                                Upload documents to this Knowledge Layer, then ask anything. The hybrid retriever will find the most relevant context.
                            </p>
                        </div>
                    )}

                    {messages.map((msg, i) => <Message key={i} msg={msg} />)}

                    {querying && (
                        <div className="flex mb-4 animate-fade-in">
                            <div className="w-7 h-7 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-500 text-xs font-bold shrink-0 mr-3">
                                AI
                            </div>
                            <div className="px-4 py-3 rounded-2xl bg-white border border-slate-200 flex items-center gap-2 text-slate-500 text-sm shadow-sm">
                                <Spinner />
                                Retrieving & synthesizing...
                            </div>
                        </div>
                    )}

                    <div ref={bottomRef} />
                </div>

                {/* Input */}
                <div className="px-6 pb-6 py-4 border-t border-slate-200/80 bg-white">
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
                            className="absolute right-3 top-1/2 -translate-y-1/2 w-9 h-9 rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center text-white transition-all"
                        >
                            {querying ? <Spinner size={4} /> : <Send />}
                        </button>
                    </form>
                    <p className="text-center text-xs text-slate-400 mt-2">
                        EnsembleRetriever · BM25 + ChromaDB · Cross-encoder rerank · SSE streaming · Source citations
                    </p>
                </div>
            </div>
        </div>
    );
}