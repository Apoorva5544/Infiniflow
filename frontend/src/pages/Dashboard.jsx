import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';
import toast from 'react-hot-toast';

const icons = {
    folder: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" />
        </svg>
    ),
    plus: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
    ),
    logout: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
        </svg>
    ),
    docs: (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" />
        </svg>
    ),
    query: (
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
        </svg>
    ),
    trash: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
    ),
};

function WorkspaceCard({ ws, onClick, onDelete }) {
    return (
        <div
            onClick={onClick}
            className="cursor-pointer group animate-fade-in rounded-xl bg-white border border-slate-200 hover:border-slate-300 hover:shadow-md transition-all p-5 relative"
        >
            <button
                onClick={e => { e.stopPropagation(); onDelete(ws); }}
                className="absolute top-4 right-4 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 hover:bg-red-50 transition-all duration-150"
                title="Delete workspace"
            >
                {icons.trash}
            </button>

            <div className="flex items-start gap-4 mb-4">
                <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200/70 text-emerald-600 shrink-0">
                    {icons.folder}
                </div>
                <div className="min-w-0">
                    <h3 className="font-headline-md text-slate-900 font-bold truncate pr-6">{ws.name}</h3>
                    {ws.description && (
                        <p className="text-slate-500 text-xs mt-0.5 line-clamp-1">{ws.description}</p>
                    )}
                </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-slate-500 font-label-mono-xs">
                <span className="flex items-center gap-1.5">
                    {icons.docs}
                    {ws.total_documents} docs
                </span>
                <span className="flex items-center gap-1.5">
                    {icons.query}
                    {ws.total_queries} queries
                </span>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                <span className="text-xs text-slate-500">
                    {new Date(ws.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
                <span className="badge bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-slow" />
                    Active
                </span>
            </div>
        </div>
    );
}

export default function Dashboard({ user, onLogout }) {
    const [workspaces, setWorkspaces] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [form, setForm] = useState({ name: '', description: '', llm_model: 'qwen/qwen3.8-27b' });
    const [creating, setCreating] = useState(false);
    const [deleteTarget, setDeleteTarget] = useState(null);
    const navigate = useNavigate();

    const fetchWorkspaces = async () => {
        setLoading(true);
        try {
            const resp = await api.get('/api/v1/workspaces');
            setWorkspaces(resp.data);
        } catch {
            toast.error('Failed to load workspaces');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchWorkspaces(); }, []);

    const createWorkspace = async (e) => {
        e.preventDefault();
        if (!form.name.trim()) return;
        setCreating(true);
        try {
            await api.post('/api/v1/workspaces', form);
            toast.success(`Knowledge Layer "${form.name}" created`);
            setForm({ name: '', description: '', llm_model: 'qwen/qwen3.8-27b' });
            setShowModal(false);
            fetchWorkspaces();
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Creation failed');
        } finally {
            setCreating(false);
        }
    };

    const confirmDelete = async () => {
        if (!deleteTarget) return;
        try {
            await api.delete(`/api/v1/workspaces/${deleteTarget.id}`);
            toast.success(`"${deleteTarget.name}" deleted`);
            setDeleteTarget(null);
            fetchWorkspaces();
        } catch {
            toast.error('Delete failed');
        }
    };

    const totalDocs = workspaces.reduce((s, w) => s + w.total_documents, 0);
    const totalQueries = workspaces.reduce((s, w) => s + w.total_queries, 0);

    return (
        <div className="min-h-screen bg-white">
            {/* ── Header ── */}
            <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-xl border-b border-slate-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
                <div className="h-20 max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <a href="/landing.html" className="flex items-center gap-3">
                            <div className="h-9 w-9 rounded-lg overflow-hidden flex items-center justify-center bg-slate-900 border border-slate-800 shadow-sm shrink-0">
                                <svg className="w-5 h-5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                                </svg>
                            </div>
                            <span className="font-headline-md text-xl font-extrabold text-slate-900 tracking-tight">InfiniFlow</span>
                        </a>
                        <span className="hidden xl:inline-flex items-center px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200/60 font-label-mono-xs text-xs font-semibold text-emerald-700 uppercase tracking-wider">
                            PRODUCTION RAG PLATFORM
                        </span>
                    </div>

                    <nav className="hidden lg:flex items-center gap-1">
                        <a className="px-3.5 py-2 text-slate-900 hover:bg-slate-100/80 transition-colors font-body-sm text-sm font-medium rounded-lg bg-emerald-50 text-emerald-700" href="#/workspaces">
                            Knowledge Layers
                        </a>
                    </nav>

                    <div className="flex items-center gap-2 sm:gap-3">
                        <a className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200/80 hover:bg-emerald-100/70 transition-all font-label-mono-xs text-xs font-medium whitespace-nowrap" href="/landing.html">
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                            Live: infiniflow-prod.onrender.com
                        </a>
                        <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-500 text-xs font-bold">
                                {user.full_name?.[0]?.toUpperCase() || 'U'}
                            </div>
                            <div className="hidden sm:block">
                                <p className="text-sm font-medium text-slate-900 leading-none">{user.full_name}</p>
                                <p className="text-xs text-slate-500">{user.email}</p>
                            </div>
                        </div>
                        <button onClick={onLogout} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-slate-400 transition-colors font-body-sm text-sm font-medium">
                            {icons.logout}
                            <span className="hidden sm:inline">Sign out</span>
                        </button>
                    </div>
                </div>
            </header>

            {/* ── Main ── */}
            <main className="max-w-7xl mx-auto px-6 lg:px-12 py-10">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <div className="font-label-mono-xs text-xs text-emerald-600 font-bold uppercase tracking-wider mb-1">
                            WORKSPACE TELEMETRY: ACTIVE RUNTIME
                        </div>
                        <h1 className="font-headline-xl text-3xl sm:text-4xl text-slate-900 font-bold tracking-tight">Knowledge Layers</h1>
                        <p className="font-body-md text-base text-slate-500 mt-1">Isolated RAG workspaces with hybrid retrieval</p>
                    </div>
                    <button onClick={() => setShowModal(true)} className="btn-primary">
                        {icons.plus} New Layer
                    </button>
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3.5 mb-8">
                    {[
                        { label: 'Knowledge Layers', value: workspaces.length, accent: 'text-emerald-600' },
                        { label: 'Documents Ingested', value: totalDocs, accent: 'text-teal-600' },
                        { label: 'Total Queries', value: totalQueries, accent: 'text-cyan-600' },
                    ].map(stat => (
                        <div key={stat.label} className="p-5 rounded-xl bg-slate-50/80 border border-slate-200/80">
                            <p className="font-label-mono-xs text-xs text-slate-500 uppercase tracking-wider font-medium mb-1.5">{stat.label}</p>
                            <p className={`font-headline-lg text-3xl font-extrabold tracking-tight ${stat.accent}`}>{stat.value}</p>
                        </div>
                    ))}
                </div>

                {/* Workspace grid */}
                {loading ? (
                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                        {[1, 2, 3].map(i => (
                            <div key={i} className="rounded-xl bg-white border border-slate-200 p-5 animate-pulse">
                                <div className="h-5 bg-slate-200 rounded mb-4 w-3/4" />
                                <div className="h-3 bg-slate-200 rounded mb-2 w-1/2" />
                                <div className="h-3 bg-slate-200 rounded w-1/3" />
                            </div>
                        ))}
                    </div>
                ) : workspaces.length === 0 ? (
                    <div className="rounded-2xl bg-white border border-slate-200 flex flex-col items-center justify-center py-16 text-center shadow-sm">
                        <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 mb-4 text-emerald-600">
                            {icons.folder}
                        </div>
                        <h3 className="font-headline-lg text-slate-900 font-bold mb-2">No Knowledge Layers yet</h3>
                        <p className="text-slate-500 text-sm mb-5">Create your first isolated RAG workspace to get started</p>
                        <button onClick={() => setShowModal(true)} className="btn-primary">{icons.plus} Create First Layer</button>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {workspaces.map(ws => (
                            <WorkspaceCard
                                key={ws.id}
                                ws={ws}
                                onClick={() => navigate(`/workspace/${ws.id}`)}
                                onDelete={setDeleteTarget}
                            />
                        ))}
                    </div>
                )}
            </main>

            {/* ── Footer ── */}
            <footer className="w-full bg-white border-t border-slate-200 mt-10">
                <div className="max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-500 font-body-sm text-xs">
                    <p>© 2025 InfiniFlow Inc. Enterprise RAG Platform &amp; Multi-Tenant Knowledge Bases. All rights reserved.</p>
                    <div className="flex items-center gap-4 font-label-mono-xs text-xs">
                        <span className="inline-flex items-center gap-1.5 text-emerald-700 font-medium">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
                            All Systems Operational
                        </span>
                    </div>
                </div>
            </footer>

            {/* ── Create Modal ── */}
            {showModal && (
                <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="rounded-2xl bg-white border border-slate-200 shadow-xl animate-slide-up p-6 w-full max-w-md relative overflow-hidden">
                        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500" />
                        <h2 className="font-headline-lg text-lg font-bold text-slate-900 mb-5">New Knowledge Layer</h2>
                        <form onSubmit={createWorkspace} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase tracking-wider">Layer Name *</label>
                                <input
                                    className="input-field"
                                    placeholder="e.g. Company Policies"
                                    value={form.name}
                                    onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                                    required
                                    autoFocus
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase tracking-wider">Description</label>
                                <input
                                    className="input-field"
                                    placeholder="Optional description..."
                                    value={form.description}
                                    onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase tracking-wider">LLM Model</label>
                                <select className="input-field" value={form.llm_model} onChange={e => setForm(f => ({ ...f, llm_model: e.target.value }))}>
                                    <option value="qwen/qwen3.8-27b">Qwen 3.8 27B (Fast)</option>
                                    <option value="qwen/qwen3.6-27b">Qwen 3.6 27B (Fallback)</option>
                                    <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                                </select>
                            </div>
                            <div className="flex gap-3 pt-2">
                                <button type="submit" disabled={creating} className="btn-primary flex-1 justify-center">
                                    {creating ? 'Creating...' : 'Create Layer'}
                                </button>
                                <button type="button" onClick={() => setShowModal(false)} className="btn-secondary">Cancel</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {/* ── Delete Confirm Modal ── */}
            {deleteTarget && (
                <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="rounded-2xl bg-white border border-slate-200 shadow-xl animate-slide-up p-6 w-full max-w-sm text-center">
                        <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center mx-auto mb-4 text-red-500">
                            {icons.trash}
                        </div>
                        <h3 className="font-headline-lg text-slate-900 font-bold mb-2">Delete Knowledge Layer?</h3>
                        <p className="text-slate-500 text-sm mb-6">
                            "<span className="text-slate-900 font-semibold">{deleteTarget.name}</span>" and all its documents will be permanently removed.
                        </p>
                        <div className="flex gap-3">
                            <button onClick={confirmDelete} className="btn-danger flex-1 justify-center">Delete</button>
                            <button onClick={() => setDeleteTarget(null)} className="btn-secondary flex-1 justify-center">Cancel</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}