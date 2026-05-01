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
            className="card glass-hover cursor-pointer group relative animate-fade-in"
        >
            <button
                onClick={e => { e.stopPropagation(); onDelete(ws); }}
                className="absolute top-4 right-4 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 hover:bg-red-400/10 transition-all duration-150"
                title="Delete workspace"
            >
                {icons.trash}
            </button>

            <div className="flex items-start gap-4 mb-4">
                <div className="p-2.5 rounded-xl bg-brand-500/10 border border-brand-500/20 text-brand-400 shrink-0">
                    {icons.folder}
                </div>
                <div className="min-w-0">
                    <h3 className="text-white font-semibold truncate pr-6">{ws.name}</h3>
                    {ws.description && (
                        <p className="text-gray-500 text-xs mt-0.5 line-clamp-1">{ws.description}</p>
                    )}
                </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1.5">
                    {icons.docs}
                    {ws.total_documents} docs
                </span>
                <span className="flex items-center gap-1.5">
                    {icons.query}
                    {ws.total_queries} queries
                </span>
            </div>

            <div className="mt-4 pt-4 border-t border-white/[0.05] flex items-center justify-between">
                <span className="text-xs text-gray-600">
                    {new Date(ws.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </span>
                <span className="badge bg-brand-500/10 text-brand-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse-slow" />
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
    const [form, setForm] = useState({ name: '', description: '', llm_model: 'llama-3.1-8b-instant' });
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
            setForm({ name: '', description: '', llm_model: 'llama-3.1-8b-instant' });
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
        <div className="flex min-h-screen bg-gray-950">
            {/* ── Sidebar ── */}
            <aside className="w-64 shrink-0 border-r border-white/[0.06] flex flex-col p-5 bg-gray-950">

                <div className="flex items-center gap-3 mb-8">
                    <div className="w-8 h-8 rounded-xl bg-brand-600/20 border border-brand-500/30 flex items-center justify-center">
                        <svg className="w-4 h-4 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                        </svg>
                    </div>
                    <div>
                        <p className="text-white font-semibold text-sm leading-none">KnowledgeFlow</p>
                        <p className="text-gray-600 text-xs">RAG Platform</p>
                    </div>
                </div>

                <nav className="flex-1 space-y-1">
                    <div className="sidebar-item active">
                        {icons.folder}
                        Knowledge Layers
                    </div>
                </nav>

                <div className="border-t border-white/[0.06] pt-4 mt-4">
                    <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-full bg-brand-600/20 flex items-center justify-center text-brand-400 text-xs font-bold">
                            {user.full_name?.[0]?.toUpperCase() || 'U'}
                        </div>
                        <div className="min-w-0">
                            <p className="text-white text-xs font-medium truncate">{user.full_name}</p>
                            <p className="text-gray-500 text-xs truncate">{user.email}</p>
                        </div>
                    </div>
                    <button onClick={onLogout} className="sidebar-item w-full text-red-400 hover:text-red-300 hover:bg-red-400/10">
                        {icons.logout} Sign out
                    </button>
                </div>
            </aside>

            {/* ── Main ── */}
            <main className="flex-1 overflow-y-auto">
                <div className="max-w-6xl mx-auto px-8 py-8">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-2xl font-bold text-white">Knowledge Layers</h1>
                            <p className="text-gray-500 text-sm mt-0.5">Isolated RAG workspaces with hybrid retrieval</p>
                        </div>
                        <button onClick={() => setShowModal(true)} className="btn-primary">
                            {icons.plus} New Layer
                        </button>
                    </div>

                    {/* Stats row */}
                    <div className="grid grid-cols-3 gap-4 mb-8">
                        {[
                            { label: 'Knowledge Layers', value: workspaces.length },
                            { label: 'Documents Ingested', value: totalDocs },
                            { label: 'Total Queries', value: totalQueries },
                        ].map(stat => (
                            <div key={stat.label} className="card">
                                <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">{stat.label}</p>
                                <p className="text-3xl font-bold text-white">{stat.value}</p>
                            </div>
                        ))}
                    </div>

                    {/* Workspace grid */}
                    {loading ? (
                        <div className="grid grid-cols-3 gap-4">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="card animate-pulse">
                                    <div className="h-5 bg-white/[0.05] rounded mb-4 w-3/4" />
                                    <div className="h-3 bg-white/[0.05] rounded mb-2 w-1/2" />
                                    <div className="h-3 bg-white/[0.05] rounded w-1/3" />
                                </div>
                            ))}
                        </div>
                    ) : workspaces.length === 0 ? (
                        <div className="card flex flex-col items-center justify-center py-16 text-center">
                            <div className="p-4 rounded-2xl bg-brand-500/10 border border-brand-500/20 mb-4 text-brand-400">
                                {icons.folder}
                            </div>
                            <h3 className="text-white font-semibold mb-2">No Knowledge Layers yet</h3>
                            <p className="text-gray-500 text-sm mb-5">Create your first isolated RAG workspace to get started</p>
                            <button onClick={() => setShowModal(true)} className="btn-primary">{icons.plus} Create First Layer</button>
                        </div>
                    ) : (
                        <div className="grid grid-cols-3 gap-4">
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
                </div>
            </main>

            {/* ── Create Modal ── */}
            {showModal && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="card w-full max-w-md animate-slide-up">
                        <h2 className="text-lg font-semibold text-white mb-5">New Knowledge Layer</h2>
                        <form onSubmit={createWorkspace} className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">Layer Name *</label>
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
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">Description</label>
                                <input
                                    className="input-field"
                                    placeholder="Optional description..."
                                    value={form.description}
                                    onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">LLM Model</label>
                                <select className="input-field" value={form.llm_model} onChange={e => setForm(f => ({ ...f, llm_model: e.target.value }))}>
                                    <option value="llama-3.1-8b-instant">Llama 3.1 8B (Fast)</option>
                                    <option value="llama-3.1-70b-versatile">Llama 3.1 70B (Powerful)</option>
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
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="card w-full max-w-sm animate-slide-up text-center">
                        <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center mx-auto mb-4 text-red-400">
                            {icons.trash}
                        </div>
                        <h3 className="text-white font-semibold mb-2">Delete Knowledge Layer?</h3>
                        <p className="text-gray-400 text-sm mb-6">
                            "<span className="text-white">{deleteTarget.name}</span>" and all its documents will be permanently removed.
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
