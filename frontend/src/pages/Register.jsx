import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../api';
import toast from 'react-hot-toast';

export default function Register() {
    const [form, setForm] = useState({ email: '', password: '', full_name: '' });
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await api.post('/api/v1/auth/signup', form);
            toast.success('Account created! Please sign in.');
            navigate('/login');
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4">
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 left-1/4 w-96 h-96 bg-brand-600/15 rounded-full blur-3xl" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-brand-900/20 rounded-full blur-3xl" />
            </div>

            <div className="relative w-full max-w-sm animate-slide-up">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-brand-600/20 border border-brand-500/30 mb-4">
                        <svg className="w-6 h-6 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                        </svg>
                    </div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">KnowledgeFlow</h1>
                    <p className="text-sm text-gray-500 mt-1">Create your account</p>
                </div>

                <div className="card">
                    <h2 className="text-lg font-semibold text-white mb-6">Get started for free</h2>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {[
                            { label: 'Full Name', key: 'full_name', type: 'text', placeholder: 'Jane Doe' },
                            { label: 'Email', key: 'email', type: 'email', placeholder: 'you@example.com' },
                            { label: 'Password', key: 'password', type: 'password', placeholder: '••••••••' },
                        ].map(({ label, key, type, placeholder }) => (
                            <div key={key}>
                                <label className="block text-xs font-medium text-gray-400 mb-1.5 uppercase tracking-wider">{label}</label>
                                <input
                                    type={type}
                                    className="input-field"
                                    placeholder={placeholder}
                                    value={form[key]}
                                    onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                                    required
                                />
                            </div>
                        ))}
                        <button type="submit" disabled={loading} className="btn-primary w-full justify-center py-3 mt-2">
                            {loading ? 'Creating account...' : 'Create Account'}
                        </button>
                    </form>
                    <p className="text-center text-sm text-gray-500 mt-5">
                        Already have an account?{' '}
                        <Link to="/login" className="text-brand-400 hover:text-brand-300 font-medium transition-colors">Sign in</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
