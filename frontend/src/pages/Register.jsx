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
        <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-emerald-100/50 rounded-full blur-[120px]" />
                <div className="absolute bottom-0 left-0 w-[400px] h-[300px] bg-teal-100/40 rounded-full blur-[100px]" />
            </div>

            <div className="relative w-full max-w-sm animate-slide-up">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 shadow-sm mb-4">
                        <svg className="w-6 h-6 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                        </svg>
                    </div>
                    <h1 className="font-headline-xl text-2xl font-extrabold text-slate-900 tracking-tight">InfiniFlow</h1>
                    <p className="font-body-sm text-sm text-slate-500 mt-1">Create your account</p>
                </div>

                <div className="rounded-2xl bg-white border border-slate-200 shadow-xl shadow-slate-200/50 p-6 relative overflow-hidden">
                    <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500" />
                    <h2 className="font-headline-lg text-lg font-bold text-slate-900 mb-6">Get started for free</h2>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {[
                            { label: 'Full Name', key: 'full_name', type: 'text', placeholder: 'Jane Doe' },
                            { label: 'Email', key: 'email', type: 'email', placeholder: 'you@example.com' },
                            { label: 'Password', key: 'password', type: 'password', placeholder: '••••••••' },
                        ].map(({ label, key, type, placeholder }) => (
                            <div key={key}>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5 uppercase tracking-wider">{label}</label>
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
                        <button type="submit" disabled={loading} className="w-full inline-flex items-center justify-center px-4 py-3 rounded-lg bg-emerald-600 text-white font-headline-md text-sm font-semibold shadow-sm shadow-emerald-600/20 hover:bg-emerald-700 transition-all mt-2 disabled:opacity-50 disabled:cursor-not-allowed">
                            {loading ? 'Creating account...' : 'Create Account'}
                        </button>
                    </form>
                    <p className="text-center text-sm text-slate-500 mt-5">
                        Already have an account?{' '}
                        <Link to="/login" className="text-emerald-600 hover:text-emerald-700 font-semibold transition-colors">Sign in</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}