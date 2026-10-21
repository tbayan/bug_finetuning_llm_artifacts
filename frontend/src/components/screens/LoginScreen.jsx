import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import PrimaryButton from '../shared/PrimaryButton';
import SecondaryButton from '../shared/SecondaryButton';
import AppLogo from '../shared/AppLogo';
import NeuronBackground from './NeuronBackground';

function GoogleLogo() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M23.49 12.27c0-.79-.07-1.55-.2-2.27H12v4.31h6.44a5.5 5.5 0 0 1-2.39 3.61v3h3.87c2.26-2.08 3.57-5.15 3.57-8.65z"
      />
      <path
        fill="#34A853"
        d="M12 24c3.24 0 5.95-1.07 7.94-2.91l-3.87-3c-1.07.72-2.43 1.15-4.07 1.15-3.13 0-5.79-2.11-6.74-4.95H1.26v3.08A12 12 0 0 0 12 24z"
      />
      <path
        fill="#FBBC05"
        d="M5.26 14.29A7.2 7.2 0 0 1 4.89 12c0-.79.14-1.55.37-2.29V6.63H1.26A12 12 0 0 0 0 12c0 1.93.46 3.75 1.26 5.37l4-3.08z"
      />
      <path
        fill="#EA4335"
        d="M12 4.77c1.76 0 3.34.61 4.58 1.82l3.43-3.43C17.94 1.21 15.24 0 12 0A12 12 0 0 0 1.26 6.63l4 3.08c.95-2.84 3.61-4.94 6.74-4.94z"
      />
    </svg>
  );
}

function AppleLogo() {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
      <path
        fill="currentColor"
        d="M16.37 12.78c.02 2.09 1.84 2.79 1.86 2.8-.02.05-.29 1.02-1 2.03-.62.87-1.26 1.74-2.28 1.76-1 .02-1.33-.58-2.48-.58s-1.5.56-2.45.6c-.98.04-1.73-.98-2.36-1.85-1.28-1.78-2.26-5.02-.95-7.3.65-1.13 1.8-1.84 3.05-1.86.95-.02 1.85.64 2.48.64.62 0 1.8-.8 3.03-.68.52.02 1.99.21 2.93 1.59-.08.05-1.75 1.03-1.73 3.05zM14.72 4.92c.52-.64.87-1.52.77-2.4-.75.03-1.66.5-2.2 1.14-.48.56-.9 1.46-.78 2.31.84.07 1.68-.43 2.21-1.05z"
      />
    </svg>
  );
}

export default function LoginScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    login(username || 'analyst@cybertriage.com');
    navigate('/ingest', { replace: true });
  };

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      {/* Left: neuron animation + tagline */}
      <div className="relative flex flex-col justify-center gap-8 overflow-hidden bg-slate-50 p-12 lg:p-16">
        <NeuronBackground />
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'linear-gradient(135deg, rgba(248,250,252,0.35) 0%, rgba(248,250,252,0.2) 60%, transparent 100%)',
          }}
        />
        <div className="relative flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-sky-200 bg-primary/40 shadow-sm">
            <AppLogo className="h-10 w-10" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-text-main">
              Intelligent Bug Classification
            </h1>
            <p className="mt-1 text-sm text-text-muted">
              Master's Thesis Project — CyberTriage
            </p>
          </div>
        </div>
        <p className="relative max-w-md text-text-muted">
          Classify and triage software defects with baseline ML and LLM-powered analysis. 
          Secure, fast, and built for modern incident response.
        </p>
      </div>

      {/* Right: form */}
      <div className="flex flex-col justify-center gap-8 p-8 sm:p-12 lg:p-16">
        <div>
          <h2 className="text-2xl font-bold text-text-main">Welcome back</h2>
          <p className="mt-2 text-text-muted">Sign in to continue to CyberTriage.</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-text-main mb-2">
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="analyst@cybertriage.com"
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-text-main placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              autoComplete="username"
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-text-main mb-2">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-text-main placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              autoComplete="current-password"
            />
          </div>
          <PrimaryButton type="submit">Sign In</PrimaryButton>
        </form>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t border-slate-200" />
          </div>
          <div className="relative flex justify-center text-xs font-medium text-text-muted">
            Or continue with
          </div>
        </div>

        <div className="flex gap-4">
          <SecondaryButton className="flex-1" onClick={() => { login('google@mock'); navigate('/ingest', { replace: true }); }}>
            <span className="mr-2 inline-flex items-center">
              <GoogleLogo />
            </span>
            Google
          </SecondaryButton>
          <SecondaryButton className="flex-1" onClick={() => { login('apple@mock'); navigate('/ingest', { replace: true }); }}>
            <span className="mr-2 inline-flex items-center">
              <AppleLogo />
            </span>
            Apple
          </SecondaryButton>
        </div>

        <p className="text-center text-sm text-text-muted">
          Need access? <button type="button" className="font-medium text-primary hover:underline">Contact Admin</button>
        </p>
      </div>
    </div>
  );
}
