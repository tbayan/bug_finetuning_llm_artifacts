import { NavLink } from 'react-router-dom';
import { Bell, User } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import AppLogo from '../shared/AppLogo';

const NAV = [
  { to: '/ingest', label: 'New Ingestion' },
  { to: '/evaluation', label: 'Batch Evaluation' },
  { to: '/history', label: 'History' },
];

export default function TopBar() {
  const { user } = useAuth();

  return (
    <header className="sticky top-0 z-50 w-full bg-white/90 border-b border-slate-200 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-sky-200 bg-primary/40">
            <AppLogo className="h-6 w-6" />
          </div>
          <span className="text-lg font-bold tracking-tight text-text-main">CyberTriage</span>
        </div>

        <nav className="flex flex-1 justify-center gap-10">
          {NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `relative py-4 text-sm font-medium transition-colors ${
                  isActive
                    ? 'text-text-main'
                    : 'text-text-muted hover:text-text-main'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {label}
                  {isActive && (
                    <span className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full bg-primary" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <button
            type="button"
            className="flex h-9 w-9 items-center justify-center rounded-full text-text-muted hover:bg-slate-100 hover:text-text-main transition-colors"
            aria-label="Notifications"
          >
            <Bell className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2 rounded-full pl-1 pr-3 py-1 hover:bg-slate-100 transition-colors">
            <div className="h-8 w-8 rounded-full bg-slate-200 flex items-center justify-center">
              <User className="h-4 w-4 text-text-muted" />
            </div>
            <span className="text-sm font-medium text-text-main hidden sm:block">
              {user?.name || user?.id || 'Guest'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
