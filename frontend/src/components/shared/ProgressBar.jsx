export default function ProgressBar({ label, value, isActive = false }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className={isActive ? '' : 'opacity-70'}>
      <div className="flex justify-between items-end mb-2">
        <span className={`text-sm font-medium ${isActive ? 'text-text-main font-bold' : 'text-text-muted'}`}>
          {label}
        </span>
        <span className={`text-sm ${isActive ? 'text-text-main font-bold' : 'text-text-muted'}`}>{pct}%</span>
      </div>
      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isActive ? 'bg-primary shadow-[0_0_10px_rgba(179,227,253,0.4)]' : 'bg-slate-300'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
