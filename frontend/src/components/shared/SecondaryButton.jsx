export default function SecondaryButton({ children, onClick, type = 'button', className = '', icon: Icon }) {
  return (
    <button
      type={type}
      onClick={onClick}
      className={`
        flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white
        px-4 py-2.5 text-sm font-medium text-text-main
        hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 transition-colors
        ${className}
      `}
    >
      {Icon && <Icon className="h-4 w-4" />}
      {children}
    </button>
  );
}
