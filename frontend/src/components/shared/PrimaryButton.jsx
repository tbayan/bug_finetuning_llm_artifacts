export default function PrimaryButton({ children, onClick, disabled = false, type = 'button', className = '', icon: Icon }) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3
        text-text-main font-bold
        hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2
        disabled:opacity-50 disabled:cursor-not-allowed transition-colors
        ${className}
      `}
    >
      {Icon && <Icon className="h-5 w-5" />}
      {children}
    </button>
  );
}
