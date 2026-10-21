import { SEVERITY_COLORS } from '../../data/mock';

export default function SeverityBadge({ severity }) {
  const styles = SEVERITY_COLORS[severity] || SEVERITY_COLORS.Minor;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${styles.bg} ${styles.text} ${styles.border}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {severity}
    </span>
  );
}
