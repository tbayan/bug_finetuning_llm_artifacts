import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import Card from '../shared/Card';
import ProgressBar from '../shared/ProgressBar';
import { SEVERITY_COLORS } from '../../data/mock';
import SecondaryButton from '../shared/SecondaryButton';

const SEVERITIES = ['Critical', 'Major', 'Minor', 'Trivial'];

function getDescription(severity, confidencePct, probabilities, modelName) {
  const pct = confidencePct ?? Math.round((probabilities[severity] ?? 0) * 100);
  const modelRef = modelName ? ` using ${modelName}` : ' by the baseline model';
  if (pct >= 70) {
    return `Based on the initial triage signatures, this issue has been classified with high confidence (${pct}%)${modelRef}. Immediate review is recommended.`;
  }
  if (pct >= 40) {
    return `This issue has been classified with moderate confidence (${pct}%)${modelRef}. Review when possible.`;
  }
  return `This issue has been classified with low confidence (${pct}%)${modelRef}. Manual review recommended.`;
}

export default function BaselineResult({ entry }) {
  const { result, modelName } = entry;
  const severity = result?.severity || 'Major';
  const probabilities = result?.probabilities || {
    Critical: 0.1,
    Major: 0.65,
    Minor: 0.2,
    Trivial: 0.05,
  };
  const confidencePct = result?.confidence != null ? Math.round(result.confidence * 100) : null;
  const description = result?.description ?? getDescription(severity, confidencePct, probabilities, modelName);
  const severityLabel = severity.charAt(0).toUpperCase() + severity.slice(1).toLowerCase();
  const severityStyles = SEVERITY_COLORS[severity] || SEVERITY_COLORS.Minor;

  return (
    <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
      <nav className="mb-8 flex items-center gap-2 text-sm text-text-muted">
        <Link to="/ingest" className="hover:text-primary">Back to Ingestion</Link>
        <ChevronRight className="h-4 w-4" />
        <span className="text-text-main">Baseline Analysis</span>
      </nav>

      <div className="grid gap-8 lg:grid-cols-2 lg:gap-12">
        <div className="flex flex-col gap-6">
          <Card>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-4">Classification Result</h2>
            <h1 className="text-5xl md:text-6xl font-bold leading-tight tracking-tight pb-8 mb-0">
              <span className="block text-text-main">Severity:</span>
              <span
              className={`block ${severityStyles.text}`}
              style={{
                textShadow: severityStyles.glow
                  ? `0 0 8px ${severityStyles.glow}, 0 0 16px ${severityStyles.glow}, 0 0 24px ${severityStyles.glow}`
                  : undefined,
              }}
            >
              {severityLabel}
            </span>
            </h1>
            <p className="text-base text-text-muted leading-relaxed mb-6">{description}</p>
            <SecondaryButton>View Detailed Report</SecondaryButton>
          </Card>
        </div>

        <Card>
          <h3 className="text-lg font-semibold text-text-main mb-6">Probability Breakdown</h3>
          <div className="flex flex-col gap-6">
            {SEVERITIES.map((label) => (
              <ProgressBar
                key={label}
                label={label}
                value={probabilities[label] ?? 0}
                isActive={label === severity}
              />
            ))}
          </div>
          {modelName && (
            <p className="mt-6 text-xs text-text-muted">Model version: {modelName}</p>
          )}
        </Card>
      </div>
    </main>
  );
}
