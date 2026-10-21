import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import Card from '../shared/Card';
import SeverityBadge from '../shared/SeverityBadge';
import PrimaryButton from '../shared/PrimaryButton';
import SecondaryButton from '../shared/SecondaryButton';
import Dropdown from '../shared/Dropdown';

const PRIORITY_OPTIONS = [
  { value: 'Critical', label: 'Critical' },
  { value: 'Major', label: 'Major' },
  { value: 'Minor', label: 'Minor' },
  { value: 'Trivial', label: 'Trivial' },
];

function formatReason(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
}

export default function LLMResult({ entry }) {
  const { result, modelName } = entry;
  const status = result?.status || 'completed';
  const reasoning = formatReason(result?.reason ?? result?.reasoning) || 'LLM analysis reasoning not available.';
  const [summary, setSummary] = useState(result?.summary ?? '');
  const [priority, setPriority] = useState(result?.priority ?? result?.severity ?? 'Major');
  const [description, setDescription] = useState(result?.description ?? '');

  return (
    <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
      <nav className="mb-8 flex items-center gap-2 text-sm text-text-muted">
        <Link to="/ingest" className="hover:text-primary">Back to Ingestion</Link>
        <ChevronRight className="h-4 w-4" />
        <span className="text-text-main">Bug #{entry.id?.slice(-6) || 'Result'}</span>
      </nav>

      <div className="mb-8 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-main">Analysis Results</h1>
          <div className="mt-2 flex items-center gap-3">
            <SeverityBadge severity={entry.severity || priority} />
            <span className="text-sm text-text-muted capitalize">{status}</span>
          </div>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-1">
        <Card>
          <h3 className="text-lg font-semibold text-text-main mb-4">AI Reasoning</h3>
          <p className="text-text-muted whitespace-pre-wrap">{reasoning}</p>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-text-main mb-6">Export to Jira</h3>
          <div className="flex flex-col gap-6 max-w-2xl">
            <div>
              <label className="block text-sm font-medium text-text-main mb-2">Summary</label>
              <input
                type="text"
                value={summary}
                onChange={(e) => setSummary(e.target.value)}
                placeholder="Brief summary for Jira"
                className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-text-main placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <Dropdown
              label="Priority"
              value={priority}
              onChange={setPriority}
              options={PRIORITY_OPTIONS}
            />
            <div>
              <label className="block text-sm font-medium text-text-main mb-2">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Full description for the Jira ticket"
                rows={4}
                className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-text-main placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary resize-y"
              />
            </div>
            <div className="flex gap-4">
              <SecondaryButton>Save Draft</SecondaryButton>
              <PrimaryButton>Create Jira Ticket</PrimaryButton>
            </div>
          </div>
        </Card>
      </div>

      {modelName && (
        <p className="mt-6 text-xs text-text-muted">Model: {modelName}</p>
      )}
    </main>
  );
}
