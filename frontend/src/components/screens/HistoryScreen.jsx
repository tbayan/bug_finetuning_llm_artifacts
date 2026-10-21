import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Download, Trash2 } from 'lucide-react';
import Card from '../shared/Card';
import SeverityBadge from '../shared/SeverityBadge';
import { useHistory } from '../../context/HistoryContext';

const PAGE_SIZE = 10;

const LLM_MODEL_IDS = new Set(['ollama', 'qwen', 'finetuned', 'qwen_finetuned']);
const BASELINE_MODEL_IDS = new Set(['lr', 'svm', 'rf', 'nb', 'xgb', 'ensemble']);

function isLLMEntry(entry) {
  if (LLM_MODEL_IDS.has(entry.modelId)) return true;
  if (BASELINE_MODEL_IDS.has(entry.modelId)) return false;
  return entry.modelType === 'llm';
}

function formatReason(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
}

function entrySearchText(entry) {
  const result = entry?.result || {};
  return [
    entry?.textSnippet,
    entry?.text,
    entry?.modelName,
    entry?.modelId,
    entry?.severity,
    entry?.timestamp,
    result?.analysis_summary,
    result?.analysis_description,
    result?.summary,
    result?.description,
    formatReason(result?.reasoning ?? result?.reason),
  ]
    .filter(Boolean)
    .map((x) => String(x).toLowerCase())
    .join(' ');
}

const ACTION_BTN_BASE =
  'inline-flex items-center justify-center gap-1 rounded-lg border bg-white px-2 py-1 text-xs font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 transition-colors min-w-[58px]';
const ACTION_BTN_VIEW =
  `${ACTION_BTN_BASE} border-slate-200 text-text-main hover:bg-slate-50 focus:ring-slate-200`;
const ACTION_BTN_DELETE =
  `${ACTION_BTN_BASE} border-red-200 text-red-600 hover:bg-red-50 focus:ring-red-200`;

// ── Jira CSV Export ─────────────────────────────────────────────────────────

const escapeCSV = (str) => '"' + String(str || '').replace(/"/g, '""') + '"';

function mapSeverityToJiraPriority(severity) {
  switch ((severity || '').toLowerCase()) {
    case 'critical':
      return 'Highest';
    case 'major':
      return 'High';
    case 'minor':
      return 'Low';
    case 'trivial':
      return 'Lowest';
    default:
      // Fallback: default Major -> High if unknown
      return 'High';
  }
}

function exportToJiraCSV(llmHistory) {
  const headers = ['Issue Type', 'Summary', 'Description', 'Priority'];

  const rows = llmHistory.map((item) => {
    const summary =
      item.result?.analysis_summary || item.result?.summary || 'Bug Report';

    const desc =
      item.result?.analysis_description || item.result?.description || '';
    const reasoning = formatReason(item.result?.reasoning ?? item.result?.reason);

    let description;
    if (desc || reasoning) {
      description = desc + (reasoning ? '\n\nAI Reasoning:\n' + reasoning : '');
    } else {
      description = item.text || item.textSnippet || '';
    }

    const priority = mapSeverityToJiraPriority(item.severity || 'Major');

    return [
      escapeCSV('Bug'),
      escapeCSV(summary),
      escapeCSV(description),
      escapeCSV(priority),
    ].join(',');
  });

  const csv = [headers.map(escapeCSV).join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'jira_bug_import.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Component ───────────────────────────────────────────────────────────────

export default function HistoryScreen() {
  const { entries, removeEntry } = useHistory();
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [activeTab, setActiveTab] = useState('llm');

  const llmEntries = useMemo(() => entries.filter(isLLMEntry), [entries]);
  const baselineEntries = useMemo(
    () => entries.filter((e) => !isLLMEntry(e)),
    [entries],
  );

  const currentEntries = activeTab === 'llm' ? llmEntries : baselineEntries;

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return currentEntries;
    return currentEntries.filter((e) => entrySearchText(e).includes(q));
  }, [currentEntries, search]);

  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const start = (page - 1) * PAGE_SIZE;
  const pageEntries = filtered.slice(start, start + PAGE_SIZE);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setPage(1);
    setSearch('');
  };

  const handleDeleteEntry = (entryId) => {
    if (!entryId) return;
    const ok = window.confirm('Delete this history record? This action cannot be undone.');
    if (!ok) return;
    removeEntry(entryId);
  };

  const tabs = [
    { id: 'llm', label: 'LLM Triage', count: llmEntries.length },
    { id: 'baseline', label: 'Baseline Triage', count: baselineEntries.length },
  ];

  return (
    <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
      <nav className="mb-6 text-sm text-text-muted">
        <span className="text-text-main">Analysis Request History</span>
      </nav>

      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text-main">
          Analysis Request History
        </h1>
        <p className="mt-2 text-text-muted">
          View and open past classification results.
        </p>
      </div>

      {/* ── Tabs ──────────────────────────────────────────────────────── */}
      <div className="mb-6 flex border-b border-slate-200">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => handleTabChange(tab.id)}
            className={`relative px-6 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-primary'
                : 'text-text-muted hover:text-text-main'
            }`}
          >
            {tab.label}
            <span className="ml-2 inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-text-muted">
              {tab.count}
            </span>
            {activeTab === tab.id && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-t" />
            )}
          </button>
        ))}
      </div>

      {/* ── LLM Export bar ────────────────────────────────────────────── */}
      {activeTab === 'llm' && (
        <div className="mb-6 flex flex-col gap-2">
          <div className="w-fit">
            <button
              type="button"
              onClick={() => exportToJiraCSV(filtered)}
              disabled={filtered.length === 0}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-bold text-text-main hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="h-4 w-4" />
              Export to CSV (Jira)
            </button>
          </div>
          <p className="text-sm text-gray-500 italic">
            Tip: You can import this CSV directly into Jira. Go to your Jira
            Board &gt; Issues &gt; Import issues from CSV.
          </p>
        </div>
      )}

      {/* ── Search ────────────────────────────────────────────────────── */}
      <Card className="mb-6">
        <div className="flex flex-wrap items-center gap-4">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted" />
            <input
              type="search"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              placeholder={
                activeTab === 'llm'
                  ? 'Search by summary, model, severity…'
                  : 'Search by snippet, model, severity…'
              }
              className="w-full rounded-xl border border-slate-200 bg-white py-2.5 pl-10 pr-4 text-sm text-text-main placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
      </Card>

      {/* ── Tab Content ───────────────────────────────────────────────── */}
      {activeTab === 'baseline' ? (
        <BaselineTable
          entries={pageEntries}
          navigate={navigate}
          onDelete={handleDeleteEntry}
        />
      ) : (
        <LLMCards
          entries={pageEntries}
          navigate={navigate}
          onDelete={handleDeleteEntry}
        />
      )}

      {/* ── Pagination ────────────────────────────────────────────────── */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-6 px-2">
          <p className="text-sm text-text-muted">
            Showing {start + 1} to {Math.min(start + PAGE_SIZE, total)} of{' '}
            {total}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-text-main hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-text-main hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </main>
  );
}

// ── Baseline table (severity + original text) ───────────────────────────────

function BaselineTable({ entries, navigate, onDelete }) {
  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-hidden">
        <table className="w-full text-left text-sm table-fixed">
          <colgroup>
            <col style={{ width: '14%' }} />
            <col style={{ width: '14%' }} />
            <col style={{ width: '12%' }} />
            <col style={{ width: '46%' }} />
            <col style={{ width: '14%' }} />
          </colgroup>
          <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wider text-text-muted border-b border-slate-200">
            <tr>
              <th className="px-3 py-3" scope="col">Date &amp; Time</th>
              <th className="px-3 py-3" scope="col">Model</th>
              <th className="px-3 py-3" scope="col">Severity</th>
              <th className="px-3 py-3" scope="col">Text</th>
              <th className="px-3 py-3 text-right" scope="col">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {entries.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-text-muted">
                  No baseline results yet. Run an analysis from New Ingestion.
                </td>
              </tr>
            ) : (
              entries.map((entry) => (
                <tr
                  key={entry.id}
                  onClick={() => navigate(`/result/${entry.id}`)}
                  className="hover:bg-slate-50 cursor-pointer transition-colors"
                >
                  <td className="px-3 py-3 font-medium text-text-main overflow-hidden text-ellipsis text-xs">
                    {entry.timestamp}
                  </td>
                  <td className="px-3 py-3 text-text-muted overflow-hidden text-ellipsis">
                    {entry.modelName || entry.modelId}
                  </td>
                  <td className="px-3 py-3">
                    <SeverityBadge severity={entry.severity || 'Minor'} />
                  </td>
                  <td className="px-3 py-3 text-text-main overflow-hidden text-ellipsis max-w-0">
                    {entry.textSnippet || '—'}
                  </td>
                  <td className="px-3 py-3" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center justify-end gap-1.5">
                      <button
                        type="button"
                        className={ACTION_BTN_VIEW}
                        onClick={() => navigate(`/result/${entry.id}`)}
                      >
                        View
                      </button>
                      <button
                        type="button"
                        className={ACTION_BTN_DELETE}
                        onClick={() => onDelete(entry.id)}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ── LLM cards (severity, summary, description, reasoning) ───────────────────

function LLMCards({ entries, navigate, onDelete }) {
  if (entries.length === 0) {
    return (
      <Card>
        <p className="py-8 text-center text-text-muted">
          No LLM results yet. Run an analysis from New Ingestion using an LLM
          model.
        </p>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {entries.map((entry) => {
        const summary =
          entry.result?.analysis_summary || entry.result?.summary || '';
        const description =
          entry.result?.analysis_description || entry.result?.description || '';
        const reasoning = formatReason(
          entry.result?.reasoning ?? entry.result?.reason,
        );

        return (
          <Card key={entry.id} className="relative">
            {/* Header row */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3 flex-wrap">
                <SeverityBadge severity={entry.severity || 'Minor'} />
                <span className="text-sm font-medium text-text-muted">
                  {entry.modelName || entry.modelId}
                </span>
                <span className="text-xs text-text-muted">
                  {entry.timestamp}
                </span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  className={ACTION_BTN_VIEW}
                  onClick={() => navigate(`/result/${entry.id}`)}
                >
                  View
                </button>
                <button
                  type="button"
                  className={ACTION_BTN_DELETE}
                  onClick={() => onDelete(entry.id)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </button>
              </div>
            </div>

            {/* Fields */}
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-1">
                  Summary
                </h4>
                <p className="text-sm text-text-main leading-relaxed">
                  {summary || '—'}
                </p>
              </div>

              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-1">
                  Description
                </h4>
                <p className="text-sm text-text-main leading-relaxed line-clamp-3">
                  {description || '—'}
                </p>
              </div>

              <div className="sm:col-span-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-1">
                  Reasoning
                </h4>
                <p className="text-sm text-text-main leading-relaxed whitespace-pre-wrap line-clamp-4">
                  {reasoning || '—'}
                </p>
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
}
