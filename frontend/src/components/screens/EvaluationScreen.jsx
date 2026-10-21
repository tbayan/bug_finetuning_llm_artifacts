import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import Card from '../shared/Card';
import Dropdown from '../shared/Dropdown';
import DataTable from '../shared/DataTable';
import * as api from '../../api/client';

const BASELINE_IDS = new Set(['lr', 'svm', 'rf', 'nb', 'xgb', 'ensemble', 'codebert']);

function splitModels(list) {
  const baseline = [];
  const llm = [];
  for (const m of list || []) {
    const item = { value: m.id, label: m.name };
    if (BASELINE_IDS.has(m.id)) baseline.push(item);
    else llm.push(item);
  }
  return { baseline, llm };
}

const BATCH_SIZES = [
  { value: '50', label: '50' },
  { value: '100', label: '100' },
  { value: '200', label: '200' },
  { value: '500', label: '500' },
  { value: '1000', label: '1000' },
  { value: '2000', label: '2000' },
  { value: '2500', label: '2500' },
  { value: '4000', label: '4000' },
];

const EVAL_COLUMNS = [
  { key: 'model', label: 'Model Name' },
  { key: 'samples', label: 'Samples', render: (val) => (val == null || val === '' ? '—' : Number(val)), highlight: false },
  { key: 'accuracy', label: 'Accuracy', percent: true },
  { key: 'precision', label: 'Precision', percent: true },
  { key: 'recall', label: 'Recall', percent: true },
  { key: 'f1', label: 'F1 Score', percent: true },
  {
    key: 'duration_sec',
    label: 'Duration (s)',
    render: (val) => (val == null || val === '' ? '—' : (typeof val === 'number' ? val.toFixed(2) : val)),
  },
];

function mapEvalRows(rows) {
  return rows.map((r) => ({
    ...r,
    samples: r.samples != null ? r.samples : '—',
    duration_sec: r.duration_sec != null ? r.duration_sec : '—',
  }));
}

function formatElapsed(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function EvaluationScreen() {
  const [batchSize, setBatchSize] = useState('50');
  const [baselineRows, setBaselineRows] = useState([]);
  const [llmRows, setLlmRows] = useState([]);
  const [lastRunBaseline, setLastRunBaseline] = useState(null);
  const [lastRunLlms, setLastRunLlms] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [runError, setRunError] = useState(null);
  const [running, setRunning] = useState(false);
  const [elapsedRunSec, setElapsedRunSec] = useState(0);
  const [models, setModels] = useState({ baseline: [], llm: [] });
  const [selectedBaselineId, setSelectedBaselineId] = useState('lr');
  const [selectedLlmId, setSelectedLlmId] = useState('ollama');

  useEffect(() => {
    if (!running) return;
    setElapsedRunSec(0);
    const start = Date.now();
    const id = setInterval(() => {
      setElapsedRunSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    api.getModels().then((list) => {
      const { baseline, llm } = splitModels(list);
      setModels({ baseline, llm });
      if (baseline.length && !baseline.some((m) => m.value === selectedBaselineId)) setSelectedBaselineId(baseline[0].value);
      if (llm.length && !llm.some((m) => m.value === selectedLlmId)) setSelectedLlmId(llm[0].value);
    }).catch(() => {});
  }, []);

  const refetchEvaluation = () => {
    return api.getEvaluation().then((data) => {
      setBaselineRows(mapEvalRows(data.baseline || []));
      setLlmRows(mapEvalRows(data.llms || []));
      setLastRunBaseline(data.last_run_baseline ?? null);
      setLastRunLlms(data.last_run_llms ?? null);
    });
  };

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setLoading(true);
    refetchEvaluation()
      .catch((err) => {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : 'Failed to load evaluation data.';
          setError(
            `${msg} Ensure the API is running (npm run dev). To see metrics, run scripts/evaluate_baselines.py and scripts/evaluate_ollama.py.`
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const handleRunBaseline = async () => {
    setRunError(null);
    setRunning(true);
    try {
      await api.runEvaluationRun({
        type: 'baseline',
        model_id: selectedBaselineId,
        batch_size: Number(batchSize),
      });
      await refetchEvaluation();
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Run failed');
    } finally {
      setRunning(false);
    }
  };

  const handleRunLlm = async () => {
    setRunError(null);
    setRunning(true);
    try {
      await api.runEvaluationRun({
        type: 'llm',
        model_id: selectedLlmId,
        batch_size: Number(batchSize),
      });
      await refetchEvaluation();
    } catch (err) {
      setRunError(err instanceof Error ? err.message : 'Run failed');
    } finally {
      setRunning(false);
    }
  };

  if (loading && baselineRows.length === 0 && llmRows.length === 0) {
    return (
      <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-text-main">Batch Evaluation Metrics</h1>
          <p className="mt-2 text-text-muted">Compare baseline and LLM model performance.</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-text-muted">
          Loading evaluation data…
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text-main">Batch Evaluation Metrics</h1>
        <p className="mt-2 text-text-muted">Compare baseline and LLM model performance. Run evaluation for a selected model and batch size.</p>
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800" role="alert">
          {error} Run <code className="rounded bg-amber-100 px-1">scripts/evaluate_baselines.py</code>, <code className="rounded bg-amber-100 px-1">scripts/evaluate_llama_zeroshot.py</code> and <code className="rounded bg-amber-100 px-1">scripts/evaluate_llama_finetuned.py</code> to generate results.
        </div>
      )}

      {runError && (
        <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800" role="alert">
          {runError}
        </div>
      )}

      <div className="flex flex-col gap-12">
        <Card>
          <div className="flex flex-wrap items-end gap-4 mb-6">
            <div className="flex items-end gap-4">
              <div className="w-48">
                <Dropdown
                  label="Model"
                  value={selectedBaselineId}
                  onChange={setSelectedBaselineId}
                  options={models.baseline.length ? models.baseline : [{ value: 'lr', label: '—' }]}
                />
              </div>
              <div className="w-40">
                <Dropdown
                  label="Batch Size"
                  value={batchSize}
                  onChange={setBatchSize}
                  options={BATCH_SIZES}
                />
              </div>
            </div>
            <div className="ml-auto flex items-center gap-3 shrink-0">
              {running && (
                <span className="text-sm font-medium text-text-muted tabular-nums" aria-live="polite">
                  {formatElapsed(elapsedRunSec)}
                </span>
              )}
              <button
                type="button"
                onClick={handleRunBaseline}
                disabled={running}
                className="inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-medium text-text-main hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors h-[42px] box-border"
              >
                {running ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running…
                  </>
                ) : (
                  'Run evaluation'
                )}
              </button>
            </div>
          </div>
          <DataTable
            title="Baseline Models"
            subtitle={lastRunBaseline ? `Last run: ${lastRunBaseline}` : 'No results yet. Run evaluation above.'}
            columns={EVAL_COLUMNS}
            rows={baselineRows}
          />
        </Card>

        <Card>
          <div className="flex flex-wrap items-end gap-4 mb-6">
            <div className="flex items-end gap-4">
              <div className="w-56">
                <Dropdown
                  label="Model"
                  value={selectedLlmId}
                  onChange={setSelectedLlmId}
                  options={models.llm.length ? models.llm : [{ value: 'ollama', label: '—' }]}
                />
              </div>
              <div className="w-40">
                <Dropdown
                  label="Batch Size"
                  value={batchSize}
                  onChange={setBatchSize}
                  options={BATCH_SIZES}
                />
              </div>
            </div>
            <div className="ml-auto flex items-center gap-3 shrink-0">
              {running && (
                <span className="text-sm font-medium text-text-muted tabular-nums" aria-live="polite">
                  {formatElapsed(elapsedRunSec)}
                </span>
              )}
              <button
                type="button"
                onClick={handleRunLlm}
                disabled={running}
                className="inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-medium text-text-main hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors h-[42px] box-border"
              >
                {running ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running…
                  </>
                ) : (
                  'Run evaluation'
                )}
              </button>
            </div>
          </div>
          <DataTable
            title="Large Language Models (LLMs)"
            subtitle={lastRunLlms ? `Last run: ${lastRunLlms}` : 'No results yet. Run evaluation above.'}
            columns={EVAL_COLUMNS}
            rows={llmRows}
          />
        </Card>
      </div>
    </main>
  );
}
