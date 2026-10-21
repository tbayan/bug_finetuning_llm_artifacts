import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, Lightbulb, Trash2, Upload, Copy } from 'lucide-react';
import Card from '../shared/Card';
import PrimaryButton from '../shared/PrimaryButton';
import Dropdown from '../shared/Dropdown';
import { useHistory } from '../../context/HistoryContext';
import { BASELINE_MODELS as FALLBACK_BASELINE, LLM_MODELS as FALLBACK_LLM } from '../../data/mock';
import * as api from '../../api/client';

const MODEL_TYPE_OPTIONS = [
  { value: 'baseline', label: 'Baseline ML' },
  { value: 'llm', label: 'LLMs' },
];

/** Registry baseline model ids (src.engines.registry); LLM = all others. */
const BASELINE_IDS = new Set(['lr', 'svm', 'rf', 'nb', 'xgb', 'ensemble', 'codebert']);

function splitModels(apiModels) {
  const baseline = [];
  const llm = [];
  for (const m of apiModels || []) {
    const item = { id: m.id, name: m.name, version: 'v1.0' };
    if (BASELINE_IDS.has(m.id)) baseline.push(item);
    else llm.push(item);
  }
  return {
    baseline: baseline.length ? baseline : FALLBACK_BASELINE,
    llm: llm.length ? llm : FALLBACK_LLM,
  };
}

export default function IngestScreen() {
  const [modelType, setModelType] = useState('baseline');
  const [modelId, setModelId] = useState(FALLBACK_BASELINE[0]?.id || 'lr');
  const [models, setModels] = useState({ baseline: FALLBACK_BASELINE, llm: FALLBACK_LLM });
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingElapsedSec, setRecordingElapsedSec] = useState(0);
  const [speechError, setSpeechError] = useState(null);
  const fileInputRef = useRef(null);
  const recognitionRef = useRef(null);
  const { addEntry } = useHistory();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isRecording) return;
    setRecordingElapsedSec(0);
    const start = Date.now();
    const id = setInterval(() => {
      setRecordingElapsedSec(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [isRecording]);

  const formatElapsed = (sec) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  };

  const getSpeechRecognition = useCallback(() => {
    if (typeof window === 'undefined') return null;
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }, []);

  const handleMicClick = useCallback(() => {
    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) {
      setSpeechError('Speech recognition is not supported in this browser. Try Chrome or Edge.');
      return;
    }
    setSpeechError(null);
    setError(null);

    if (isRecording) {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (_) {}
        recognitionRef.current = null;
      }
      setIsRecording(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.onresult = (e) => {
      let finalTranscript = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) {
          finalTranscript += e.results[i][0].transcript;
        }
      }
      if (finalTranscript) {
        setText((prev) => (prev ? prev + ' ' + finalTranscript : finalTranscript));
      }
    };
    recognition.onend = () => {
      if (recognitionRef.current) {
        setIsRecording(false);
        recognitionRef.current = null;
      }
    };
    recognition.onerror = (e) => {
      if (e.error === 'not-allowed') {
        setSpeechError('Microphone access denied. Allow the site to use your microphone.');
      } else if (e.error !== 'aborted') {
        setSpeechError(e.error || 'Speech recognition error.');
      }
      setIsRecording(false);
      recognitionRef.current = null;
    };
    try {
      recognition.start();
      recognitionRef.current = recognition;
      setIsRecording(true);
    } catch (err) {
      setSpeechError(err instanceof Error ? err.message : 'Could not start microphone.');
    }
  }, [isRecording, getSpeechRecognition]);

  useEffect(() => {
    api
      .getModels()
      .then((list) => {
        const { baseline, llm } = splitModels(list);
        setModels({ baseline, llm });
        setModelId((current) => {
          const nextList = modelType === 'baseline' ? baseline : llm;
          const exists = nextList.some((m) => m.id === current);
          return exists ? current : nextList[0]?.id || current;
        });
      })
      .catch(() => {
        setModels({ baseline: FALLBACK_BASELINE, llm: FALLBACK_LLM });
      });
  }, []);

  const currentList = modelType === 'baseline' ? models.baseline : models.llm;
  const modelOptions = currentList.map((m) => ({ value: m.id, label: m.name }));

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setError(null);
    setLoading(true);
    try {
      const result = await api.predict(modelId, text.trim());
      const entry = addEntry({
        modelType,
        modelId,
        modelName: currentList.find((m) => m.id === modelId)?.name,
        text: text.trim(),
        textSnippet: text.slice(0, 80) + (text.length > 80 ? '…' : ''),
        severity: result.severity || result.priority || 'Major',
        result,
      });
      navigate(`/result/${entry.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const acceptTypes = ['.txt', '.log', '.json', '.pdf', '.docx'];
  const maxSizeMB = 10;
  const maxSizeBytes = maxSizeMB * 1024 * 1024;

  const readFile = useCallback(async (file) => {
    const ext = '.' + (file.name || '').split('.').pop()?.toLowerCase();
    if (!acceptTypes.includes(ext)) {
      setError(`Unsupported format. Use: ${acceptTypes.join(', ')}`);
      return;
    }
    if (file.size > maxSizeBytes) {
      setError(`File too large (max ${maxSizeMB}MB).`);
      return;
    }
    setError(null);
    if (ext === '.pdf' || ext === '.docx') {
      try {
        const { text: extracted } = await api.parseDocument(file);
        setText((prev) => prev + (extracted || ''));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Document parsing failed.');
      }
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setText((prev) => prev + (reader.result || ''));
    reader.onerror = () => setError('Failed to read file.');
    reader.readAsText(file, 'utf-8');
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) readFile(file).catch(() => {});
  }, [readFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleFileSelect = useCallback((e) => {
    const file = e.target?.files?.[0];
    if (file) readFile(file).catch(() => {});
    e.target.value = '';
  }, [readFile]);

  const handleCopyAll = useCallback(() => {
    if (!text) return;
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).catch(() => {
        // Silently ignore clipboard errors to avoid disrupting the user.
      });
    }
  }, [text]);

  return (
    <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text-main">Ingest New Bug Report</h1>
        <p className="mt-2 text-text-muted">Paste logs or a bug description and choose a classification model.</p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2 lg:gap-12">
        {/* Left column */}
        <div className="flex flex-col gap-8">
          <Card>
            <h2 className="text-lg font-semibold text-text-main mb-4">Classification Model</h2>
            <div className="flex gap-4 mb-4">
              {MODEL_TYPE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => {
                    setModelType(opt.value);
                    const list = opt.value === 'baseline' ? models.baseline : models.llm;
                    setModelId(list[0]?.id || '');
                  }}
                  className={`flex-1 rounded-xl border px-4 py-3 text-sm font-medium transition-colors ${
                    modelType === opt.value
                      ? 'border-primary bg-primary text-text-main'
                      : 'border-slate-200 bg-white text-text-muted hover:bg-slate-50'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <Dropdown
              label="Model Version"
              value={modelId}
              onChange={setModelId}
              options={modelOptions}
            />
          </Card>

          <Card className="bg-slate-50/50 border-slate-100">
            <div className="flex items-start gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-text-main">Tip</h3>
                {modelType === 'baseline' ? (
                  <p className="mt-1 text-sm text-text-muted">
                    <strong className="text-text-main">Traditional ML</strong> predicts severity only (label and probabilities). Add stack traces or drop a document (TXT, LOG, JSON, PDF, DOCX) for better results.
                  </p>
                ) : (
                  <p className="mt-1 text-sm text-text-muted">
                    <strong className="text-text-main">LLM</strong> adds natural-language reasoning for the decision. You get a severity label plus an explanation you can export to Jira.
                  </p>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-6">
          <Card className="overflow-hidden p-0">
            <div className="flex items-center justify-between gap-2 px-6 py-4 border-b border-slate-100 bg-white">
              <span className="text-base font-semibold text-text-main">Bug Description / Logs</span>
              <div className="flex items-center gap-2">
                {isRecording && (
                  <span className="flex items-center gap-2 text-sm text-red-600 tabular-nums">
                    <span className="relative flex h-2 w-2" aria-hidden>
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
                    </span>
                    {formatElapsed(recordingElapsedSec)}
                  </span>
                )}
                <button
                  type="button"
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:bg-slate-100 hover:text-text-main transition-colors disabled:opacity-50"
                  aria-label="Copy all text"
                  title="Copy all text"
                  onClick={handleCopyAll}
                  disabled={!text}
                >
                  <Copy className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:bg-slate-100 hover:text-text-main transition-colors disabled:opacity-50"
                  aria-label="Clear all text"
                  title="Clear all text"
                  onClick={() => setText('')}
                  disabled={isRecording}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  onClick={handleMicClick}
                  aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
                  title={isRecording ? 'Stop recording' : 'Start voice input'}
                  className={`flex h-8 w-8 items-center justify-center rounded-lg transition-colors ${
                    isRecording
                      ? 'bg-red-100 text-red-600 hover:bg-red-200'
                      : 'text-text-muted hover:bg-slate-100 hover:text-text-main'
                  }`}
                >
                  <Mic className="h-4 w-4" />
                </button>
              </div>
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste stack trace, error logs, or user description here..."
              rows={14}
              className="w-full resize-none border-none bg-transparent px-6 py-4 text-sm text-text-main placeholder:text-text-muted focus:ring-0 min-h-[320px] font-mono"
            />
            {/* Bottom attachments: upload zone */}
            <div className="px-6 py-6 bg-slate-50 border-t border-slate-100">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.log,.json,.pdf,.docx,text/plain,application/json,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                className="hidden"
                onChange={handleFileSelect}
              />
              <div
                role="button"
                tabIndex={0}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`flex-1 w-full border-2 border-dashed rounded-xl p-4 transition-colors cursor-pointer group flex items-center justify-center gap-3 bg-white/50 hover:border-primary hover:bg-primary/5 ${dragOver ? 'border-primary bg-primary/5' : 'border-slate-300'}`}
              >
                <div className="p-2 bg-slate-100 rounded-lg text-slate-400 group-hover:text-primary transition-colors">
                  <Upload className="h-5 w-5" />
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-text-main group-hover:text-primary transition-colors">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-xs text-text-muted">TXT, LOG, JSON, PDF, DOCX up to 10MB</p>
                </div>
              </div>
            </div>
          </Card>

          {speechError && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800" role="alert">
              {speechError}
            </div>
          )}
          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700" role="alert">
              {error}
            </div>
          )}
          <PrimaryButton onClick={handleAnalyze} disabled={loading || !text.trim()}>
            {loading ? 'Processing neural network…' : 'Analyze Bug'}
          </PrimaryButton>
        </div>
      </div>
    </main>
  );
}
