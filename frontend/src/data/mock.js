export const BASELINE_MODELS = [
  { id: 'lr', name: 'Logistic Regression', version: 'v1.0' },
  { id: 'svm', name: 'SVM (Linear)', version: 'v1.0' },
  { id: 'rf', name: 'Random Forest', version: 'v1.0' },
  { id: 'nb', name: 'Naive Bayes', version: 'v1.0' },
  { id: 'xgb', name: 'XGBoost', version: 'v1.0' },
  { id: 'ensemble', name: 'Ensemble', version: 'v1.0' },
];

export const LLM_MODELS = [
  { id: 'ollama', name: 'Llama 3 8B (Zero-Shot)', version: 'v8b' },
  { id: 'finetuned', name: 'Llama 3 8B (Fine-Tuned)', version: 'v8b-ft' },
];

export const SEVERITY_COLORS = {
  Critical: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', glow: 'rgba(185, 28, 28, 0.6)' },
  Major: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', glow: 'rgba(180, 83, 9, 0.6)' },
  Minor: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200', glow: 'rgba(22, 163, 74, 0.6)' },
  Trivial: { bg: 'bg-sky-50', text: 'text-sky-700', border: 'border-sky-200', glow: 'rgba(2, 132, 199, 0.6)' },
};

export const MOCK_BASELINE_EVAL = [
  { model: 'Random Forest', accuracy: 0.85, precision: 0.82, recall: 0.84, f1: 0.83, latency: 45 },
  { model: 'SVM (Linear)', accuracy: 0.79, precision: 0.76, recall: 0.78, f1: 0.77, latency: 120 },
  { model: 'Logistic Regression', accuracy: 0.72, precision: 0.68, recall: 0.7, f1: 0.69, latency: 25 },
  { model: 'Naive Bayes', accuracy: 0.68, precision: 0.65, recall: 0.66, f1: 0.65, latency: 15 },
];

export const MOCK_LLM_EVAL = [
  { model: 'Llama 3 8B (Zero-Shot)', accuracy: 0.78, precision: 0.75, recall: 0.77, f1: 0.76, latency: 1200 },
  { model: 'Llama 3 8B (Fine-Tuned)', accuracy: 0.82, precision: 0.8, recall: 0.81, f1: 0.8, latency: 1100 },
];
