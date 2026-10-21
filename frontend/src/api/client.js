/**
 * React API client for CyberTriage backend.
 * Uses absolute URL to FastAPI backend (port 8000) to avoid proxy/port mismatch.
 * Set VITE_API_URL to override (e.g. for production).
 */

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

/** Default timeout for predict (LLM can take ~1 min on local hardware). */
const PREDICT_TIMEOUT_MS = 120_000;

/**
 * Fetch with timeout using AbortController.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = PREDICT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return res;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out. Try a shorter input or a baseline model.');
    }
    throw err;
  }
}

/**
 * @returns {Promise<Array<{id: string, name: string}>>}
 */
export async function getModels() {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    const text = await res.text();
    if (!res.ok) {
      console.error('[getModels]', res.status, text);
      throw new Error(text || 'Failed to fetch models');
    }
    return JSON.parse(text);
  } catch (err) {
    console.error('[getModels]', err);
    throw err;
  }
}

/**
 * Run severity prediction.
 * Sends JSON body: { text: string, model_type: string } (e.g. {"text": "test", "model_type": "lr"}).
 * @returns {Promise<{severity: string, confidence?: number, probabilities?: object, model: string, reason?: unknown}>}
 */
export async function predict(modelId, text) {
  const url = `${API_BASE}/api/predict`;
  const body = JSON.stringify({ text: text.trim(), model_type: modelId });

  let res;
  try {
    res = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      },
      PREDICT_TIMEOUT_MS
    );
  } catch (err) {
    console.error('[predict] fetch failed', err);
    throw err;
  }

  let responseText;
  try {
    responseText = await res.text();
  } catch (err) {
    console.error('[predict] failed to read response body', err);
    throw new Error('Failed to read response: ' + (err instanceof Error ? err.message : String(err)));
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = JSON.parse(responseText);
      const d = data.detail;
      if (typeof d === 'string') detail = d;
      else if (Array.isArray(d)) detail = d.map((x) => x.msg ?? JSON.stringify(x)).join('; ');
      else if (d != null) detail = typeof d === 'object' ? JSON.stringify(d) : String(d);
    } catch (_) {
      if (responseText) detail = responseText;
    }
    console.error('[predict]', res.status, detail);
    throw new Error(detail || 'Prediction failed');
  }

  try {
    return JSON.parse(responseText);
  } catch (err) {
    console.error('[predict] invalid JSON response', responseText?.slice(0, 200), err);
    throw new Error('Invalid JSON in prediction response');
  }
}

/**
 * Fetch batch evaluation metrics (from results/ CSVs via API).
 */
export async function getEvaluation() {
  try {
    const res = await fetch(`${API_BASE}/api/evaluation`);
    const text = await res.text();
    if (!res.ok) {
      console.error('[getEvaluation]', res.status, text);
      throw new Error(text || 'Failed to fetch evaluation metrics');
    }
    return JSON.parse(text);
  } catch (err) {
    console.error('[getEvaluation]', err);
    throw err;
  }
}

/** Timeout for run evaluation (match backend LLM timeout: 2 hours). */
const RUN_EVALUATION_TIMEOUT_MS = 2 * 60 * 60 * 1000;

/**
 * Run evaluation for a specific model.
 * @param {{ type: 'baseline'|'llm', model_id: string, batch_size: number }} body
 */
export async function runEvaluationRun(body) {
  try {
    const res = await fetchWithTimeout(
      `${API_BASE}/api/evaluation/run`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
      RUN_EVALUATION_TIMEOUT_MS
    );
    const text = await res.text();
    if (!res.ok) {
      let detail = text;
      try {
        const data = JSON.parse(text);
        if (data.detail) detail = data.detail;
      } catch (_) {}
      throw new Error(detail || 'Evaluation run failed');
    }
    return JSON.parse(text || '{}');
  } catch (err) {
    console.error('[runEvaluationRun]', err);
    throw err;
  }
}

/**
 * Parse an uploaded document (PDF, DOCX, TXT, LOG, JSON) and return extracted text.
 * @param {File} file
 * @returns {Promise<{ text: string }>}
 */
export async function parseDocument(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/api/parse-document`, {
    method: 'POST',
    body: form,
  });
  const text = await res.text();
  if (!res.ok) {
    let detail = text;
    try {
      const data = JSON.parse(text);
      if (data.detail) detail = data.detail;
    } catch (_) {}
    throw new Error(detail || 'Document parsing failed');
  }
  return JSON.parse(text || '{}');
}

export async function fetchResults(type) {
  try {
    const res = await fetch(`${API_BASE}/api/results/${type}`);
    if (!res.ok) return { exists: false, data: [] };
    const data = await res.json();
    return { exists: true, data };
  } catch (err) {
    console.error('[fetchResults]', err);
    return { exists: false, data: [] };
  }
}

/**
 * Load persisted ingestion history entries from backend local storage.
 * @returns {Promise<Array<object>>}
 */
export async function getHistoryEntries() {
  const res = await fetch(`${API_BASE}/api/history`);
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || 'Failed to fetch history');
  }
  const data = JSON.parse(text || '[]');
  return Array.isArray(data) ? data : [];
}

/**
 * Save one ingestion history entry to backend local storage.
 * @param {object} entry
 * @returns {Promise<void>}
 */
export async function saveHistoryEntry(entry) {
  const res = await fetch(`${API_BASE}/api/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entry }),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || 'Failed to save history');
  }
}

/**
 * Delete one persisted history entry by id.
 * @param {string} entryId
 * @returns {Promise<void>}
 */
export async function deleteHistoryEntry(entryId) {
  const res = await fetch(`${API_BASE}/api/history/${encodeURIComponent(entryId)}`, {
    method: 'DELETE',
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(text || 'Failed to delete history entry');
  }
}
