import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import * as api from '../api/client';

const STORAGE_KEY = 'cybertriage_history';

const HistoryContext = createContext(null);

export function HistoryProvider({ children }) {
  const [entries, setEntries] = useState([]);

  useEffect(() => {
    let cancelled = false;
    api
      .getHistoryEntries()
      .then((rows) => {
        if (!cancelled) setEntries(rows);
      })
      .catch(() => {
        // Fallback to local storage if backend is unavailable.
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw && !cancelled) setEntries(JSON.parse(raw));
        } catch (_) {}
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
    } catch (_) {}
  }, [entries]);

  const addEntry = useCallback((entry) => {
    const id = `id-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' ');
    const newEntry = { ...entry, id, timestamp };
    setEntries((prev) => [newEntry, ...prev]);
    // Persist in project-local backend storage (results/history_entries.json).
    api.saveHistoryEntry(newEntry).catch(() => {});
    return newEntry;
  }, []);

  const getEntry = useCallback((id) => {
    return entries.find((e) => e.id === id) || null;
  }, [entries]);

  const removeEntry = useCallback((id) => {
    const target = String(id || '');
    setEntries((prev) => prev.filter((e) => String(e.id || '') !== target));
    api.deleteHistoryEntry(target).catch(() => {});
  }, []);

  return (
    <HistoryContext.Provider value={{ entries, addEntry, getEntry, removeEntry }}>
      {children}
    </HistoryContext.Provider>
  );
}

export function useHistory() {
  const ctx = useContext(HistoryContext);
  if (!ctx) throw new Error('useHistory must be used within HistoryProvider');
  return ctx;
}
