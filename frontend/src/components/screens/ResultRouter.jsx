import { useParams, Navigate } from 'react-router-dom';
import { useHistory } from '../../context/HistoryContext';
import BaselineResult from './BaselineResult';
import LLMResult from './LLMResult';

export default function ResultRouter() {
  const { id } = useParams();
  const { getEntry } = useHistory();
  const entry = getEntry(id);

  if (!entry) return <Navigate to="/history" replace />;
  if (entry.modelType === 'baseline') return <BaselineResult entry={entry} />;
  return <LLMResult entry={entry} />;
}
