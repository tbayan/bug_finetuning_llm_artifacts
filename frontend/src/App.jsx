import { Routes, Route, Navigate } from 'react-router-dom';
import TopBar from './components/layout/TopBar';
import LoginScreen from './components/screens/LoginScreen';
import IngestScreen from './components/screens/IngestScreen';
import ResultRouter from './components/screens/ResultRouter';
import EvaluationScreen from './components/screens/EvaluationScreen';
import HistoryScreen from './components/screens/HistoryScreen';
import { useAuth } from './context/AuthContext';

function ProtectedLayout({ children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return (
    <>
      <TopBar />
      {children}
    </>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginScreen />} />
      <Route
        path="/"
        element={
          <ProtectedLayout>
            <Navigate to="/ingest" replace />
          </ProtectedLayout>
        }
      />
      <Route
        path="/ingest"
        element={
          <ProtectedLayout>
            <IngestScreen />
          </ProtectedLayout>
        }
      />
      <Route
        path="/result/:id"
        element={
          <ProtectedLayout>
            <ResultRouter />
          </ProtectedLayout>
        }
      />
      <Route
        path="/evaluation"
        element={
          <ProtectedLayout>
            <EvaluationScreen />
          </ProtectedLayout>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedLayout>
            <HistoryScreen />
          </ProtectedLayout>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
