#!/usr/bin/env bash
# One command: start backend + frontend. Run from anywhere, e.g.:
#   /path/to/thesis-bug-triage/run.sh
#   or: cd thesis-bug-triage && ./run.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BACKEND_PID=""
cleanup() {
  if [[ -n "$BACKEND_PID" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup INT TERM

echo "Starting backend at http://127.0.0.1:8000 ..."
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS="${REQUIREMENTS:-requirements-backend.txt}"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q -r "$REQUIREMENTS"
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

sleep 2
echo "Starting frontend at http://localhost:3000 ..."
(cd "$ROOT/frontend" && npm run dev)
cleanup
