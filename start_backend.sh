#!/usr/bin/env bash
# Start the FastAPI backend (api/server.py) with venv and deps installed.
# Run from project root: ./start_backend.sh

set -e
cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS="${REQUIREMENTS:-requirements-backend.txt}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment ..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Installing dependencies from $REQUIREMENTS ..."
pip install -q -r "$REQUIREMENTS"

echo "Starting FastAPI server at http://127.0.0.1:8000 ..."
exec python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
