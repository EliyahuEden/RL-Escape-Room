#!/usr/bin/env bash
# ============================================================
#  RL Escape Room - one-command launcher (macOS / Linux)
#
#    ./start.sh        developer mode (DEFAULT): backend (8000) + Vite
#                      hot-reload frontend (5173). Every source edit shows
#                      up instantly - no rebuild needed. First run also sets
#                      up the Python venv + npm packages.
#    ./start.sh prod   single-server production mode on http://localhost:8000
#                      (rebuilds the frontend bundle first if it's missing)
#    ./start.sh dev    same as running ./start.sh with no arguments
#    ./start.sh build  force-rebuild the frontend bundle, then run (port 8000)
#
#  Prerequisites: Python 3.10+ and Node.js 18+
# ============================================================
set -e
cd "$(dirname "$0")"

# ---------- 1. Python virtual environment ----------
if [ ! -x .venv/bin/python ]; then
  echo "[start.sh] First run: creating the Python virtual environment..."
  python3 -m venv .venv
fi
PY=.venv/bin/python

# ---------- 2. Python packages ----------
if ! $PY -c "import fastapi, uvicorn, torch, numpy" >/dev/null 2>&1; then
  echo "[start.sh] Installing Python packages (PyTorch is a large download)..."
  $PY -m pip install -r requirements.txt
fi

# ---------- 3. Frontend packages ----------
if [ ! -d frontend/node_modules ]; then
  echo "[start.sh] First run: installing frontend packages..."
  (cd frontend && npm install --no-audit --no-fund)
fi

# ---------- developer hot-reload mode (default, or `dev`) ----------
if [ -z "$1" ] || [ "$1" = "dev" ]; then
  echo "[start.sh] Developer mode: backend on 8000, Vite dev server on 5173."
  echo "[start.sh] Open http://localhost:5173 - edits update automatically."
  $PY -m backend.api.main &
  BACKEND_PID=$!
  trap "kill $BACKEND_PID 2>/dev/null" EXIT
  (cd frontend && npm run dev)
  exit 0
fi

# ---------- production single-server build (prod / build) ----------
if [ "$1" = "build" ] || [ ! -f frontend/dist/index.html ]; then
  echo "[start.sh] Building the frontend..."
  (cd frontend && npm run build)
fi

# ---------- Run (single server: site + API on port 8000) ----------
echo "[start.sh] Starting RL Escape Room on http://localhost:8000 (Ctrl+C to stop)"
$PY -m backend.api.main
