#!/usr/bin/env sh
# One-command launcher for Linux / macOS:  sh run.sh
# Creates .venv (Python 3.12), installs dependencies, downloads the models, starts the app.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then uv venv .venv --python 3.12; else python3 -m venv .venv; fi
fi
if [ ! -f .venv/.installed ]; then
  if command -v uv >/dev/null 2>&1; then uv pip install --python .venv/bin/python -r requirements.txt
  else .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt; fi
  touch .venv/.installed
fi
.venv/bin/python scripts/download_models.py
exec .venv/bin/python -m streamlit run app/streamlit_app.py
