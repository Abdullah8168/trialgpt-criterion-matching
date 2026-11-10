#!/usr/bin/env bash
# One-click launcher (Linux/macOS). Opens http://127.0.0.1:7860
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install torch && .venv/bin/pip install -r requirements.txt; }
[ -f outputs/adapter/adapter_config.json ] || .venv/bin/python scripts/download_weights.py
.venv/bin/python app.py
