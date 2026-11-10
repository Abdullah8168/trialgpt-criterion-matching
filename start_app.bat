@echo off
REM One-click launcher for the screening app (Windows). Opens http://127.0.0.1:7860
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  .venv\Scripts\python -m pip install -r requirements.txt
)
if not exist outputs\adapter\adapter_config.json .venv\Scripts\python scripts\download_weights.py
.venv\Scripts\python app.py
pause
