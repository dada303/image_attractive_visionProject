@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_local.cmd first.
  pause
  exit /b 1
)
echo Open http://localhost:8766 after the model loads.
".venv\Scripts\python.exe" serve.py
pause
