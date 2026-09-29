@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONHOME="
if not exist ".venv\Scripts\python.exe" (
  echo GlitchScribe virtual environment is missing: .venv\Scripts\python.exe
  exit /b 1
)
".venv\Scripts\python.exe" -m apps.call_response %*
