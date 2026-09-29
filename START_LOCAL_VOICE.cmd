@echo off
setlocal
cd /d "%~dp0"

rem Do not let Hermes' inherited PYTHONPATH override this project's venv.
set "PYTHONPATH="
set "PYTHONHOME="

rem Launch the existing single-instance tray runner without keeping a console
rem window attached to the long-running application.
set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYTHONW%" (
  where pythonw.exe >nul 2>&1
  if errorlevel 1 (
    echo GlitchScribe could not find pythonw.exe.
    echo Install the project environment or run: python -m pip install -r requirements.txt
    exit /b 1
  )
  set "PYTHONW=pythonw.exe"
)

start "" /b "%PYTHONW%" "%~dp0keepalive_launcher.py"
exit /b 0
