@echo off
rem Double-click to start VeriCar locally on Windows.
rem Reuses an existing environment in %USERPROFILE%\vericar_work\.venv if present (fast);
rem otherwise falls back to run.ps1, which creates .venv and installs requirements once.
cd /d "%~dp0"
set "PY=%USERPROFILE%\vericar_work\.venv\Scripts\python.exe"
if exist "%PY%" (
  "%PY%" scripts\download_models.py
  "%PY%" -m streamlit run app\streamlit_app.py
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1
)
pause
