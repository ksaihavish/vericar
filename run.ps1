# One-command launcher for Windows:  powershell -ExecutionPolicy Bypass -File run.ps1
# Creates .venv (Python 3.12), installs dependencies, downloads the models, starts the app.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) { uv venv .venv --python 3.12 }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv }
    else { python -m venv .venv }
}
if (-not (Test-Path ".venv\.installed")) {
    if (Get-Command uv -ErrorAction SilentlyContinue) { uv pip install --python $py -r requirements.txt }
    else { & $py -m pip install --upgrade pip; & $py -m pip install -r requirements.txt }
    New-Item ".venv\.installed" -ItemType File | Out-Null
}
& $py scripts\download_models.py
& $py -m streamlit run app\streamlit_app.py
