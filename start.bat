@echo off
REM Installs the Python packages, runs first-time setup if needed, then starts Analytixa on http://127.0.0.1:4000
cd /d "%~dp0"
python -m pip install -r requirements.txt
if not exist config.env (
  python setup.py
  if errorlevel 1 goto end
)
python server.py
:end
pause
