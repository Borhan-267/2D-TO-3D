@echo off
title IRC Chem3D Studio Launcher
echo ===================================================
echo   Starting IRC Chem3D Studio... Please wait...
echo ===================================================

if not exist "venv" (
    echo [1/3] Creating virtual environment for first-time setup...
    python -m venv venv
)

echo [2/3] Checking and installing required libraries...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

echo [3/3] Launching IRC Chem3D Studio in your browser...
python -m streamlit run app.py
pause