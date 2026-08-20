@echo off
REM Crop Disease AI - launcher
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing/checking dependencies...
pip install -r requirements.txt

echo Starting Streamlit app...
streamlit run app\farm_app.py

pause
