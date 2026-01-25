@echo off
REM Setup script for Python virtual environments
REM This script creates isolated Python environments for both v1 and v2 projects

echo.
echo ========================================
echo Setting up Python Environments
echo ========================================
echo.

cd /d C:\proje

REM Create v2 environment (skin-quality-qa)
echo [1/4] Creating virtual environment for skin-quality-qa (v2)...
python -m venv skin-quality-qa\.venv
echo.

REM Activate v2 and install dependencies
echo [2/4] Installing dependencies for skin-quality-qa...
call skin-quality-qa\.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r skin-quality-qa\requirements.txt
call deactivate
echo.

REM Create v1 environment (skin-noise-detection)
echo [3/4] Creating virtual environment for skin-noise-detection (v1)...
python -m venv skin-noise-detection\.venv
echo.

REM Activate v1 and install dependencies
echo [4/4] Installing dependencies for skin-noise-detection...
call skin-noise-detection\.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r skin-noise-detection\requirements.txt
call deactivate
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual Environments Created:
echo   - skin-quality-qa\.venv    (v2: LLM Classification)
echo   - skin-noise-detection\.venv (v1: Model Training)
echo.
echo Next Steps:
echo.
echo For v2 (skin-quality-qa):
echo   1. Activate: .\skin-quality-qa\.venv\Scripts\activate.bat
echo   2. Setup: copy .env.example .env
echo   3. Add Google API key to .env
echo   4. Run: python -m src.enrich_labels_with_llm
echo.
echo For v1 (skin-noise-detection):
echo   1. Activate: .\skin-noise-detection\.venv\Scripts\activate.bat
echo   2. Run: python src/prepare.py
echo   3. Run: python src/model.py
echo.
pause
