@echo off
echo ============================================
echo Starting Deconstruct API Server
echo ============================================
echo.

cd gpu
call venv\Scripts\activate.bat
python simple_api.py
