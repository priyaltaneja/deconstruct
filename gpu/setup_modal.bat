@echo off
REM Modal Setup Script for Windows
REM This opens your browser to authenticate - no pasting needed!

echo ====================================
echo Modal Authentication Setup
echo ====================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
)

echo.
echo Starting Modal authentication...
echo This will open your browser automatically.
echo.

REM This opens your browser - no token pasting needed!
modal token set

echo.
echo ====================================
echo Next Steps:
echo ====================================
echo.
echo 1. Create API key secret:
echo    modal secret create anthropic-api-key
echo.
echo 2. You'll be prompted to enter your Anthropic API key
echo    (you can paste this with Shift+Insert or right-click)
echo.
echo 3. Deploy the app:
echo    modal deploy modal_app.py
echo.

pause
