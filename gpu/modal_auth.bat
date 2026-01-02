@echo off
REM Simple Modal Authentication for Windows
REM Run this in Command Prompt (cmd.exe), NOT Git Bash

echo.
echo ================================================
echo   MODAL AUTHENTICATION SETUP
echo ================================================
echo.
echo IMPORTANT: Run this in Command Prompt, not Git Bash!
echo If you're seeing this in Git Bash, close it and:
echo   1. Press Windows+R
echo   2. Type: cmd
echo   3. Navigate here: cd C:\Users\priya\deconstruct\gpu
echo   4. Run: modal_auth.bat
echo.
pause
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    py -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if modal is installed
pip show modal >nul 2>&1
if errorlevel 1 (
    echo Installing Modal...
    pip install -q modal pydantic anthropic
    echo.
)

echo ================================================
echo   STEP 1: MODAL AUTHENTICATION
echo ================================================
echo.
echo A URL will appear below. You have 2 options:
echo.
echo OPTION A (Automatic):
echo   - The URL should open in your browser automatically
echo   - Click "Authorize" in the browser
echo   - Come back here when done
echo.
echo OPTION B (Manual - if browser doesn't open):
echo   1. Copy the URL that appears below
echo   2. Open your browser manually
echo   3. Paste the URL and press Enter
echo   4. Click "Authorize"
echo.
echo Press any key to start authentication...
pause >nul
echo.

modal token set

echo.
echo ================================================
echo   CHECKING AUTHENTICATION...
echo ================================================
echo.

REM Verify authentication
modal app list >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Authentication failed or incomplete.
    echo.
    echo Please try again or see MODAL_SETUP.md for help.
    pause
    exit /b 1
)

echo [SUCCESS] Modal authentication complete!
echo.
echo.
echo ================================================
echo   STEP 2: CREATE ANTHROPIC API SECRET
echo ================================================
echo.
echo You need to add your Anthropic API key.
echo Get it from: https://console.anthropic.com/
echo.
echo When prompted, paste your API key using:
echo   - Right-click in this window, OR
echo   - Press Ctrl+V (Windows 10/11)
echo.
echo The key will be hidden as you type (this is normal).
echo.
echo Press any key to continue...
pause >nul
echo.

modal secret create anthropic-api-key

echo.
echo ================================================
echo   SETUP COMPLETE!
echo ================================================
echo.
echo Next steps:
echo   1. Deploy the Modal app:
echo      modal deploy modal_app.py
echo.
echo   2. Start the web dashboard:
echo      cd ..\web
echo      npm install
echo      npm run dev
echo.
echo   3. Open http://localhost:3000
echo.
echo For detailed instructions, see README.md
echo.
pause
