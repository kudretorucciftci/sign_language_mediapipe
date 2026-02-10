@echo off
echo ==========================================
echo   Sign Language Mediapipe Starting
echo ==========================================
echo.

:: Check for Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11 not found. Please ensure Python 3.11 is installed.
    pause
    exit /b
)

echo [1/2] Checking dependencies...
py -3.11 -m pip install -r requirements.txt

echo.
echo [2/2] Starting GUI...
py -3.11 gui_inference.py

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Application closed or an error occurred.
)

pause
