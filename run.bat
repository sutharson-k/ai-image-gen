@echo off
setlocal
title Nebula Diffusion // Real-Time Studio
cd /d "%~dp0"
set PYTHONUNBUFFERED=1

echo ========================================================
echo   NEBULA DIFFUSION: REAL-TIME PROGRESSIVE STUDIO
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo.
    echo   Create it once with:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [*] Checking if port 7860 is already in use...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr /R /C:":7860 .*LISTENING"') do (
    if not "%%a"=="0" (
        echo [*] Port 7860 is busy ^(PID %%a^). Stopping the old instance...
        taskkill /PID %%a /F >nul 2>&1
    )
)
echo [*] Port 7860 is free.
echo.
echo [*] Starting the studio server...
echo [*] The web UI opens in a few seconds. The AI runtime keeps loading
echo     in the background, so the first generation can take a moment.
echo [*] Manual address: http://127.0.0.1:7860
echo.

".venv\Scripts\python.exe" app.py
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
    echo [ERROR] The studio stopped with exit code %EXITCODE%.
    echo         Read the message above for the cause.
) else (
    echo [*] Studio closed.
)
echo.
pause
