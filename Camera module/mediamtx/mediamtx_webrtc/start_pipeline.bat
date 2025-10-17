@echo off
chcp 65001 >nul
echo Starting Video Inference Pipeline...

REM Change to script directory
cd /d "%~dp0"

REM Check if MediaMTX exists
if not exist "mediamtx.exe" (
    echo Error: mediamtx.exe not found in laptop directory!
    echo Please download it first.
    pause
    exit /b 1
)

echo Starting MediaMTX server...
start "MediaMTX Server" mediamtx.exe mediamtx.yml

REM Wait for MediaMTX to start
timeout /t 3 /nobreak >nul

echo Starting inference server...
python inference_server.py

echo.
echo Pipeline stopped.
echo Closing MediaMTX...
taskkill /f /im mediamtx.exe >nul 2>&1
pause