@echo off
rem ============================================================
rem  RL Escape Room - one-click launcher
rem
rem    start.bat        run the app on http://localhost:8000
rem                     (builds the frontend first if needed)
rem    start.bat dev    developer mode: backend (8000) + Vite
rem                     hot-reload frontend (5173) in two windows
rem    start.bat build  rebuild the frontend bundle, then run
rem ============================================================
cd /d "%~dp0"

if "%1"=="dev" goto dev
if "%1"=="build" goto build

if not exist "frontend\dist\index.html" goto build
goto run

:build
echo [start.bat] Building the frontend...
if not exist "frontend\node_modules" call npm.cmd --prefix frontend install
call npm.cmd --prefix frontend run build
if errorlevel 1 (
  echo [start.bat] Frontend build failed - is Node.js installed?
  pause
  exit /b 1
)

:run
echo [start.bat] Starting RL Escape Room on http://localhost:8000 ...
echo [start.bat] Close this window (or press Ctrl+C) to stop the server.
start /b "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"
python -m backend.api.main
exit /b

:dev
echo [start.bat] Developer mode: backend on 8000, Vite dev server on 5173.
start "RL Escape Room - backend" cmd /k python -m backend.api.main
start "RL Escape Room - frontend" cmd /k npm.cmd --prefix frontend run dev
start /b "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:5173"
exit /b
