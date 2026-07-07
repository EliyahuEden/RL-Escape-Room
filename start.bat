@echo off
rem ============================================================
rem  RL Escape Room - one-click launcher
rem
rem    start.bat        first run: sets up EVERYTHING (Python venv,
rem                     pip packages, npm packages, frontend build),
rem                     then runs the app on http://localhost:8000.
rem                     Later runs: starts the app directly.
rem    start.bat dev    developer mode: backend (8000) + Vite
rem                     hot-reload frontend (5173) in two windows
rem    start.bat build  rebuild the frontend bundle, then run
rem
rem  Prerequisites (install once, from the official sites):
rem    * Python 3.10+  (64-bit)  https://python.org
rem    * Node.js 18+             https://nodejs.org
rem ============================================================
cd /d "%~dp0"

rem ---------- 1. Python virtual environment (created on first run) ----------
if exist ".venv\Scripts\python.exe" goto have_venv
echo [start.bat] First run: creating the Python virtual environment...
py -3 -m venv .venv 2>nul
if exist ".venv\Scripts\python.exe" goto have_venv
python -m venv .venv
if exist ".venv\Scripts\python.exe" goto have_venv
echo.
echo [start.bat] ERROR: could not create .venv
echo             Install 64-bit Python 3.10+ from https://python.org
echo             (tick "Add python.exe to PATH" in the installer), then re-run.
pause
exit /b 1

:have_venv
set PY=.venv\Scripts\python.exe

rem ---------- 2. Python packages (installed when missing) ----------
%PY% -c "import fastapi, uvicorn, torch, numpy" >nul 2>nul
if not errorlevel 1 goto deps_ok
echo [start.bat] Installing Python packages - the first time can take a few
echo             minutes (PyTorch is a large download). Please wait...
%PY% -m pip install -r requirements.txt
if not errorlevel 1 goto deps_ok
echo.
echo [start.bat] ERROR: pip install failed - check your internet connection,
echo             then re-run start.bat.
pause
exit /b 1

:deps_ok
if "%1"=="dev" goto dev
if "%1"=="build" goto build
if not exist "frontend\dist\index.html" goto build
goto run

rem ---------- 3. Frontend build (first run / start.bat build) ----------
:build
call :need_npm || exit /b 1
call :need_node_modules || exit /b 1
echo [start.bat] Building the frontend...
pushd frontend
call npm.cmd run build
set BUILD_ERR=%errorlevel%
popd
if "%BUILD_ERR%"=="0" goto run
echo.
echo [start.bat] ERROR: frontend build failed - see the messages above.
pause
exit /b 1

rem ---------- 4. Run (single server: site + API on port 8000) ----------
:run
echo [start.bat] Starting RL Escape Room on http://localhost:8000 ...
echo [start.bat] Close this window (or press Ctrl+C) to stop the server.
start /b "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8000"
%PY% -m backend.api.main
exit /b

rem ---------- developer mode ----------
:dev
call :need_npm || exit /b 1
call :need_node_modules || exit /b 1
echo [start.bat] Developer mode: backend on 8000, Vite dev server on 5173.
start "RL Escape Room - backend" cmd /k %PY% -m backend.api.main
pushd frontend
start "RL Escape Room - frontend" cmd /k npm.cmd run dev
popd
start /b "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:5173"
exit /b

rem ---------- helpers ----------
:need_npm
where npm.cmd >nul 2>nul
if not errorlevel 1 exit /b 0
echo.
echo [start.bat] ERROR: npm not found - install Node.js 18+ from
echo             https://nodejs.org then re-run start.bat.
pause
exit /b 1

:need_node_modules
if exist "frontend\node_modules" exit /b 0
echo [start.bat] First run: installing frontend packages...
pushd frontend
call npm.cmd install --no-audit --no-fund
set NPM_ERR=%errorlevel%
popd
if "%NPM_ERR%"=="0" exit /b 0
echo.
echo [start.bat] ERROR: npm install failed - check your internet connection,
echo             then re-run start.bat.
pause
exit /b 1
