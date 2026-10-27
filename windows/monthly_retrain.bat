@echo off
rem Windows Task Scheduler runs this once a month, windows\schedule_retrain.bat is what sets that up.
rem It runs training\monthly_retrain.py and saves everything it prints to data\monthly_retrain.log,
rem since nobody is watching the window at 3am. The short version of each run goes in results\retrain_history.txt.
setlocal
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" exit /b 1
if not exist data mkdir data
echo ===== %date% %time% =====>> data\monthly_retrain.log
"%PY%" training\monthly_retrain.py GMMark12 >> data\monthly_retrain.log 2>&1
