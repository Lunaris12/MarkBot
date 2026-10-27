@echo off
rem Sets up Windows Task Scheduler to retrain the bot on my newest games on the 1st of every month at 3am.
rem If my PC is off or asleep then, it runs as soon as the PC is back on instead of skipping that month.
rem I only need to run this once, and running it again just replaces the old schedule.
rem To turn it off again, run schtasks /delete /tn "MarkBot monthly retrain" /f
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" goto :no_setup

rem The task itself is described in windows\monthly_retrain_task.xml. PowerShell fills in where this repo is
rem and registers it, replacing the old one if there is one.
powershell -NoProfile -Command "$x = (Get-Content -Raw 'windows\monthly_retrain_task.xml').Replace('REPO_FOLDER', (Get-Location).Path); Register-ScheduledTask -TaskName 'MarkBot monthly retrain' -Xml $x -User $env:USERNAME -Force | Out-Null" || goto :failed

if exist "deploy\server.txt" (
  echo Checking that this PC can reach the server. gcloud might ask a couple of questions the first time.
  ".venv\Scripts\python.exe" training\monthly_retrain.py GMMark12 --check-server || goto :failed
) else (
  echo There's no deploy\server.txt yet, so new models will only be saved on this PC.
)

echo.
echo Done. The bot retrains on the 1st of every month at 3am, or as soon as the PC is on after that.
echo Each run is logged in data\monthly_retrain.log and results\retrain_history.txt.
pause
exit /b 0

:no_setup
echo Couldn't find .venv, run windows\setup.bat first.
pause
exit /b 1

:failed
echo.
echo Something went wrong, check the messages above.
pause
exit /b 1
