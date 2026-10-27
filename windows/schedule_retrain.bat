@echo off
rem Sets up Windows Task Scheduler to retrain the bot on my newest games on the 1st of every month at 3am.
rem If my PC is off or asleep then, it runs as soon as the PC is back on instead of skipping that month.
rem I only need to run this once, and running it again just replaces the old schedule.
rem To turn it off again, run schtasks /delete /tn "MarkBot monthly retrain" /f
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" goto :no_setup

schtasks /create /tn "MarkBot monthly retrain" /tr "\"%CD%\windows\monthly_retrain.bat\"" /sc monthly /d 1 /st 03:00 /f >nul || goto :failed
rem schtasks can't turn on "run as soon as possible after a missed start" by itself, so PowerShell does that part.
rem It also lets it run on a laptop that's unplugged.
powershell -NoProfile -Command "$t = Get-ScheduledTask -TaskName 'MarkBot monthly retrain'; $t.Settings.StartWhenAvailable = $true; $t.Settings.DisallowStartIfOnBatteries = $false; $t.Settings.StopIfGoingOnBatteries = $false; Set-ScheduledTask -InputObject $t | Out-Null" || goto :failed

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
