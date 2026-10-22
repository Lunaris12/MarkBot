@echo off
rem This puts MarkBot online on Lichess through lichess-bot.
rem The first time you run it, it downloads lichess-bot into a lichess-bot folder in this repo, makes
rem lichess-bot\config.yml from deploy\lichess-bot-config.yml and then stops so you can paste the token in.
rem Run windows\lichess_bot.bat to go online, and close the window whenever you want it offline.
rem Run windows\lichess_bot.bat -u just once at the very start, it turns the account into a bot account for good.
setlocal
cd /d "%~dp0.."
set PY=%CD%\.venv\Scripts\python.exe
if not exist "%PY%" goto :no_setup
if not exist "engine\markbot.exe" goto :no_exe

if not exist "lichess-bot\lichess-bot.py" (
  echo Downloading lichess-bot...
  git clone --quiet --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git lichess-bot || goto :no_git
)
if not exist "lichess-bot\config.yml" goto :make_config
findstr /C:"PASTE_YOUR_TOKEN_HERE" "lichess-bot\config.yml" >nul && goto :no_token

"%PY%" -m pip install --quiet -r lichess-bot\requirements.txt || goto :failed
if /i "%~1"=="-u" goto :confirm_upgrade

:run
cd lichess-bot
echo Starting MarkBot on Lichess, close this window whenever you want it offline.
"%PY%" lichess-bot.py %*
pause
exit /b 0

:confirm_upgrade
echo.
echo This turns the Lichess account in lichess-bot\config.yml into a bot account FOR GOOD.
echo There's no undoing it and the account can't have played any games yet, so never use your main account.
set /p ANSWER=Type YES if you're sure and press Enter 
if /i not "%ANSWER%"=="YES" goto :cancelled
goto :run

:make_config
copy /y "deploy\lichess-bot-config.yml" "lichess-bot\config.yml" >nul
echo Made lichess-bot\config.yml for you. Paste the bot account's token into the first line, save it and run this again.
start notepad "lichess-bot\config.yml"
pause
exit /b 0

:no_token
echo lichess-bot\config.yml still says PASTE_YOUR_TOKEN_HERE, so paste the bot account's token in, save it and try again.
pause
exit /b 1

:cancelled
echo Cancelled, nothing was changed.
pause
exit /b 0

:no_setup
echo Couldn't find .venv, run windows\setup.bat first.
pause
exit /b 1

:no_exe
echo Couldn't find engine\markbot.exe, run engine\build_exe.bat first.
pause
exit /b 1

:no_git
echo Couldn't download lichess-bot. Git might not be installed, you can get it from git-scm.com and then run this again.
pause
exit /b 1

:failed
echo.
echo Something went wrong, check the messages above.
pause
exit /b 1
