@echo off
rem Retrains the bot from scratch on my latest chess.com games, running these steps in order
rem Fetch games, prep data, fine-tune, build the opening book, export to ONNX, evaluate.
rem It takes about 30-45 minutes on a regular CPU, almost all of it in the fine-tuning step.
rem Usage: windows\retrain.bat              (defaults to GMMark12)
rem        windows\retrain.bat SomeoneElse  (any chess.com username)
setlocal
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" goto :no_setup

rem USERNAME is already taken by Windows, so the chess.com name goes in PLAYER.
set PLAYER=%~1
if "%PLAYER%"=="" set PLAYER=GMMark12

echo [1/6] Fetching every chess.com game for %PLAYER%...
"%PY%" training\fetch_games.py %PLAYER% || goto :failed
echo [2/6] Turning the games into training data...
"%PY%" training\prep_data.py %PLAYER% || goto :failed
echo [3/6] Fine-tuning Maia-2. This can take about 40 minutes on a CPU...
"%PY%" training\finetune.py || goto :failed
echo [4/6] Building the opening book from every game...
"%PY%" training\build_book.py %PLAYER% --holdout 0 || goto :failed
echo [5/6] Exporting the model to ONNX...
"%PY%" training\export_onnx.py || goto :failed
echo [6/6] Checking how well it matches the real games...
"%PY%" training\evaluate.py %PLAYER% || goto :failed

echo.
echo All done. The new model and book are in the models folder.
echo markbot.exe reads them when it starts, so just restart the engine in your chess app. No rebuild needed.
pause
exit /b 0

:no_setup
echo .venv was not found. Run windows\setup.bat first.
pause
exit /b 1

:failed
echo.
echo That step failed, see the messages above. The steps after it were not run.
pause
exit /b 1
