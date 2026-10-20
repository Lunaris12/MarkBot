@echo off
rem One-time setup on Windows. Makes a .venv folder in the repo and installs everything training needs.
rem After this, windows\retrain.bat runs the whole pipeline and engine\build_exe.bat builds the bot.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo Creating .venv ...
  python -m venv .venv || goto :no_python
)

echo Installing the training packages. PyTorch is big, so this can take a few minutes...
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install -r training\requirements.txt || goto :failed
rem maia2 pins old versions that have dont support Python 3.13 on Windows, so skip its pins.
"%PY%" -m pip install --no-deps maia2==0.10.0 || goto :failed
"%PY%" -m pip install -r engine\requirements.txt || goto :failed

rem Make sure everything actually imports before saying it worked.
"%PY%" -c "import chess, torch, maia2, onnxruntime; print('All packages import fine.')" || goto :failed

echo.
echo Setup done. In VS Code, press Ctrl+Shift+P, choose "Python: Select Interpreter" and pick .venv.
echo Next: windows\retrain.bat trains the bot, then engine\build_exe.bat builds markbot.exe.
pause
exit /b 0

:no_python
echo Python was not found. Install Python 3.10 or newer from python.org, click "Add python.exe to PATH", then run this again.
pause
exit /b 1

:failed
echo.
echo Something went wrong, see the messages above.
pause
exit /b 1
