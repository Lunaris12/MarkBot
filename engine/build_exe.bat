@echo off
rem Builds engine\markbot.exe, a single file that chess apps like En Croissant can load directly.
rem It uses its own small Python environment (.venv-engine) with only what the bot needs.
rem If PyTorch is around, PyInstaller drags it into the .exe and it gets huge.
cd /d "%~dp0.."
if not exist ".venv-engine\Scripts\python.exe" (
  echo Setting up a clean environment for the build, one time only...
  python -m venv .venv-engine || (echo Python not found. Install Python 3.10+ from python.org. & pause & exit /b 1)
)
".venv-engine\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv-engine\Scripts\python.exe" -m pip install --quiet -r engine\requirements.txt pyinstaller || (echo Could not install the packages. & pause & exit /b 1)
".venv-engine\Scripts\python.exe" -m PyInstaller --onefile --name markbot --distpath engine --workpath build\pyinstaller --specpath build\pyinstaller engine\markbot.py || (echo The build failed, see the messages above. & pause & exit /b 1)
echo.
echo Built engine\markbot.exe. Point your chess app at that file.
echo It reads mark_model.onnx, moves.json and book.json from the models folder.
pause
