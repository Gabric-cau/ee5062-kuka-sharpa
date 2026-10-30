@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
set PYTHONNOUSERSITE=1
set PIP_NO_CACHE_DIR=1
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv --without-pip .venv
  ) else (
    py -3 -m venv --without-pip .venv
  )
)
if not exist ".venv\Scripts\python.exe" goto failed
set "PATH=%~dp0.venv\Scripts;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
".venv\Scripts\python.exe" -m ensurepip --upgrade --default-pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import tkinter, mujoco, numpy, PIL, imageio; print('V7 simulation environment ready')"
if errorlevel 1 goto failed
echo Run START_V7_WORKBENCH.cmd for interactive simulation.
echo Run START_V7_SORTING.cmd for the automatic three-tool task.
pause
exit /b 0
:failed
echo Setup failed. Keep this console output for diagnosis.
pause
exit /b 1
