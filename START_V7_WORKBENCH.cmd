@echo off
setlocal
cd /d "%~dp0"
set PYTHONHOME=
set PYTHONNOUSERSITE=1
set MUJOCO_GL=
if exist ".venv\Scripts\python.exe" (
  set "PATH=%~dp0.venv\Scripts;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
  ".venv\Scripts\python.exe" scripts\run_v7_workbench.py --view --duration 180 %*
) else (
  echo Run SETUP_V7.cmd first.
)
pause
