param([string]$Tag = (Get-Date -Format 'yyyyMMdd_HHmmss'))
$ErrorActionPreference = 'Stop'
if ($Tag -notmatch '^[a-zA-Z0-9_]+$') { throw 'Tag must be an identifier' }
$Project = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Project '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Run SETUP_V7.cmd first' }
$Output = Join-Path $Project "results\v7_windows_verify_$Tag"
if (Test-Path $Output) { throw 'Output already exists' }
New-Item -ItemType Directory -Path $Output | Out-Null
Set-Location $Project
$env:PYTHONPATH = Join-Path $Project 'src'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$Temporary = Join-Path $env:LOCALAPPDATA "Temp\ee5062_tests_$Tag"
if (Test-Path $Temporary) { throw 'Use a new tag for verification' }
New-Item -ItemType Directory -Path $Temporary | Out-Null
$env:TEMP = $Temporary
$env:TMP = $Temporary
Remove-Item Env:MUJOCO_GL -ErrorAction SilentlyContinue
Start-Transcript -Path (Join-Path $Output 'windows_verification.log') -NoClobber | Out-Null
try {
    & $Python -m pytest tests -q --color=no --basetemp (Join-Path $Temporary 'pytest') *> (Join-Path $Output 'pytest.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Software regression failed' }
    & $Python scripts\check_hardware.py --output (Join-Path $Output 'hardware.json')
    if ($LASTEXITCODE -ne 0) { throw 'Native OpenGL hardware probe failed' }
    $Hardware = Get-Content (Join-Path $Output 'hardware.json') -Raw | ConvertFrom-Json
    if (-not $Hardware.render.available -or $Hardware.render.opengl_renderer -notmatch 'NVIDIA') {
        throw 'An actual NVIDIA OpenGL context was not verified'
    }
    $Gui = Join-Path $Output 'gui'
    $Replay = Join-Path $Output 'replay'
    & $Python scripts\run_v7_workbench.py --view --automated-gui-check --duration 3 --fps 20 --seed 0 --output $Gui
    if ($LASTEXITCODE -ne 0) { throw 'Interactive GUI recording failed' }
    & $Python scripts\run_v7_workbench.py --replay (Join-Path $Gui 'action_log.json') --duration 3 --fps 20 --seed 0 --output $Replay
    if ($LASTEXITCODE -ne 0) { throw 'Physical action replay failed' }
    & $Python scripts\verify_v7_gui_replay.py --gui $Gui --replay $Replay --output (Join-Path $Output 'gui_replay_receipt.json')
    if ($LASTEXITCODE -ne 0) { throw 'GUI coverage, physical equivalence or video checks failed' }
    & $Python scripts\run_v7_tool_sorting.py --seed 0 --video --output (Join-Path $Output 'sorting')
    if ($LASTEXITCODE -ne 0) { throw 'Continuous physical sorting failed' }
    @{ status = 'accepted_windows_simulation_and_automated_gui'; human_operator = $false;
       completed_utc = (Get-Date).ToUniversalTime().ToString('o'); project = $Project;
       hardware_renderer = $Hardware.render.opengl_renderer;
       gui_receipt = 'gui_replay_receipt.json'; sorting_result = 'sorting\summary.json'
    } | ConvertTo-Json | Set-Content (Join-Path $Output 'completion.json') -Encoding UTF8
    Stop-Transcript | Out-Null
    exit 0
} catch {
    $_ | Out-String | Write-Output
    Stop-Transcript | Out-Null
    exit 1
}
