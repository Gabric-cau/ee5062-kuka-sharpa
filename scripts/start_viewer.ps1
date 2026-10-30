param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ViewerArguments
)

$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $Project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Missing project Python: $ProjectPython"
}
Set-Location -LiteralPath $Project
$BasePython = & $ProjectPython -c 'import sys; print(sys._base_executable)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve base Python.' }
$BaseDirectory = Split-Path -Parent $BasePython
$env:PATH = "$Project\.venv\Scripts;$BaseDirectory;$BaseDirectory\Scripts;$env:SystemRoot\System32;$env:SystemRoot"
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PIP_NO_CACHE_DIR = '1'
$env:MPLCONFIGDIR = Join-Path $Project 'results\local_cache\matplotlib'
$env:XDG_CACHE_HOME = Join-Path $Project 'results\local_cache'
New-Item -ItemType Directory -Path $env:MPLCONFIGDIR -Force | Out-Null

# A hardware report belongs to this process launch, not to a past CUDA check.
$LaunchId = (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$HardwarePath = Join-Path $Project "results\launcher_checks\$LaunchId.json"
& $ProjectPython scripts\check_hardware.py --output $HardwarePath
if ($LASTEXITCODE -ne 0) { throw "OpenGL check failed. Read $HardwarePath" }
$Hardware = Get-Content -LiteralPath $HardwarePath -Raw | ConvertFrom-Json
Write-Host "Actual OpenGL: $($Hardware.render.opengl_vendor) | $($Hardware.render.opengl_renderer)"
if (-not $Hardware.render.nvidia_rtx2060_verified) {
    Write-Warning 'This launch is not verified on RTX 2060. See DELL_RUNBOOK.md GPU selection.'
}
& $ProjectPython scripts\view_rollout.py @ViewerArguments
exit $LASTEXITCODE
