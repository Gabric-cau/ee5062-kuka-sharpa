param([double]$Duration = 60, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
if ($Duration -le 0 -or $Duration -gt 180) { throw 'Duration must be in (0, 180] simulation seconds' }
$Project = Join-Path ([Environment]::GetFolderPath('Desktop')) 'kuka_sharpa_v7_20261014_r3'
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$Workbench = Join-Path $Project 'scripts\run_v7_workbench.py'
$Verifier = Join-Path $PSScriptRoot 'verify_v7_operator_recording.py'
foreach ($Required in @($Python, $Workbench, $Verifier)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) { throw "Missing file: $Required" }
}
$env:PYTHONHOME = $null
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $Project 'src'
$env:MUJOCO_GL = $null
$env:PATH = (Split-Path $Python -Parent) + ";$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\Wbem;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
Set-Location -LiteralPath $Project
if ($CheckOnly) {
    & $Python -c 'import sys, tkinter, mujoco, numpy, imageio; print(sys.version)'
    if ($LASTEXITCODE -ne 0) { throw 'Recording dependency check failed' }
    & $Python $Verifier --help
    if ($LASTEXITCODE -ne 0) { throw 'Verifier dependency check failed' }
    exit 0
}
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Session = Join-Path $PSScriptRoot "recordings\take_$Stamp"
if (Test-Path -LiteralPath $Session) { throw 'Recording directory already exists' }
New-Item -ItemType Directory -Path $Session | Out-Null
$Gui = Join-Path $Session 'operator'
$Replay = Join-Path $Session 'replay'
$Audit = Join-Path $Session 'audit.json'
Start-Transcript -Path (Join-Path $Session 'recording.log') -NoClobber | Out-Null
try {
    Write-Output "Recording: $Gui"
    & $Python $Workbench --view --duration $Duration --fps 20 --seed 0 --output $Gui
    if ($LASTEXITCODE -ne 0) { throw 'Interactive recording failed' }
    $Recorded = Get-Content -LiteralPath (Join-Path $Gui 'metadata.json') -Raw | ConvertFrom-Json
    $ReplayDuration = ([double]$Recorded.duration_s).ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    & $Python $Workbench --replay (Join-Path $Gui 'action_log.json') --duration $ReplayDuration --fps 20 --seed 0 --output $Replay
    if ($LASTEXITCODE -ne 0) { throw 'Replay failed' }
    & $Python $Verifier --gui $Gui --replay $Replay --output $Audit
    if ($LASTEXITCODE -ne 0) { throw 'Independent recording audit failed' }
    @{ status = 'recording_and_replay_consistent'; task_success = $null;
       human_identity_or_input_technically_verified = $false;
       completed_utc = (Get-Date).ToUniversalTime().ToString('o');
       gui = $Gui; replay = $Replay; audit = $Audit;
       scope = 'Input provenance and physics replay consistency; operator identity and task outcome are not inferred'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Session 'completion.json') -Encoding UTF8
    Write-Output "Recording audit: $Audit"
} catch {
    @{ status = 'recording_or_audit_failed'; error = $_.Exception.Message;
       completed_utc = (Get-Date).ToUniversalTime().ToString('o');
       human_identity_or_input_technically_verified = $false
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Session 'failure.json') -Encoding UTF8
    throw
} finally {
    Stop-Transcript | Out-Null
}
