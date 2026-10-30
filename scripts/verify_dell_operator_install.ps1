$ErrorActionPreference = 'Stop'
$Desktop = [Environment]::GetFolderPath('Desktop')
$Project = Join-Path $Desktop 'kuka_sharpa_v7_20261014_r3'
$Link = Join-Path $Desktop 'EE5062 Record and Verify 20261022.lnk'
$Launcher = Join-Path $PSScriptRoot 'start_dell_operator_recording.ps1'
if (-not (Test-Path -LiteralPath $Link -PathType Leaf)) { throw 'Recording shortcut is missing' }
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($Link)
$ExpectedTarget = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$ExpectedArguments = '-NoProfile -NoExit -ExecutionPolicy Bypass -File "' + $Launcher + '"'
if ($Shortcut.TargetPath -ne $ExpectedTarget -or $Shortcut.Arguments -ne $ExpectedArguments -or
    $Shortcut.WorkingDirectory -ne $PSScriptRoot) { throw 'Saved shortcut differs from intended launcher' }
$Output = Join-Path $PSScriptRoot 'deployment_receipt.json'
if (Test-Path -LiteralPath $Output) { throw 'Deployment receipt already exists' }
& $ExpectedTarget -NoProfile -ExecutionPolicy Bypass -File $Launcher -CheckOnly
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
$Files = @('start_dell_operator_recording.ps1', 'install_dell_operator_shortcut.ps1',
    'verify_dell_operator_install.ps1', 'verify_v7_operator_recording.py', 'V7_OPERATOR_RECORDING_AUDIT.md')
$Hashes = @($Files | ForEach-Object {
    Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $_) -Algorithm SHA256 |
        Select-Object Path, Hash
})
$RuntimeHashes = @('v7_session.py', 'v7_task.py', 'v7_episode.py') | ForEach-Object {
    Get-FileHash -LiteralPath (Join-Path $Project "src\kuka_sharpa_grasp\$_") -Algorithm SHA256 |
        Select-Object Path, Hash
}
@{ format = 'ee5062_dell_operator_deployment_v1';
   measured_utc = (Get-Date).ToUniversalTime().ToString('o'); host = $env:COMPUTERNAME;
   project = $Project; helper = $PSScriptRoot; shortcut_file = $Link;
   shortcut_target = $Shortcut.TargetPath; shortcut_arguments = $Shortcut.Arguments;
   shortcut_working_directory = $Shortcut.WorkingDirectory;
   files = $Hashes; runtime_files = @($RuntimeHashes);
   video_controllers = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion);
   dependencies_checked = $true; shortcut_readback_checked = $true;
   gui_launched = $false; opengl_rendering_rechecked = $false;
   human_operator_recording_created = $false;
   scope = 'SSH, files, dependency imports and saved shortcut readback only. No new GUI operation or OpenGL rendering claim.'
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Output -Encoding UTF8
Get-Content -LiteralPath $Output -Raw
