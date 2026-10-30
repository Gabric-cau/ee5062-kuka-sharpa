$ErrorActionPreference = 'Stop'
$Desktop = [Environment]::GetFolderPath('Desktop')
$Launcher = Join-Path $PSScriptRoot 'start_dell_operator_recording.ps1'
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) { throw 'Missing recording launcher' }
$Shell = New-Object -ComObject WScript.Shell
$Link = Join-Path $Desktop 'EE5062 Record and Verify 20261022.lnk'
if (Test-Path -LiteralPath $Link) { throw 'Recording shortcut already exists' }
$Shortcut = $Shell.CreateShortcut($Link)
$Shortcut.TargetPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$Shortcut.Arguments = '-NoProfile -NoExit -ExecutionPolicy Bypass -File "' + $Launcher + '"'
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = 'EE5062 simulated robot recording and independent replay audit'
$Shortcut.Save()
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'recordings') -Force | Out-Null
Get-Item -LiteralPath $Link | Select-Object FullName, Length | ConvertTo-Json -Compress
