param(
    [ValidateSet('sft','dpo')][string]$Arm = 'sft',
    [ValidateSet('Train','Smoke','Resume')][string]$Mode = 'Train',
    [string]$RunName = ('ministral-v2-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$Checkpoint = ''
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid run name.' }
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Resume') {
        if ($Checkpoint -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid checkpoint.' }
        & wsl -d Ubuntu-24.04 -- bash ./run_bilingual_control.sh $Arm $RunName $Mode $Checkpoint
    } else {
        & wsl -d Ubuntu-24.04 -- bash ./run_bilingual_control.sh $Arm $RunName $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "Control failed with code $LASTEXITCODE" }
} finally { Pop-Location }
