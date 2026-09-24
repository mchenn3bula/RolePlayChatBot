param(
    [ValidateSet('Train','Smoke','Resume')][string]$Mode = 'Train',
    [string]$RunName = ('ministral-d1-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$Checkpoint = ''
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid run name.' }
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Resume') {
        if ($Checkpoint -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid checkpoint name.' }
        & wsl -d Ubuntu-24.04 -- bash ./run_dpo.sh $RunName $Mode $Checkpoint
    } else {
        & wsl -d Ubuntu-24.04 -- bash ./run_dpo.sh $RunName $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "DPO exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
