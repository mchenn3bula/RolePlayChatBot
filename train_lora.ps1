param(
    [ValidateSet('Train', 'Smoke', 'Resume')][string]$Mode = 'Train',
    [string]$RunName = ('ministral-s1-lora-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$Checkpoint = ''
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid run name.' }
if ($Mode -eq 'Resume' -and $Checkpoint -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw 'Resume requires a checkpoint directory name within this run.'
}
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Resume') {
        & wsl -d Ubuntu-24.04 -- bash ./run_lora.sh $RunName $Mode $Checkpoint
    } else {
        & wsl -d Ubuntu-24.04 -- bash ./run_lora.sh $RunName $Mode
    }
    if ($LASTEXITCODE -ne 0) { throw "LoRA $Mode exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
