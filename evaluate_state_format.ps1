$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & wsl -d Ubuntu-24.04 -- bash ./run_state_format.sh
    if ($LASTEXITCODE -ne 0) { throw "State comparison failed: $LASTEXITCODE" }
} finally { Pop-Location }
