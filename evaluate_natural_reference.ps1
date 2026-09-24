$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & wsl -d Ubuntu-24.04 -- bash ./run_natural_reference.sh
    if ($LASTEXITCODE -ne 0) { throw "Natural-reference comparison failed: $LASTEXITCODE" }
} finally { Pop-Location }
