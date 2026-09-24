$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & wsl -d Ubuntu-24.04 -- bash ./run_response_refinement.sh
    if ($LASTEXITCODE -ne 0) { throw "Response-refinement comparison failed: $LASTEXITCODE" }
} finally { Pop-Location }
