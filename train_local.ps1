param(
    [ValidateRange(1, 1000)][int]$Epochs = 1,
    [ValidateRange(1, 32)][int]$MicroBatch = 1,
    [ValidateRange(1, 1024)][int]$GradAccum = 32,
    [string]$OutputDir = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'The project virtual environment is missing. Follow LOCAL_TRAINING.md.'
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectDirectory ('checkpoints/local-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}

Push-Location -LiteralPath $projectDirectory
try {
    & $pythonExecutable -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable; install requirements-cuda.txt'; print(torch.cuda.get_device_name())"
    if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
    foreach ($splitName in @('train', 'validation')) {
        $metadataPath = Join-Path $projectDirectory "data/bluemoon_${splitName}_tok_ds/roleplay_format.json"
        if (-not (Test-Path -LiteralPath $metadataPath)) {
            throw "Prepared $splitName dataset is missing. Follow LOCAL_TRAINING.md."
        }
    }
    if ($CheckOnly) {
        Write-Output 'Local training preflight passed. GPU and prepared datasets are available.'
        return
    }
    & $pythonExecutable -u train.py `
        --config configs/small.json `
        --data-dir data/bluemoon_train_tok_ds `
        --validation-dir data/bluemoon_validation_tok_ds `
        --output-dir $OutputDir `
        --device cuda `
        --micro-batch $MicroBatch `
        --grad-accum $GradAccum `
        --epochs $Epochs `
        --log-every 10
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
