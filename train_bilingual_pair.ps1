$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'train_bilingual_control.ps1') -Arm sft -Mode Train -RunName 'ministral-v2-csft-v1'
& (Join-Path $PSScriptRoot 'train_bilingual_control.ps1') -Arm dpo -Mode Train -RunName 'ministral-v2-dpo-v1'
Write-Host 'Both matched controls completed. Checkpoint directories are preserved.'
