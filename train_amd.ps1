param(
    [ValidateSet('Test', 'Smoke', 'Benchmark', 'Train', 'Resume')]
    [string]$Mode = 'Smoke',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$RunName,
    [ValidateSet('Small', 'Overnight', 'Modern')]
    [string]$Profile = 'Small',
    [string]$Distribution = 'Ubuntu-24.04'
)
$ErrorActionPreference = 'Stop'
if ($Mode -in @('Train', 'Resume') -and -not $RunName) {
    throw 'Specify -RunName. Each new training run needs a distinct directory.'
}
if (-not $RunName) {
    $RunName = 'rx7900xtx-' + $Profile.ToLowerInvariant() + '-' + $Mode.ToLowerInvariant() + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
}
$linuxProject = (& wsl -d $Distribution -u n3bula --exec wslpath -a $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the project path inside WSL.' }
& wsl -d $Distribution -u n3bula --cd $linuxProject --exec bash './run_amd.sh' $Mode $RunName $Profile
exit $LASTEXITCODE
