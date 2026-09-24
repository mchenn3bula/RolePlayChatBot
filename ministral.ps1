param(
    [ValidateSet('Chat', 'Evaluate')][string]$Mode = 'Chat',
    [ValidateSet('P0', 'P1')][string]$Profile = 'P1',
    [ValidateSet('en', 'fr')][string]$Language = 'en',
    [string]$StateFile = '',
    [string]$Adapter = '',
    [string]$RunName = ('ministral-' + $Profile.ToLowerInvariant() + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)
$ErrorActionPreference = 'Stop'
# Resolve a custom path against the caller's directory before changing directory.
$LinuxState = ''
if ($StateFile) {
    if ($Mode -ne 'Chat' -or $Profile -ne 'P1') { throw 'StateFile is supported by P1 Chat only.' }
    $ResolvedState = (Resolve-Path -LiteralPath $StateFile).Path
    $ConvertedState = & wsl -d Ubuntu-24.04 -- wslpath -a -u ($ResolvedState.Replace('\', '/'))
    if ($LASTEXITCODE -ne 0 -or -not $ConvertedState) { throw 'Could not convert state path for WSL.' }
    $LinuxState = ($ConvertedState | Out-String).Trim()
}
$LinuxAdapter = ''
if ($Adapter) {
    $ResolvedAdapter = (Resolve-Path -LiteralPath $Adapter).Path
    $ConvertedAdapter = & wsl -d Ubuntu-24.04 -- wslpath -a -u ($ResolvedAdapter.Replace('\', '/'))
    if ($LASTEXITCODE -ne 0 -or -not $ConvertedAdapter) { throw 'Could not convert adapter path for WSL.' }
    $LinuxAdapter = ($ConvertedAdapter | Out-String).Trim()
}
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Evaluate') {
        if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid run name.' }
        $WslArgs = @('-d', 'Ubuntu-24.04', '--', 'bash', './run_ministral.sh', $RunName, $Profile)
        if ($LinuxAdapter) { $WslArgs += $LinuxAdapter }
    } else {
        $WslArgs = @('-d', 'Ubuntu-24.04', '--', 'bash', './chat_ministral.sh', $Language, $Profile)
        if ($LinuxState) { $WslArgs += @('--state-file', $LinuxState) }
        if ($LinuxAdapter) { $WslArgs += @('--adapter', $LinuxAdapter) }
    }
    & wsl @WslArgs
    if ($LASTEXITCODE -ne 0) { throw "Ministral exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
