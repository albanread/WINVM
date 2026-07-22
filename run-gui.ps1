# Launch the WINVM GUI — the Strongtalk HTML programming environment hosted in
# a native Win32 + WebView2 shell (gui/, MIGRATION.md §6 / M6). Builds the
# winvm-gui binary, then runs it. The macOS twin is run-gui.sh.
#
# Runs from the repo root so the VM finds world/ (and, when MACVM_IMAGE_PATH is
# set, the versioned SQLite image the class browser points at). The language
# thread runs the real embedded VM (src/embed.rs, VmHandle) on its own thread,
# so a long doit never freezes the window.
#
# Usage:
#   .\run-gui.ps1                       # release build (default), JIT ON
#   .\run-gui.ps1 -Debug                # unoptimized build — ONLY for chasing a
#                                       # crash; the JIT, GC, and runtime helpers
#                                       # all run dramatically slower
#   $env:MACVM_JIT="off";  .\run-gui.ps1   # force the pure interpreter
#   $env:MACVM_JIT="threshold=1"; .\run-gui.ps1   # compile aggressively
#   $env:MACVM_IMAGE_PATH="world\image.sqlite3"; .\run-gui.ps1
#
# Debugging the page itself (DevTools against the live web view):
#   $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS="--remote-debugging-port=9222"
# then open http://127.0.0.1:9222 in Edge. This is how the shell's JS<->Rust
# bridge was verified end to end.
#
# Requires the WebView2 Runtime, which ships with Windows 11; on older Windows
# install the Evergreen runtime (MIGRATION.md §5, risk table).
[CmdletBinding()]
param(
    [switch]$Debug,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = 'Stop'

# Repo root — so relative paths (world/, the image) resolve regardless of where
# this script is invoked from.
Set-Location -LiteralPath $PSScriptRoot

if ($Debug) {
    $profileArgs = @()
    $binDir = 'debug'
} else {
    $profileArgs = @('--release')
    $binDir = 'release'
}

Write-Host "> building winvm-gui ($binDir)..."
cargo build -p winvm-gui @profileArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$jit = if ($env:MACVM_JIT) { $env:MACVM_JIT } else { 'threshold=10 (GUI default)' }
Write-Host "> launching WINVM GUI  (MACVM_JIT=$jit)"

$exe = Join-Path $PSScriptRoot "target\$binDir\winvm-gui.exe"
& $exe @AppArgs
exit $LASTEXITCODE
