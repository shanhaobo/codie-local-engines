# Build the local ASR sidecar (our codie-asr-local wrapper around faster-whisper)
# into a distributable zip on Windows. Unlike build-sidecar.ps1 (which bundles
# host_mcp into the app Resources), the ASR/TTS sidecars are delivered by RUNTIME
# DOWNLOAD: this script produces dist\codie-asr-local-windows-x64.zip + prints its
# SHA-256, for an operator to upload and paste into LocalEngineSpec
# (id codie-asr-local, win).
#
# The zip's root holds the launchable binary + _internal\, so the Bridge can
# extract it straight into <dataDir>\ai-tools\codie-asr-local\ and run
# codie-asr-local.exe there.
#
# Compatible with Windows PowerShell 5.1 AND PowerShell 7+.
#
# Usage:  .\scripts\build-asr-sidecar.ps1

Set-StrictMode -Version Latest

function Invoke-Checked {
    if ($args.Count -lt 1) { throw 'Invoke-Checked: no command given' }
    $exe = $args[0]
    $argv = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
    & $exe @argv
    if ($LASTEXITCODE -ne 0) {
        throw "command failed (exit $LASTEXITCODE): $($args -join ' ')"
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EngineDir = Resolve-Path (Join-Path $ScriptDir '..\local_engines\asr')
$Id = 'codie-asr-local'
$Arch = 'windows-x64'
Set-Location $EngineDir
Write-Host "[build-asr-sidecar] working in $EngineDir"

$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) { $pyCmd = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $pyCmd) { throw 'python not found on PATH' }
Write-Host "[build-asr-sidecar] using python at $($pyCmd.Source)"

if (-not (Test-Path '.venv')) {
    Write-Host '[build-asr-sidecar] creating .venv'
    Invoke-Checked $pyCmd.Source -m venv .venv
}
$activate = Join-Path (Resolve-Path '.venv') 'Scripts\Activate.ps1'
if (-not (Test-Path $activate)) {
    $activate = Join-Path (Resolve-Path '.venv') 'bin/Activate.ps1'
}
. $activate
Write-Host "[build-asr-sidecar] venv activated ($activate)"

Write-Host '[build-asr-sidecar] pip install -e .[dev,win]'
Invoke-Checked python -m pip install -e '.[dev,win]'

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist  }

Write-Host '[build-asr-sidecar] running pyinstaller'
Invoke-Checked python -m PyInstaller --clean --noconfirm codie_asr_local.spec

$built = Join-Path 'dist' $Id
$exe = Join-Path $built "$Id.exe"
if (-not (Test-Path $exe)) {
    throw "pyinstaller did not produce $exe"
}

# Zip the CONTENTS of dist\<id>\ (binary + _internal\) so extraction lands the
# binary directly at <dataDir>\ai-tools\<id>\<id>.exe (no double nesting).
$zip = Join-Path 'dist' "$Id-$Arch.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $built '*') -DestinationPath $zip
$sha = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
$full = (Resolve-Path $zip).Path

Write-Host ''
Write-Host '[build-asr-sidecar] done.'
Write-Host "  artifact : $full"
Write-Host "  sha256   : $sha"
Write-Host ''
Write-Host '  Next: upload this zip to your release host, then set in'
Write-Host '  lib/models/local_engine_spec.dart (id codie-asr-local, EnginePlatform.win):'
Write-Host "    url:    '<download URL of the uploaded zip>'"
Write-Host "    sha256: '$sha'"
