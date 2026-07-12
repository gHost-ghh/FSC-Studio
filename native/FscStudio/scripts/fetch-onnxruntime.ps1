param(
    [string]$Version = "1.27.0",
    [ValidateSet("cpu", "directml")]
    [string]$Flavor = "cpu",
    [string]$OutputRoot = "$PSScriptRoot\..\..\..\.deps\onnxruntime"
)

$ErrorActionPreference = "Stop"

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "fsc-onnxruntime-$Version-$Flavor"
$zipPath = "$tempRoot.zip"

if ($Flavor -eq "directml") {
    $asset = "onnxruntime-win-x64-directml-$Version.zip"
} else {
    $asset = "onnxruntime-win-x64-$Version.zip"
}

$url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$asset"

if (Test-Path $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}
if (Test-Path $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $resolvedOutput) -Force | Out-Null

Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zipPath
Expand-Archive -LiteralPath $zipPath -DestinationPath $tempRoot -Force

$expanded = Get-ChildItem -LiteralPath $tempRoot -Directory | Select-Object -First 1
if ($null -eq $expanded) {
    throw "ONNX Runtime archive did not contain a root directory."
}
Move-Item -LiteralPath $expanded.FullName -Destination $resolvedOutput

Write-Host "ONNXRUNTIME_ROOT=$resolvedOutput"
Write-Host "Configure with:"
Write-Host "  cmake --preset msvc-release -DONNXRUNTIME_ROOT=`"$resolvedOutput`""
