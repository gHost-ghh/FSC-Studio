param(
    [string]$Version = "1.27.0",
    [ValidateSet("cpu", "directml", "cuda")]
    [string]$Flavor = "cpu",
    [ValidateSet("12", "13")]
    [string]$CudaMajor = "13",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $folder = if ($Flavor -eq "cpu") { "onnxruntime" } else { "onnxruntime-$Flavor-$Version" }
    $OutputRoot = Join-Path $workspaceRoot ".deps\$folder"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
$dependencyRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot ".deps"))
if (-not $resolvedOutput.StartsWith($dependencyRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must stay under $dependencyRoot"
}
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "fsc-onnxruntime-$Version-$Flavor"
$zipPath = "$tempRoot.zip"

switch ($Flavor) {
    "directml" { $asset = "onnxruntime-win-x64-directml-$Version.zip" }
    "cuda" { $asset = "onnxruntime-win-x64-gpu_cuda$CudaMajor-$Version.zip" }
    default { $asset = "onnxruntime-win-x64-$Version.zip" }
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
& curl.exe --location --fail --retry 3 --continue-at - --output $zipPath $url
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download ONNX Runtime archive (curl exit code $LASTEXITCODE)."
}
Expand-Archive -LiteralPath $zipPath -DestinationPath $tempRoot -Force

$expanded = Get-ChildItem -LiteralPath $tempRoot -Directory | Select-Object -First 1
if ($null -eq $expanded) {
    throw "ONNX Runtime archive did not contain a root directory."
}
Move-Item -LiteralPath $expanded.FullName -Destination $resolvedOutput

Write-Host "ONNXRUNTIME_ROOT=$resolvedOutput"
Write-Host "Configure with:"
Write-Host "  cmake --preset msvc-release -DONNXRUNTIME_ROOT=`"$resolvedOutput`""
