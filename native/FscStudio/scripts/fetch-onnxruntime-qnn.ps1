param(
    [string]$Version = "1.24.4",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$dependencyRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot ".deps"))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $dependencyRoot "onnxruntime-qnn-$Version-nuget"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $resolvedOutput.StartsWith($dependencyRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must stay under $dependencyRoot"
}

$packageId = "microsoft.ml.onnxruntime.qnn"
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "fsc-onnxruntime-qnn-$Version"
$packagePath = "$tempRoot.zip"
$url = "https://api.nuget.org/v3-flatcontainer/$packageId/$Version/$packageId.$Version.nupkg"

if (Test-Path $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

Write-Host "Downloading $url"
& curl.exe --location --fail --retry 3 --continue-at - --output $packagePath $url
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download ONNX Runtime QNN package (curl exit code $LASTEXITCODE)."
}

Expand-Archive -LiteralPath $packagePath -DestinationPath $tempRoot -Force
$runtimeDirectory = Join-Path $tempRoot "runtimes\win-arm64\native"
foreach ($required in @("onnxruntime.dll", "onnxruntime.lib")) {
    if (-not (Test-Path (Join-Path $runtimeDirectory $required))) {
        throw "QNN package is missing runtimes\win-arm64\native\$required"
    }
}
if (-not (Get-ChildItem -LiteralPath $runtimeDirectory -Filter "Qnn*.dll" -File -ErrorAction SilentlyContinue)) {
    throw "QNN package is missing Qualcomm backend DLLs under runtimes\win-arm64\native."
}

if (Test-Path $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}
Move-Item -LiteralPath $tempRoot -Destination $resolvedOutput

Write-Host "ONNXRUNTIME_ROOT=$resolvedOutput"
Write-Host "Configure the ARM64 preset with this root to enable QNN NPU/GPU modes."
