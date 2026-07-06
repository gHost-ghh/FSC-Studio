param(
    [string]$Version = "1.24.4",
    [string]$Destination = "D:\FSC\.deps\onnxruntime-directml-1.24.4-nuget"
)

$ErrorActionPreference = "Stop"

$packageDir = "D:\FSC\.deps\nuget"
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
$packagePath = Join-Path $packageDir "Microsoft.ML.OnnxRuntime.DirectML.$Version.nupkg"
$zipPath = Join-Path $packageDir "Microsoft.ML.OnnxRuntime.DirectML.$Version.zip"
$uri = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$Version"

if (-not (Test-Path $packagePath)) {
    Invoke-WebRequest -Uri $uri -OutFile $packagePath
}

Copy-Item -LiteralPath $packagePath -Destination $zipPath -Force
if (Test-Path $Destination) {
    Remove-Item -LiteralPath $Destination -Recurse -Force
}
Expand-Archive -LiteralPath $zipPath -DestinationPath $Destination

$header = Join-Path $Destination "build\native\include\dml_provider_factory.h"
$runtime = Join-Path $Destination "runtimes\win-x64\native\onnxruntime.dll"
$library = Join-Path $Destination "runtimes\win-x64\native\onnxruntime.lib"
if (-not (Test-Path $header) -or -not (Test-Path $runtime) -or -not (Test-Path $library)) {
    throw "DirectML package was downloaded but required native files were not found."
}

Write-Host "DirectML ONNX Runtime extracted to $Destination"
