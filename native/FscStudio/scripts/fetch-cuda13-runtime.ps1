param(
    [string]$Python = "",
    [string]$OutputRoot = "",
    [string]$CublasVersion = "13.6.0.2",
    [string]$CufftVersion = "12.3.0.29",
    [string]$CudnnVersion = "9.23.2.1"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\..\.."))
$dependencyRoot = Join-Path $workspaceRoot ".deps"
if (-not $OutputRoot) { $OutputRoot = Join-Path $dependencyRoot "cuda13-redist" }
$outputFull = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $outputFull.StartsWith($dependencyRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must stay under $dependencyRoot"
}
if (-not $Python) {
    $candidate = Join-Path $workspaceRoot ".venv312\Scripts\python.exe"
    $Python = if (Test-Path $candidate) { $candidate } else { (Get-Command python.exe -ErrorAction Stop).Source }
}

$packages = @(
    "nvidia-cublas==$CublasVersion",
    "nvidia-cufft==$CufftVersion",
    "nvidia-cudnn-cu13==$CudnnVersion"
)
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "fsc-cuda13-redist"
if (Test-Path $tempRoot) { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

foreach ($package in $packages) {
    & $Python -m pip download --disable-pip-version-check --no-deps --only-binary=:all: --dest $tempRoot $package
    if ($LASTEXITCODE -ne 0) { throw "Failed to download CUDA redistributable package $package" }
}

if (Test-Path $outputFull) { Remove-Item -LiteralPath $outputFull -Recurse -Force }
$binRoot = Join-Path $outputFull "bin"
$licenseRoot = Join-Path $outputFull "licenses"
New-Item -ItemType Directory -Force -Path $binRoot, $licenseRoot | Out-Null

foreach ($wheel in Get-ChildItem -LiteralPath $tempRoot -Filter "*.whl") {
    $expanded = Join-Path $tempRoot $wheel.BaseName
    Expand-Archive -LiteralPath $wheel.FullName -DestinationPath $expanded -Force
    Get-ChildItem -LiteralPath $expanded -Recurse -Filter "*.dll" -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $binRoot -Force
    }
    $packageLicense = Join-Path $licenseRoot $wheel.BaseName
    $licenseFiles = Get-ChildItem -LiteralPath $expanded -Recurse -File |
        Where-Object { $_.Name -match '^(LICENSE|License|NOTICE|Notice|EULA)' }
    if ($licenseFiles) {
        New-Item -ItemType Directory -Force -Path $packageLicense | Out-Null
        $licenseFiles | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $packageLicense -Force }
    }
}

$requiredRuntimeFiles = @(
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cufft64_12.dll",
    "cudnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll"
)
foreach ($required in $requiredRuntimeFiles) {
    if (-not (Test-Path (Join-Path $binRoot $required))) { throw "CUDA runtime is missing $required" }
}

Write-Host "CUDA 13 redistributable runtime ready: $outputFull"
