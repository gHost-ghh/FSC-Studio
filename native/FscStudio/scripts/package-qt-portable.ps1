param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [ValidateSet("x64", "arm64")]
    [string]$Architecture = "x64",
    [ValidateSet("directml", "cuda", "qnn")]
    [string]$Accelerator = "directml",
    [string]$AppVersion = "0.2.0",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$ModelRoot = "",
    [string]$FaceMeshModel = "",
    [string]$CudaRuntimeRoot = "",
    [string]$VcRedistPath = "",
    [switch]$AllowOutsideOutput,
    [switch]$SkipSmoke,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "..\.."))
$depsRoot = Join-Path $workspaceRoot ".deps"

if ($Architecture -eq "arm64" -and $Accelerator -ne "qnn") {
    throw "The ARM64 release uses the QNN CPU/Adreno/HTP runtime. Choose -Accelerator qnn."
}
if ($Architecture -eq "x64" -and $Accelerator -eq "qnn") {
    throw "QNN packaging is available only for the native ARM64 release."
}
if (-not $ModelRoot) { $ModelRoot = Join-Path $workspaceRoot "model\insightface\models" }
if (-not $FaceMeshModel) { $FaceMeshModel = Join-Path $workspaceRoot "model\mediapipe\face_landmarks_detector.onnx" }

if (-not $BuildDir) {
    if ($Architecture -eq "arm64") {
        $BuildDir = Join-Path $projectRoot "out\build\arm64-qt-qnn"
    } elseif ($Accelerator -eq "cuda") {
        $BuildDir = Join-Path $projectRoot "out\build\msvc-vs-qt-camera-cuda-release"
    } else {
        $BuildDir = Join-Path $projectRoot "out\build\msvc-vs-qt-camera-dml-release"
    }
}
if (-not $OutputDir) {
    $name = "FSC-Studio-Windows-$Architecture-$($Accelerator.ToUpperInvariant())-$Configuration"
    $OutputDir = Join-Path $projectRoot "out\package\$name"
}

$packageRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "out\package"))
$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
$insidePackageRoot = $outputFull.Equals($packageRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $outputFull.StartsWith($packageRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
if (-not $AllowOutsideOutput -and -not $insidePackageRoot) {
    throw "Refusing to clean output outside $packageRoot."
}

$buildRoot = [System.IO.Path]::GetFullPath($BuildDir)
$binaryDir = if (Test-Path (Join-Path $buildRoot "$Configuration\FscStudioQt.exe")) {
    Join-Path $buildRoot $Configuration
} else {
    $buildRoot
}
$exePath = Join-Path $binaryDir "FscStudioQt.exe"
if (-not (Test-Path $exePath)) { throw "FscStudioQt.exe was not found under $binaryDir" }

function Get-PeMachine([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $reader = [System.IO.BinaryReader]::new($stream)
        $stream.Position = 0x3c
        $peOffset = $reader.ReadInt32()
        $stream.Position = $peOffset + 4
        return $reader.ReadUInt16()
    } finally {
        $stream.Dispose()
    }
}

$expectedMachine = if ($Architecture -eq "arm64") { 0xAA64 } else { 0x8664 }
if ((Get-PeMachine $exePath) -ne $expectedMachine) {
    throw "FscStudioQt.exe architecture does not match requested package architecture $Architecture."
}

$buffaloRoot = if ((Split-Path -Leaf ([System.IO.Path]::GetFullPath($ModelRoot))) -eq "buffalo_l") {
    [System.IO.Path]::GetFullPath($ModelRoot)
} else {
    Join-Path ([System.IO.Path]::GetFullPath($ModelRoot)) "buffalo_l"
}
foreach ($required in @(
    $FaceMeshModel,
    (Join-Path $buffaloRoot "det_10g.onnx"),
    (Join-Path $buffaloRoot "w600k_r50.onnx"),
    (Join-Path $buffaloRoot "2d106det.onnx"),
    (Join-Path $buffaloRoot "1k3d68.onnx"),
    (Join-Path $buffaloRoot "genderage.onnx"),
    (Join-Path $binaryDir "qt.conf"),
    (Join-Path $binaryDir "platforms\qwindows.dll"),
    (Join-Path $binaryDir "platforms\qminimal.dll"),
    (Join-Path $binaryDir "imageformats\qjpeg.dll"),
    (Join-Path $binaryDir "onnxruntime.dll")
)) {
    if (-not (Test-Path $required)) { throw "Required release file is missing: $required" }
}
if ($Architecture -eq "arm64") {
    foreach ($name in @("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "1k3d68.onnx")) {
        $path = Join-Path $buffaloRoot "qnn_htp\$name"
        if (-not (Test-Path $path)) { throw "QNN HTP model is missing: $path" }
    }
}

if (Test-Path $outputFull) { Remove-Item -LiteralPath $outputFull -Recurse -Force }
New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
Copy-Item -LiteralPath $exePath -Destination $outputFull

$excludedRuntimeFiles = @("onnxruntime_providers_tensorrt.dll", "libmediapipe.dll")
Get-ChildItem -LiteralPath $binaryDir -File |
    Where-Object {
        $_.Extension -in @(".dll", ".so", ".cat") -and
        $_.Name -notin $excludedRuntimeFiles
    } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $outputFull -Force }
foreach ($entry in @("qt.conf", "platforms", "imageformats")) {
    Copy-Item -LiteralPath (Join-Path $binaryDir $entry) -Destination $outputFull -Recurse -Force
}

$modelOut = Join-Path $outputFull "models\insightface\models\buffalo_l"
New-Item -ItemType Directory -Force -Path $modelOut | Out-Null
foreach ($name in @("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "1k3d68.onnx", "genderage.onnx")) {
    Copy-Item -LiteralPath (Join-Path $buffaloRoot $name) -Destination $modelOut -Force
}
if ($Architecture -eq "arm64") {
    Copy-Item -LiteralPath (Join-Path $buffaloRoot "qnn_htp") -Destination $modelOut -Recurse -Force
}
$faceMeshOut = Join-Path $outputFull "models\mediapipe"
New-Item -ItemType Directory -Force -Path $faceMeshOut | Out-Null
Copy-Item -LiteralPath $FaceMeshModel -Destination (Join-Path $faceMeshOut "face_landmarks_detector.onnx") -Force

if ($Accelerator -eq "cuda") {
    if (-not $CudaRuntimeRoot) { $CudaRuntimeRoot = Join-Path $depsRoot "cuda13-redist" }
    $cudaBin = if (Test-Path (Join-Path $CudaRuntimeRoot "bin")) { Join-Path $CudaRuntimeRoot "bin" } else { $CudaRuntimeRoot }
    $cudaRuntimeFiles = @(
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
    foreach ($required in $cudaRuntimeFiles) {
        if (-not (Test-Path (Join-Path $cudaBin $required))) { throw "CUDA redistributable is missing $required. Run fetch-cuda13-runtime.ps1." }
        Copy-Item -LiteralPath (Join-Path $cudaBin $required) -Destination $outputFull -Force
    }
    if (Test-Path (Join-Path $CudaRuntimeRoot "licenses")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $outputFull "licenses") | Out-Null
        Copy-Item -LiteralPath (Join-Path $CudaRuntimeRoot "licenses") -Destination (Join-Path $outputFull "licenses\NVIDIA") -Recurse -Force
    }
}

if (-not $VcRedistPath) {
    $redistName = if ($Architecture -eq "arm64") { "VC_redist.arm64.exe" } else { "VC_redist.x64.exe" }
    $redistCandidates = @()
    if ($Architecture -eq "arm64") {
        $redistCandidates += Get-ChildItem (Join-Path $depsRoot "vs-arm64-layout") -Recurse -Filter $redistName -File -ErrorAction SilentlyContinue
    }
    $redistCandidates += Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\$redistName" -File -ErrorAction SilentlyContinue
    $selectedRedist = $redistCandidates | Sort-Object FullName -Descending | Select-Object -First 1
    if ($selectedRedist) { $VcRedistPath = $selectedRedist.FullName }
}
if (-not $VcRedistPath -or -not (Test-Path $VcRedistPath)) {
    throw "The $Architecture Visual C++ Redistributable installer was not found."
}
$redistOut = Join-Path $outputFull "_redist"
New-Item -ItemType Directory -Force -Path $redistOut | Out-Null
$redistFileName = if ($Architecture -eq "arm64") { "VC_redist.arm64.exe" } else { "VC_redist.x64.exe" }
Copy-Item -LiteralPath $VcRedistPath -Destination (Join-Path $redistOut $redistFileName) -Force

$launcher = Join-Path $outputFull "Launch-FSCStudio.bat"
Set-Content -LiteralPath $launcher -Encoding ASCII -Value "@echo off`r`nstart `"`" `"%~dp0FscStudioQt.exe`" %*`r`n"

$manifest = [ordered]@{
    app = "FSC Studio"
    version = $AppVersion
    configuration = $Configuration
    architecture = $Architecture
    accelerator = $Accelerator
    camera = $true
    includes_python_runtime = $false
    includes_user_database = $false
    insightface_models = "models/insightface/models/buffalo_l"
    dense_mesh_model = "models/mediapipe/face_landmarks_detector.onnx"
    runtime = [ordered]@{
        onnx_runtime = (Get-Item (Join-Path $outputFull "onnxruntime.dll")).VersionInfo.FileVersion
        cuda_redistributables = ($Accelerator -eq "cuda")
        qnn = ($Accelerator -eq "qnn")
    }
    hashes = [ordered]@{
        executable_sha256 = (Get-FileHash (Join-Path $outputFull "FscStudioQt.exe") -Algorithm SHA256).Hash
        dense_mesh_sha256 = (Get-FileHash (Join-Path $faceMeshOut "face_landmarks_detector.onnx") -Algorithm SHA256).Hash
        recognition_model_sha256 = (Get-FileHash (Join-Path $modelOut "w600k_r50.onnx") -Algorithm SHA256).Hash
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputFull "package-manifest.json") -Encoding UTF8

if (-not $SkipSmoke -and $Architecture -eq "x64") {
    $smokeInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $smokeInfo.FileName = Join-Path $outputFull "FscStudioQt.exe"
    $smokeInfo.WorkingDirectory = $outputFull
    $smokeInfo.UseShellExecute = $false
    $smokeInfo.CreateNoWindow = $true
    $smokeInfo.Arguments = "--ui-language-smoke en"
    $smoke = [System.Diagnostics.Process]::Start($smokeInfo)
    if (-not $smoke.WaitForExit(20000)) {
        $smoke.Kill($true)
        throw "Packaged Qt smoke test timed out."
    }
    if ($smoke.ExitCode -ne 0) { throw "Packaged Qt smoke test failed with exit code $($smoke.ExitCode)." }
}

if ($Zip) {
    $zipPath = "$outputFull.zip"
    if (Test-Path $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Compress-Archive -Path (Join-Path $outputFull "*") -DestinationPath $zipPath
    Write-Host "Created $zipPath"
}

$size = (Get-ChildItem -LiteralPath $outputFull -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "Created $outputFull ($([math]::Round($size / 1MB, 1)) MiB)"
