param(
    [ValidateSet("Release")]
    [string]$Configuration = "Release",
    [ValidateSet("x64", "arm64")]
    [string]$Architecture = "x64",
    [ValidateSet("directml", "cuda", "qnn")]
    [string]$Accelerator = "directml",
    [string]$PackageDir = "",
    [string]$OutputDir = "",
    [string]$AppVersion = "0.2.0",
    [switch]$SkipPackageSmoke
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
if (-not $PackageDir) {
    $packageArguments = @(
        "-Configuration", $Configuration,
        "-Architecture", $Architecture,
        "-Accelerator", $Accelerator,
        "-AppVersion", $AppVersion
    )
    if ($SkipPackageSmoke) { $packageArguments += "-SkipSmoke" }
    & (Join-Path $scriptRoot "package-qt-portable.ps1") @packageArguments
    if ($LASTEXITCODE -ne 0) { throw "Portable package staging failed." }
    $PackageDir = Join-Path $projectRoot "out\package\FSC-Studio-Windows-$Architecture-$($Accelerator.ToUpperInvariant())-$Configuration"
}
if (-not $OutputDir) { $OutputDir = Join-Path $projectRoot "out\installer" }

$packageFull = [System.IO.Path]::GetFullPath($PackageDir)
$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
$redistFileName = if ($Architecture -eq "arm64") { "VC_redist.arm64.exe" } else { "VC_redist.x64.exe" }
foreach ($entry in @(
    "FscStudioQt.exe",
    "qt.conf",
    "platforms\qwindows.dll",
    "imageformats\qjpeg.dll",
    "onnxruntime.dll",
    "models\mediapipe\face_landmarks_detector.onnx",
    "_redist\$redistFileName"
)) {
    if (-not (Test-Path (Join-Path $packageFull $entry))) { throw "Package directory is missing required runtime entry: $entry" }
}
if ($Accelerator -eq "cuda" -and -not (Test-Path (Join-Path $packageFull "onnxruntime_providers_cuda.dll"))) {
    throw "CUDA package is missing onnxruntime_providers_cuda.dll"
}
if ($Accelerator -eq "qnn" -and -not (Test-Path (Join-Path $packageFull "onnxruntime_providers_qnn.dll"))) {
    throw "QNN package is missing onnxruntime_providers_qnn.dll"
}

if ($Architecture -eq "x64") {
    $runtimeSmokeInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $runtimeSmokeInfo.FileName = Join-Path $packageFull "FscStudioQt.exe"
    $runtimeSmokeInfo.WorkingDirectory = $packageFull
    $runtimeSmokeInfo.UseShellExecute = $false
    $runtimeSmokeInfo.CreateNoWindow = $true
    $runtimeSmokeInfo.Arguments = "--ui-language-smoke en"
    $runtimeSmokeInfo.Environment["QT_QPA_PLATFORM"] = "windows"
    $runtimeSmokeInfo.Environment["FSC_QT_SMOKE_PLATFORM"] = "windows"
    $runtimeSmokeInfo.Environment.Remove("QT_PLUGIN_PATH") | Out-Null
    $runtimeSmoke = [System.Diagnostics.Process]::Start($runtimeSmokeInfo)
    if (-not $runtimeSmoke.WaitForExit(20000)) {
        $runtimeSmoke.Kill($true)
        throw "Qt Windows platform runtime smoke timed out."
    }
    if ($runtimeSmoke.ExitCode -ne 0) { throw "Qt Windows platform runtime smoke failed with exit code $($runtimeSmoke.ExitCode)." }
}

$isccCandidates = @(
    @(
        (Get-Command iscc.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
)
if (-not $isccCandidates) { throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup first." }

$setupBaseName = if ($Architecture -eq "arm64") {
    "FSC-Studio-Setup-arm64"
} elseif ($Accelerator -eq "cuda") {
    "FSC-Studio-CUDA-Setup-x64"
} else {
    "FSC-Studio-Setup-x64"
}

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
$source = Join-Path $projectRoot "installer\FSCStudio.iss"
& $isccCandidates[0] "/DSourceDir=$packageFull" "/DOutputDir=$outputFull" "/DAppVersion=$AppVersion" "/DArchitecture=$Architecture" "/DSetupBaseName=$setupBaseName" "/DVCRedistFile=$redistFileName" $source
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed." }

$installer = Join-Path $outputFull "$setupBaseName.exe"
if (-not (Test-Path $installer)) { throw "Installer output was not created: $installer" }
Get-Item $installer | Select-Object FullName, Length, LastWriteTime
