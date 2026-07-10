param(
    [ValidateSet("Release")]
    [string]$Configuration = "Release",

    [string]$PackageDir = "",
    [string]$OutputDir = "",
    [string]$AppVersion = "0.1.0"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptRoot "..")
if (-not $PackageDir) {
    & (Join-Path $scriptRoot "package-qt-portable.ps1") -Configuration $Configuration -Camera -DirectML
    $PackageDir = Join-Path $projectRoot "out\package\FSC-Studio-Native-Camera-DirectML-Release"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot "out\installer"
}

$packageFull = [System.IO.Path]::GetFullPath($PackageDir)
$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path (Join-Path $packageFull "FscStudioQt.exe"))) {
    throw "Package directory does not contain FscStudioQt.exe: $packageFull"
}
foreach ($entry in @("qt.conf", "platforms\qwindows.dll", "imageformats\qjpeg.dll", "onnxruntime.dll", "libmediapipe.dll")) {
    if (-not (Test-Path (Join-Path $packageFull $entry))) {
        throw "Package directory is missing required runtime entry: $entry"
    }
}

$runtimeSmokeInfo = [System.Diagnostics.ProcessStartInfo]::new()
$runtimeSmokeInfo.FileName = Join-Path $packageFull "FscStudioQt.exe"
$runtimeSmokeInfo.Arguments = "--ui-language-smoke en"
$runtimeSmokeInfo.WorkingDirectory = $packageFull
$runtimeSmokeInfo.UseShellExecute = $false
$runtimeSmokeInfo.CreateNoWindow = $true
$runtimeSmokeInfo.Environment["QT_QPA_PLATFORM"] = "windows"
$runtimeSmokeInfo.Environment.Remove("QT_PLUGIN_PATH") | Out-Null
$runtimeSmoke = [System.Diagnostics.Process]::Start($runtimeSmokeInfo)
if (-not $runtimeSmoke.WaitForExit(15000)) {
    $runtimeSmoke.Kill($true)
    throw "Qt Windows platform runtime smoke timed out."
}
if ($runtimeSmoke.ExitCode -ne 0) {
    throw "Qt Windows platform runtime smoke failed with exit code $($runtimeSmoke.ExitCode)."
}

$isccCandidates = @(
    @(
        (Get-Command iscc.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }
)
if (-not $isccCandidates) {
    throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup first."
}

New-Item -ItemType Directory -Force -Path $outputFull | Out-Null
$source = Join-Path $projectRoot "installer\FSCStudio.iss"
& $isccCandidates[0] "/DSourceDir=$packageFull" "/DOutputDir=$outputFull" "/DAppVersion=$AppVersion" $source
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed."
}

Get-ChildItem -LiteralPath $outputFull -Filter "FSC-Studio-Setup-x64.exe" | Select-Object FullName,Length,LastWriteTime
