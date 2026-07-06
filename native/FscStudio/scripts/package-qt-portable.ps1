param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",

    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$ModelRoot = "D:\FSC\model\insightface\models",
    [switch]$Camera,
    [switch]$AllowOutsideOutput,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptRoot "..")
if (-not $BuildDir) {
    $flavor = if ($Camera) { "camera-" } else { "" }
    $presetName = if ($Configuration -eq "Release") { "msvc-vs-qt-$flavor" + "release" } else { "msvc-vs-qt-$flavor" + "debug" }
    $BuildDir = Join-Path $projectRoot "out\build\$presetName"
}
if (-not $OutputDir) {
    $packageName = if ($Camera) { "FSC-Studio-Native-Camera-$Configuration" } else { "FSC-Studio-Native-$Configuration" }
    $OutputDir = Join-Path $projectRoot "out\package\$packageName"
}

$packageRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "out\package"))
$outputFull = [System.IO.Path]::GetFullPath($OutputDir)
$insidePackageRoot = $outputFull.Equals($packageRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $outputFull.StartsWith($packageRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
if (-not $AllowOutsideOutput -and -not $insidePackageRoot) {
    throw "Refusing to clean output outside $packageRoot. Pass -AllowOutsideOutput only for an explicitly chosen package directory."
}

$buildRoot = Resolve-Path $BuildDir
$binaryDir = Join-Path $buildRoot $Configuration
$exePath = Join-Path $binaryDir "FscStudioQt.exe"
if (-not (Test-Path $exePath)) {
    throw "FscStudioQt.exe was not found. Build the Qt preset first: cmake --build --preset msvc-vs-qt-debug"
}

$modelRootPath = Resolve-Path $ModelRoot
$vcpkgInstalled = Join-Path $buildRoot "vcpkg_installed\x64-windows"
$qtPluginRoot = Join-Path $vcpkgInstalled ("debug\Qt6\plugins")
if ($Configuration -eq "Release") {
    $qtPluginRoot = Join-Path $vcpkgInstalled "Qt6\plugins"
}
$platformPlugin = Join-Path $qtPluginRoot "platforms"
if (-not (Test-Path $platformPlugin)) {
    throw "Qt platform plugins were not found under $qtPluginRoot"
}

if (Test-Path $outputFull) {
    Remove-Item -LiteralPath $outputFull -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $outputFull | Out-Null

Copy-Item -LiteralPath $exePath -Destination $outputFull
Get-ChildItem -LiteralPath $binaryDir -Filter "*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $outputFull
}

$pluginsOut = Join-Path $outputFull "platforms"
New-Item -ItemType Directory -Force -Path $pluginsOut | Out-Null
Get-ChildItem -LiteralPath $platformPlugin -Filter "qwindows*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $pluginsOut
}

$modelOut = Join-Path $outputFull "models\insightface\models"
New-Item -ItemType Directory -Force -Path $modelOut | Out-Null
Copy-Item -LiteralPath (Join-Path $modelRootPath "buffalo_l") -Destination $modelOut -Recurse -Force

$launcher = Join-Path $outputFull "Launch-FSCStudio.bat"
Set-Content -LiteralPath $launcher -Encoding ASCII -Value "@echo off`r`nstart """" ""%~dp0FscStudioQt.exe"" %*`r`n"

$manifest = [ordered]@{
    app = "FSC Studio Native"
    configuration = $Configuration
    camera = [bool]$Camera
    packaged_at = (Get-Date).ToString("o")
    includes_python_runtime = $false
    includes_user_database = $false
    model_root = "models/insightface/models"
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputFull "package-manifest.json") -Encoding UTF8

if ($Zip) {
    $zipPath = "$outputFull.zip"
    if (Test-Path $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $outputFull "*") -DestinationPath $zipPath
    Write-Host "Created $zipPath"
}

Write-Host "Created $outputFull"
