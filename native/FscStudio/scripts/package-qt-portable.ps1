param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",

    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [string]$ModelRoot = "D:\FSC\model\insightface\models",
    [string]$MediaPipeModel = "",
    [switch]$Camera,
    [switch]$DirectML,
    [switch]$AllowOutsideOutput,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptRoot "..")
$workspaceRoot = Resolve-Path (Join-Path $projectRoot "..\..")
if (-not $MediaPipeModel) {
    $MediaPipeModel = Join-Path $workspaceRoot "model\mediapipe\face_landmarker.task"
}
if (-not $BuildDir) {
    $flavor = ""
    if ($Camera) {
        $flavor += "camera-"
    }
    if ($DirectML) {
        $flavor += "dml-"
    }
    $presetName = if ($Configuration -eq "Release") { "msvc-vs-qt-$flavor" + "release" } else { "msvc-vs-qt-$flavor" + "debug" }
    $BuildDir = Join-Path $projectRoot "out\build\$presetName"
}
if (-not $OutputDir) {
    $parts = @("FSC-Studio-Native")
    if ($Camera) {
        $parts += "Camera"
    }
    if ($DirectML) {
        $parts += "DirectML"
    }
    $parts += $Configuration
    $packageName = $parts -join "-"
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
$mediaPipeModelPath = Resolve-Path $MediaPipeModel
$qtRuntimeEntries = @("qt.conf", "platforms", "imageformats")
foreach ($entry in $qtRuntimeEntries) {
    if (-not (Test-Path (Join-Path $binaryDir $entry))) {
        throw "Qt runtime entry '$entry' was not staged beside FscStudioQt.exe. Reconfigure and rebuild the Qt preset first."
    }
}

if (Test-Path $outputFull) {
    Remove-Item -LiteralPath $outputFull -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $outputFull | Out-Null

Copy-Item -LiteralPath $exePath -Destination $outputFull
Get-ChildItem -LiteralPath $binaryDir -Filter "*.dll" | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $outputFull
}
if (-not (Test-Path (Join-Path $outputFull "libmediapipe.dll"))) {
    throw "libmediapipe.dll was not staged by the native build. Configure FSC_MEDIAPIPE_RUNTIME_PATH and rebuild before packaging."
}
foreach ($entry in $qtRuntimeEntries) {
    Copy-Item -LiteralPath (Join-Path $binaryDir $entry) -Destination $outputFull -Recurse -Force
}

$modelOut = Join-Path $outputFull "models\insightface\models"
New-Item -ItemType Directory -Force -Path $modelOut | Out-Null
Copy-Item -LiteralPath (Join-Path $modelRootPath "buffalo_l") -Destination $modelOut -Recurse -Force
$mediaPipeOut = Join-Path $outputFull "models\mediapipe"
New-Item -ItemType Directory -Force -Path $mediaPipeOut | Out-Null
Copy-Item -LiteralPath $mediaPipeModelPath -Destination (Join-Path $mediaPipeOut "face_landmarker.task") -Force

$launcher = Join-Path $outputFull "Launch-FSCStudio.bat"
Set-Content -LiteralPath $launcher -Encoding ASCII -Value "@echo off`r`nstart """" ""%~dp0FscStudioQt.exe"" %*`r`n"

$installScript = @'
param(
    [string]$InstallDir = "",
    [switch]$AllUsers,
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"
$source = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $InstallDir) {
    if ($AllUsers) {
        $InstallDir = Join-Path $env:ProgramFiles "FSC Studio"
    } else {
        $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\FSC Studio"
    }
}

if (-not (Test-Path $source\FscStudioQt.exe)) {
    throw "Run this installer from the unpacked FSC Studio package directory."
}
if (Test-Path $InstallDir) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Path (Join-Path $source "*") -Destination $InstallDir -Recurse -Force

if (-not $NoShortcut) {
    $startMenuRoot = if ($AllUsers) {
        Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs"
    } else {
        Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
    }
    $shortcutPath = Join-Path $startMenuRoot "FSC Studio.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $InstallDir "FscStudioQt.exe"
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Description = "FSC Studio"
    $shortcut.Save()
    Write-Host "Shortcut: $shortcutPath"
}

Write-Host "Installed FSC Studio to $InstallDir"
'@
Set-Content -LiteralPath (Join-Path $outputFull "Install-FSCStudioNative.ps1") -Encoding UTF8 -Value $installScript
Set-Content -LiteralPath (Join-Path $outputFull "Install-FSCStudioNative.bat") -Encoding ASCII -Value "@echo off`r`npowershell -ExecutionPolicy Bypass -File ""%~dp0Install-FSCStudioNative.ps1"" %*`r`npause`r`n"

$uninstallScript = @'
param(
    [string]$InstallDir = "",
    [switch]$AllUsers
)

$ErrorActionPreference = "Stop"
if (-not $InstallDir) {
    if ($AllUsers) {
        $InstallDir = Join-Path $env:ProgramFiles "FSC Studio"
    } else {
        $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\FSC Studio"
    }
}
$startMenuRoot = if ($AllUsers) {
    Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs"
} else {
    Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
}
$shortcutPath = Join-Path $startMenuRoot "FSC Studio.lnk"
if (Test-Path $shortcutPath) {
    Remove-Item -LiteralPath $shortcutPath -Force
}
if (Test-Path $InstallDir) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}
Write-Host "Removed FSC Studio from $InstallDir"
'@
Set-Content -LiteralPath (Join-Path $outputFull "Uninstall-FSCStudioNative.ps1") -Encoding UTF8 -Value $uninstallScript

$manifest = [ordered]@{
    app = "FSC Studio"
    configuration = $Configuration
    camera = [bool]$Camera
    directml = [bool]$DirectML
    packaged_at = (Get-Date).ToString("o")
    includes_python_runtime = $false
    includes_user_database = $false
    model_root = "models/insightface/models"
    mediapipe = [ordered]@{
        runtime = "libmediapipe.dll"
        model = "models/mediapipe/face_landmarker.task"
        python_runtime_required = $false
    }
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
