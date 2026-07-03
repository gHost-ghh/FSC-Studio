param(
    [ValidateSet("cpu", "gpu", "universal")]
    [string]$Variant = "universal",
    [string]$OutputRoot = "release\optimized",
    [string]$PythonVersion = "3.12.10",
    [bool]$IncludeDenseMesh = $false,
    [bool]$IncludeCudaDlls = $false,
    [bool]$SkipInstaller = $false
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PackageRoot = $PSScriptRoot
$OutputRoot = Join-Path $RepoRoot $OutputRoot
$CacheRoot = Join-Path $OutputRoot "cache"
$WorkRoot = Join-Path $OutputRoot "work"
$PayloadRoot = Join-Path $OutputRoot "payloads"
$InstallerRoot = Join-Path $OutputRoot "installer"
$StageCommon = Join-Path $WorkRoot "common"
$StageCpu = Join-Path $WorkRoot "cpu"
$StageGpu = Join-Path $WorkRoot "gpu"

function Write-Step {
    param([string]$Message)
    Write-Host "[FSC release] $Message" -ForegroundColor Cyan
}

function Remove-IfExists {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Invoke-Download {
    param([string]$Url, [string]$OutputPath)
    if (Test-Path $OutputPath) {
        return
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null
    Write-Step "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $OutputPath
}

function Find-7Zip {
    $candidates = @(
        (Join-Path $RepoRoot "release\tools_7zip\7zr.exe"),
        (Join-Path $RepoRoot "release\tools_7zip\extra\x64\7za.exe"),
        (Join-Path $RepoRoot "release\tools_7zip\extra\7za.exe"),
        "7z.exe",
        "7za.exe",
        "7zr.exe"
    )
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "7-Zip command line tool was not found. Keep release\tools_7zip from the previous build or install 7-Zip."
}

function Find-SfxModule {
    $candidates = @(
        (Join-Path $RepoRoot "release\tools_7zip\lzma\bin\7zSD.sfx"),
        (Join-Path $RepoRoot "release\tools_7zip\lzma\bin\7zS2.sfx")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "7-Zip SFX module was not found. Keep release\tools_7zip from the previous build."
}

function Copy-FileIfExists {
    param([string]$Source, [string]$Destination)
    if (Test-Path $Source) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Initialize-EmbeddedPython {
    param([string]$TargetRoot)
    $zipName = "python-$PythonVersion-embed-amd64.zip"
    $zipPath = Join-Path $CacheRoot $zipName
    $url = "https://www.python.org/ftp/python/$PythonVersion/$zipName"
    Invoke-Download -Url $url -OutputPath $zipPath
    $pythonRoot = Join-Path $TargetRoot "python"
    Remove-IfExists $pythonRoot
    New-Item -ItemType Directory -Path $pythonRoot -Force | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $pythonRoot -Force

    $pth = Get-ChildItem -LiteralPath $pythonRoot -Filter "python*._pth" | Select-Object -First 1
    if (-not $pth) {
        throw "Embedded Python ._pth file was not found."
    }
    @(
        "python$($PythonVersion.Split('.')[0])$($PythonVersion.Split('.')[1]).zip",
        ".",
        "Lib\site-packages",
        "import site"
    ) | Set-Content -LiteralPath $pth.FullName -Encoding ASCII
    New-Item -ItemType Directory -Path (Join-Path $pythonRoot "Lib\site-packages") -Force | Out-Null
}

function Install-PipTarget {
    param(
        [string[]]$Arguments,
        [string]$TargetSite
    )
    $python = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        $python = "python"
    }
    & $python -m pip install --disable-pip-version-check --no-input --upgrade --target $TargetSite --only-binary=:all: @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed: $($Arguments -join ' ')"
    }
}

function Install-CommonDependencies {
    param([string]$TargetRoot)
    $site = Join-Path $TargetRoot "python\Lib\site-packages"
    Install-PipTarget -TargetSite $site -Arguments @("-r", (Join-Path $PackageRoot "requirements-common.txt"))
    Install-PipTarget -TargetSite $site -Arguments @("--no-deps", "insightface==1.0.1")
    if ($IncludeDenseMesh) {
        Install-PipTarget -TargetSite $site -Arguments @("--no-deps", "mediapipe==0.10.35", "absl-py==2.3.1", "sounddevice==0.5.3")
    }
}

function Install-RuntimeOverlay {
    param([string]$TargetRoot, [string]$RequirementsPath)
    $site = Join-Path $TargetRoot "python\Lib\site-packages"
    New-Item -ItemType Directory -Path $site -Force | Out-Null
    Install-PipTarget -TargetSite $site -Arguments @("--no-deps", "-r", $RequirementsPath)
}

function Copy-CudaDllsIfRequested {
    param([string]$TargetRoot)
    if (-not $IncludeCudaDlls) {
        return
    }
    $source = Join-Path $RepoRoot ".venv312\Lib\site-packages\nvidia"
    if (-not (Test-Path $source)) {
        Write-Warning "IncludeCudaDlls was requested, but no local nvidia package directory exists at $source."
        return
    }
    $destination = Join-Path $TargetRoot "python\Lib\site-packages\nvidia"
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

function Remove-MatchingDirectories {
    param([string]$Root, [string[]]$Names)
    foreach ($name in $Names) {
        Get-ChildItem -LiteralPath $Root -Recurse -Force -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ieq $name } |
            Sort-Object FullName -Descending |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

function Prune-Runtime {
    param([string]$TargetRoot)
    $pythonRoot = Join-Path $TargetRoot "python"
    $site = Join-Path $pythonRoot "Lib\site-packages"
    if (-not (Test-Path $site)) {
        return
    }
    Remove-MatchingDirectories -Root $site -Names @("__pycache__", "tests", "test", "testing", "docs", "doc", "examples", "sample", "samples")
    $removeExtensions = @(".pyc", ".pyo", ".pdb", ".lib", ".a", ".h", ".hpp", ".pxd")
    Get-ChildItem -LiteralPath $site -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $removeExtensions -contains $_.Extension.ToLowerInvariant() } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    foreach ($name in @("pip", "setuptools", "wheel")) {
        Remove-IfExists (Join-Path $site $name)
        Get-ChildItem -LiteralPath $site -Force -Directory -Filter "$name-*.dist-info" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    $pyside = Join-Path $site "PySide6"
    if (Test-Path $pyside) {
        foreach ($relative in @(
            "doc",
            "include",
            "lib",
            "qml",
            "metatypes",
            "glue",
            "resources",
            "translations",
            "typesystems",
            "plugins\designer",
            "plugins\generic",
            "plugins\qmllint",
            "plugins\qmltooling",
            "plugins\sqldrivers",
            "plugins\tls",
            "plugins\networkinformation",
            "plugins\vectorimageformats"
        )) {
            Remove-IfExists (Join-Path $pyside $relative)
        }
        $pysideRemoveExtensions = @(".exe", ".lib", ".pdb")
        Get-ChildItem -LiteralPath $pyside -Force -File -ErrorAction SilentlyContinue |
            Where-Object { $pysideRemoveExtensions -contains $_.Extension.ToLowerInvariant() } |
            Remove-Item -Force -ErrorAction SilentlyContinue
        $unusedQtPatterns = @(
            "opengl32sw.dll",
            "Qt6Designer*.dll",
            "Qt6Help.dll",
            "Qt6Labs*.dll",
            "Qt6Pdf*.dll",
            "Qt6Qml*.dll",
            "Qt6Quick*.dll",
            "Qt6Sql.dll",
            "Qt6Test.dll",
            "Qt6UiTools.dll",
            "QtDesigner.pyd",
            "QtHelp.pyd",
            "QtPdf*.pyd",
            "QtQml*.pyd",
            "QtQuick*.pyd",
            "QtQuickControls2.pyd",
            "QtSql.pyd",
            "QtTest.pyd",
            "QtUiTools.pyd"
        )
        foreach ($pattern in $unusedQtPatterns) {
            Get-ChildItem -LiteralPath $pyside -Force -File -Filter $pattern -ErrorAction SilentlyContinue |
                Remove-Item -Force -ErrorAction SilentlyContinue
        }
    }

    $cv2 = Join-Path $site "cv2"
    if (Test-Path $cv2) {
        foreach ($relative in @("samples", "data", "cuda", "gapi", "misc", "utils", "typing")) {
            Remove-IfExists (Join-Path $cv2 $relative)
        }
        Get-ChildItem -LiteralPath $cv2 -Force -File -Filter "opencv_videoio_ffmpeg*.dll" -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

function Copy-ApplicationFiles {
    param([string]$TargetRoot)
    foreach ($file in @(
        "fsc_studio.py",
        "fsc_studio_services.py",
        "fsc_face_engine.py",
        "fsc_face_database.py",
        "README.md",
        "LICENSE"
    )) {
        Copy-FileIfExists -Source (Join-Path $RepoRoot $file) -Destination (Join-Path $TargetRoot $file)
    }
    Copy-FileIfExists -Source (Join-Path $PackageRoot "fsc_launcher.py") -Destination (Join-Path $TargetRoot "fsc_launcher.py")
    Copy-FileIfExists -Source (Join-Path $PackageRoot "runtime_probe.py") -Destination (Join-Path $TargetRoot "runtime_probe.py")

    @'
@echo off
setlocal
cd /d "%~dp0"
set "FSC_RUNTIME_MODE=auto"
if exist "runtime_mode.txt" set /p FSC_RUNTIME_MODE=<"runtime_mode.txt"
if /I "%FSC_RUNTIME_MODE%"=="cpu" set "FSC_FORCE_CPU=1"
"%~dp0python\pythonw.exe" "%~dp0fsc_launcher.py"
'@ | Set-Content -LiteralPath (Join-Path $TargetRoot "FSC Studio.cmd") -Encoding ASCII

    @'
@echo off
setlocal
cd /d "%~dp0"
set "FSC_RUNTIME_MODE=auto"
if exist "runtime_mode.txt" set /p FSC_RUNTIME_MODE=<"runtime_mode.txt"
if /I "%FSC_RUNTIME_MODE%"=="cpu" set "FSC_FORCE_CPU=1"
"%~dp0python\python.exe" "%~dp0fsc_launcher.py"
pause
'@ | Set-Content -LiteralPath (Join-Path $TargetRoot "FSC Studio Debug.cmd") -Encoding ASCII
}

function Copy-ModelFiles {
    param([string]$TargetRoot)
    $sourceModel = Join-Path $RepoRoot "model\insightface\models\buffalo_l"
    $destModel = Join-Path $TargetRoot "model\insightface\models\buffalo_l"
    if (-not (Test-Path $sourceModel)) {
        throw "InsightFace buffalo_l model folder is missing: $sourceModel"
    }
    New-Item -ItemType Directory -Path $destModel -Force | Out-Null
    Get-ChildItem -LiteralPath $sourceModel -Filter "*.onnx" -File |
        Copy-Item -Destination $destModel -Force

    if ($IncludeDenseMesh) {
        Copy-FileIfExists `
            -Source (Join-Path $RepoRoot "model\mediapipe\face_landmarker.task") `
            -Destination (Join-Path $TargetRoot "model\mediapipe\face_landmarker.task")
    }
}

function Compress-Payload {
    param([string]$SourceRoot, [string]$OutputPath, [string]$SevenZip)
    Remove-Item -LiteralPath $OutputPath -Force -ErrorAction SilentlyContinue
    Push-Location $SourceRoot
    try {
        & $SevenZip a -t7z -mx=9 -m0=LZMA2 -ms=on $OutputPath ".\*" | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed while creating $OutputPath"
        }
    } finally {
        Pop-Location
    }
}

function Join-BinaryFiles {
    param([string[]]$Inputs, [string]$Output)
    $stream = [System.IO.File]::Open($Output, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
    try {
        foreach ($input in $Inputs) {
            $bytes = [System.IO.File]::ReadAllBytes($input)
            $stream.Write($bytes, 0, $bytes.Length)
        }
    } finally {
        $stream.Dispose()
    }
}

function Build-SfxInstaller {
    param([string]$SevenZip, [string]$SfxModule)
    Remove-IfExists $InstallerRoot
    New-Item -ItemType Directory -Path $InstallerRoot -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $PackageRoot "installer\install.cmd") -Destination $InstallerRoot -Force
    Copy-Item -LiteralPath (Join-Path $PackageRoot "installer\install.ps1") -Destination $InstallerRoot -Force
    Copy-Item -LiteralPath $SevenZip -Destination (Join-Path $InstallerRoot "7zr.exe") -Force
    Copy-Item -LiteralPath (Join-Path $PayloadRoot "common.7z") -Destination $InstallerRoot -Force
    if ($Variant -in @("cpu", "universal")) {
        Copy-Item -LiteralPath (Join-Path $PayloadRoot "cpu.7z") -Destination $InstallerRoot -Force
    }
    if ($Variant -in @("gpu", "universal")) {
        Copy-Item -LiteralPath (Join-Path $PayloadRoot "gpu.7z") -Destination $InstallerRoot -Force
    }

    $setupPayload = Join-Path $OutputRoot "setup_payload.7z"
    Compress-Payload -SourceRoot $InstallerRoot -OutputPath $setupPayload -SevenZip $SevenZip
    $config = Join-Path $OutputRoot "sfx_config.txt"
    @'
;!@Install@!UTF-8!
Title="FSC Studio Setup"
BeginPrompt="Install FSC Studio?"
RunProgram="install.cmd"
;!@InstallEnd@!
'@ | Set-Content -LiteralPath $config -Encoding UTF8

    $setupName = switch ($Variant) {
        "cpu" { "FSC_Studio_CPU_Setup.exe" }
        "gpu" { "FSC_Studio_GPU_Setup.exe" }
        default { "FSC_Studio_Universal_Setup.exe" }
    }
    $setupPath = Join-Path $OutputRoot $setupName
    Join-BinaryFiles -Inputs @($SfxModule, $config, $setupPayload) -Output $setupPath
    Write-Step "Created installer: $setupPath"
}

Write-Step "Preparing release workspace: $OutputRoot"
New-Item -ItemType Directory -Path $CacheRoot, $WorkRoot, $PayloadRoot -Force | Out-Null
Remove-IfExists $WorkRoot
New-Item -ItemType Directory -Path $WorkRoot, $PayloadRoot -Force | Out-Null

$sevenZip = Find-7Zip
$sfxModule = $null
if (-not $SkipInstaller) {
    $sfxModule = Find-SfxModule
}

Write-Step "Building common payload"
New-Item -ItemType Directory -Path $StageCommon -Force | Out-Null
Initialize-EmbeddedPython -TargetRoot $StageCommon
Install-CommonDependencies -TargetRoot $StageCommon
Copy-ApplicationFiles -TargetRoot $StageCommon
Copy-ModelFiles -TargetRoot $StageCommon
Set-Content -LiteralPath (Join-Path $StageCommon "runtime_mode.txt") -Value "auto" -Encoding UTF8
Prune-Runtime -TargetRoot $StageCommon
Compress-Payload -SourceRoot $StageCommon -OutputPath (Join-Path $PayloadRoot "common.7z") -SevenZip $sevenZip

if ($Variant -in @("cpu", "universal")) {
    Write-Step "Building CPU overlay"
    New-Item -ItemType Directory -Path $StageCpu -Force | Out-Null
    Install-RuntimeOverlay -TargetRoot $StageCpu -RequirementsPath (Join-Path $PackageRoot "requirements-cpu.txt")
    Prune-Runtime -TargetRoot $StageCpu
    Compress-Payload -SourceRoot $StageCpu -OutputPath (Join-Path $PayloadRoot "cpu.7z") -SevenZip $sevenZip
}

if ($Variant -in @("gpu", "universal")) {
    Write-Step "Building GPU overlay"
    New-Item -ItemType Directory -Path $StageGpu -Force | Out-Null
    Install-RuntimeOverlay -TargetRoot $StageGpu -RequirementsPath (Join-Path $PackageRoot "requirements-gpu.txt")
    Copy-CudaDllsIfRequested -TargetRoot $StageGpu
    Prune-Runtime -TargetRoot $StageGpu
    Compress-Payload -SourceRoot $StageGpu -OutputPath (Join-Path $PayloadRoot "gpu.7z") -SevenZip $sevenZip
}

if (-not $SkipInstaller) {
    Build-SfxInstaller -SevenZip $sevenZip -SfxModule $sfxModule
}

Write-Step "Done"
