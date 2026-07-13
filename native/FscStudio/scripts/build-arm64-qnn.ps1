param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$BuildDir = "",
    [string]$QtRoot = "",
    [string]$OnnxRuntimeRoot = "",
    [string]$OpenCvRoot = "",
    [string]$SqliteRoot = "",
    [string]$ZlibRoot = "",
    [string]$MsvcRoot = "",
    [switch]$ConfigureOnly,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "..\.."))
$depsRoot = Join-Path $workspaceRoot ".deps"

if (-not $BuildDir) { $BuildDir = Join-Path $projectRoot "out\build\arm64-qt-qnn" }
if (-not $QtRoot) { $QtRoot = Join-Path $depsRoot "qt-6.11.1-arm64" }
if (-not $OnnxRuntimeRoot) { $OnnxRuntimeRoot = Join-Path $depsRoot "onnxruntime-qnn-1.24.4-nuget" }
if (-not $OpenCvRoot) { $OpenCvRoot = Join-Path $depsRoot "opencv-arm64\x64\vc17\lib" }
if (-not $SqliteRoot) { $SqliteRoot = Join-Path $depsRoot "sqlite-arm64" }
if (-not $ZlibRoot) { $ZlibRoot = Join-Path $depsRoot "zlib-arm64" }

if (-not $MsvcRoot) {
    $installedCompilers = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\arm64\cl.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    if ($installedCompilers) {
        $MsvcRoot = [System.IO.Path]::GetFullPath((Join-Path $installedCompilers[0].Directory.FullName "..\..\.."))
    } else {
        $portableTools = Get-ChildItem (Join-Path $depsRoot "msvc-arm64-portable\Contents\VC\Tools\MSVC\*") -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        if (-not $portableTools) {
            throw "MSVC ARM64 tools were not found. Install the VS 2022 ARM64 C++ workload or prepare .deps\msvc-arm64-portable."
        }
        $MsvcRoot = $portableTools[0].FullName
    }
}

$compiler = Join-Path $MsvcRoot "bin\Hostx64\arm64\cl.exe"
if (-not (Test-Path $compiler)) { throw "ARM64 compiler not found: $compiler" }
foreach ($required in @(
    (Join-Path $QtRoot "lib\cmake\Qt6\Qt6Config.cmake"),
    (Join-Path $OnnxRuntimeRoot "runtimes\win-arm64\native\onnxruntime.dll"),
    (Join-Path $OpenCvRoot "OpenCVConfig.cmake"),
    (Join-Path $SqliteRoot "include\sqlite3.h"),
    (Join-Path $SqliteRoot "lib\sqlite3.lib"),
    (Join-Path $ZlibRoot "bin\z.dll")
)) {
    if (-not (Test-Path $required)) { throw "ARM64 dependency not found: $required" }
}

$windowsKits = "${env:ProgramFiles(x86)}\Windows Kits\10"
$sdkVersions = Get-ChildItem (Join-Path $windowsKits "Include") -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        (Test-Path (Join-Path $_.FullName "um\Windows.h")) -and
        (Test-Path (Join-Path $windowsKits "Lib\$($_.Name)\um\arm64\kernel32.lib"))
    } |
    Sort-Object { [version]$_.Name } -Descending
if (-not $sdkVersions) { throw "A Windows SDK with ARM64 libraries was not found." }
$sdkVersion = $sdkVersions[0].Name

$env:INCLUDE = @(
    (Join-Path $MsvcRoot "include"),
    (Join-Path $windowsKits "Include\$sdkVersion\ucrt"),
    (Join-Path $windowsKits "Include\$sdkVersion\shared"),
    (Join-Path $windowsKits "Include\$sdkVersion\um"),
    (Join-Path $windowsKits "Include\$sdkVersion\winrt"),
    (Join-Path $windowsKits "Include\$sdkVersion\cppwinrt")
) -join ";"
$env:LIB = @(
    (Join-Path $MsvcRoot "lib\arm64"),
    (Join-Path $windowsKits "Lib\$sdkVersion\ucrt\arm64"),
    (Join-Path $windowsKits "Lib\$sdkVersion\um\arm64")
) -join ";"
$env:LIBPATH = @(
    (Join-Path $MsvcRoot "lib\arm64"),
    (Join-Path $windowsKits "UnionMetadata\$sdkVersion"),
    (Join-Path $windowsKits "References\$sdkVersion")
) -join ";"
$env:Path = @(
    (Join-Path $MsvcRoot "bin\Hostx64\arm64"),
    (Join-Path $windowsKits "bin\$sdkVersion\x64"),
    $env:Path
) -join ";"

$buildFull = [System.IO.Path]::GetFullPath($BuildDir)
$allowedRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot "out\build"))
if ($Clean -and (Test-Path $buildFull)) {
    if (-not $buildFull.StartsWith($allowedRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean ARM64 build directory outside $allowedRoot"
    }
    Remove-Item -LiteralPath $buildFull -Recurse -Force
}

$ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $ninja) {
    $ninja = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ninja) { throw "Ninja was not found." }

$configureArguments = @(
    "-S", $projectRoot,
    "-B", $buildFull,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_CXX_COMPILER=$compiler",
    "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DCMAKE_PREFIX_PATH=$QtRoot",
    "-DONNXRUNTIME_ROOT=$OnnxRuntimeRoot",
    "-DOpenCV_DIR=$OpenCvRoot",
    "-DSQLite3_INCLUDE_DIR=$(Join-Path $SqliteRoot 'include')",
    "-DSQLite3_LIBRARY=$(Join-Path $SqliteRoot 'lib\sqlite3.lib')",
    "-DFSC_SQLITE_RUNTIME_PATH=$(Join-Path $SqliteRoot 'bin\sqlite3.dll')",
    "-DFSC_ZLIB_RUNTIME_PATH=$(Join-Path $ZlibRoot 'bin\z.dll')",
    "-DFSC_CORE_ONLY=OFF",
    "-DFSC_ENABLE_ONNX=ON",
    "-DFSC_ENABLE_OPENCV=ON",
    "-DFSC_BUILD_QT_APP=ON"
)

& cmake @configureArguments
if ($LASTEXITCODE -ne 0) { throw "ARM64 CMake configure failed." }
if (-not $ConfigureOnly) {
    & cmake --build $buildFull --parallel
    if ($LASTEXITCODE -ne 0) { throw "ARM64 build failed." }
}

Write-Host "ARM64 QNN build ready: $buildFull"
