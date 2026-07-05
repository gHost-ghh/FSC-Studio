param(
    [string]$VcpkgRoot = "$PSScriptRoot\..\..\..\.deps\vcpkg"
)

$ErrorActionPreference = "Stop"

$resolvedRoot = [System.IO.Path]::GetFullPath($VcpkgRoot)
$parent = Split-Path -Parent $resolvedRoot
if (!(Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}

if (!(Test-Path (Join-Path $resolvedRoot ".git"))) {
    git clone https://github.com/microsoft/vcpkg.git $resolvedRoot
}

& (Join-Path $resolvedRoot "bootstrap-vcpkg.bat")

$env:VCPKG_ROOT = $resolvedRoot
Write-Host "VCPKG_ROOT=$resolvedRoot"
Write-Host "Configure with:"
Write-Host "  cmake --preset msvc-debug -DCMAKE_TOOLCHAIN_FILE=`"$resolvedRoot\scripts\buildsystems\vcpkg.cmake`""
Write-Host "  cmake --build --preset msvc-debug"
