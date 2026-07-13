param(
    [string]$Version = "6.11.1",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$dependencyRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot ".deps"))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $dependencyRoot "qt-$Version-arm64"
}
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $resolvedOutput.StartsWith($dependencyRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must stay under $dependencyRoot"
}

$versionToken = $Version.Replace(".", "")
$baseUrl = "https://download.qt.io/online/qtsdkrepository/windows_x86/desktop/qt6_$versionToken/qt6_${versionToken}_msvc2022_arm64_cross_compiled/qt.qt6.$versionToken.win64_msvc2022_arm64_cross_compiled"
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "fsc-qt-$Version-arm64"
$indexPath = "$tempRoot-index.html"

& curl.exe --location --fail --retry 3 --output $indexPath "$baseUrl/"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read the Qt ARM64 package index."
}
$index = Get-Content -LiteralPath $indexPath -Raw
$match = [regex]::Match($index, 'href="([^"]*qtbase[^"/]*\.7z)"')
if (-not $match.Success) {
    throw "The Qt ARM64 package index did not contain a qtbase archive."
}
$archiveName = $match.Groups[1].Value
$archivePath = "$tempRoot-$archiveName"
$checksumPath = "$archivePath.sha1"

& curl.exe --location --fail --retry 3 --continue-at - --output $archivePath "$baseUrl/$archiveName"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download Qt ARM64 qtbase."
}
& curl.exe --location --fail --retry 3 --output $checksumPath "$baseUrl/$archiveName.sha1"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download the Qt ARM64 checksum."
}
$expectedHash = (Get-Content -LiteralPath $checksumPath -Raw).Trim().ToLowerInvariant()
$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA1).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "Qt ARM64 archive checksum mismatch."
}

$sevenZip = @(
    (Get-Command 7z.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
    "$env:ProgramFiles\7-Zip\7z.exe"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $sevenZip) {
    throw "7-Zip is required to extract the official Qt package."
}

$stagingRoot = "$resolvedOutput-staging"
if (Test-Path $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
& $sevenZip x $archivePath "-o$stagingRoot" -y | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract Qt ARM64 qtbase."
}
if (Test-Path $resolvedOutput) {
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}
Move-Item -LiteralPath $stagingRoot -Destination $resolvedOutput

Write-Host "Qt ARM64 extracted to $resolvedOutput"
