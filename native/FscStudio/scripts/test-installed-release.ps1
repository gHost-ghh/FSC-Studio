[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Installer,
    [ValidateSet("x64", "arm64")][string]$Architecture = "x64",
    [ValidateSet("directml", "cuda", "qnn")][string]$Accelerator = "directml",
    [string]$OutputRoot = "",
    [string]$ApplicationId = "{594F0143-C6C1-442A-9DE7-4D2528B3EA41}",
    [switch]$PerUser,
    [switch]$KeepInstalled
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
if (-not $OutputRoot) { $OutputRoot = Join-Path $projectRoot "out\install-test" }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$Installer = (Resolve-Path -LiteralPath $Installer).Path

$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $PerUser -and -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Installer acceptance must run from an elevated PowerShell window."
}

$hostArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
if ($Architecture -eq "arm64" -and $hostArchitecture -ne "arm64") {
    throw "The ARM64 installer must be executed on an ARM64 Windows device."
}

$appId = "$ApplicationId`_is1"
$uninstallKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$appId",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$appId",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$appId",
    "HKCU:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\$appId"
)
$existing = @($uninstallKeys | Where-Object { Test-Path -LiteralPath $_ })
if ($existing.Count -gt 0) {
    throw "An existing FSC Studio Inno installation was found. Uninstall it before isolated acceptance testing."
}

$variant = "$Architecture-$Accelerator"
$installDirectory = Join-Path $OutputRoot $variant
$logDirectory = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $OutputRoot, $logDirectory | Out-Null
if (Test-Path -LiteralPath $installDirectory) {
    $resolved = [System.IO.Path]::GetFullPath($installDirectory)
    if (-not $resolved.StartsWith($OutputRoot + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean an install-test directory outside $OutputRoot."
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory)][string]$FileName,
        [Parameter(Mandatory)][string[]]$Arguments,
        [int]$TimeoutSeconds = 900,
        [hashtable]$Environment = @{}
    )
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FileName
    $info.WorkingDirectory = Split-Path -Parent $FileName
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $quotedArguments = foreach ($argument in $Arguments) {
        if ($argument -notmatch '[\s"]') {
            $argument
            continue
        }
        $escaped = [regex]::Replace($argument, '(\\*)"', '$1$1\"')
        $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
        '"' + $escaped + '"'
    }
    $info.Arguments = $quotedArguments -join ' '
    foreach ($entry in $Environment.GetEnumerator()) { $info.Environment[$entry.Key] = $entry.Value }
    $process = [Diagnostics.Process]::Start($info)
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill()
        throw "Process timed out: $FileName"
    }
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    if ($process.ExitCode -ne 0) {
        throw "Process failed with exit code $($process.ExitCode): $FileName`n$stdout`n$stderr"
    }
}

$installerLog = Join-Path $logDirectory "$variant-install.log"
$uninstaller = Join-Path $installDirectory "unins000.exe"
$installed = $false
try {
    Invoke-CheckedProcess $Installer @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/NOICONS",
        "/DIR=$installDirectory",
        "/LOG=$installerLog"
    )
    $installed = $true

    $exe = Join-Path $installDirectory "FscStudioQt.exe"
    $modelRoot = Join-Path $installDirectory "models\insightface\models\buffalo_l"
    foreach ($required in @(
        $exe,
        (Join-Path $installDirectory "platforms\qwindows.dll"),
        (Join-Path $installDirectory "FSC-Studio-User-Guide.html"),
        (Join-Path $installDirectory "THIRD-PARTY-NOTICES.txt"),
        (Join-Path $installDirectory "LICENSE.txt"),
        (Join-Path $modelRoot "det_10g.onnx"),
        $uninstaller
    )) {
        if (-not (Test-Path -LiteralPath $required)) { throw "Installed release is missing $required" }
    }

    $forbidden = @(Get-ChildItem -LiteralPath $installDirectory -Recurse -File | Where-Object {
        $_.Name -match '^python(?:w|\d+)?(?:\.exe|\.dll)?$' -or
        $_.Extension -in @(".py", ".pyc", ".pyd", ".fscdb", ".dtb", ".jpg", ".jpeg", ".bmp")
    })
    if ($forbidden.Count -gt 0) {
        throw "Installed release contains Python, user data, or sample photos: $($forbidden.FullName -join ', ')"
    }

    $qtEnvironment = @{
        "QT_QPA_PLATFORM" = "windows"
        "FSC_QT_SMOKE_PLATFORM" = "windows"
    }
    Invoke-CheckedProcess $exe @("--ui-language-smoke", "en") 30 $qtEnvironment

    $expectedProvider = switch ($Accelerator) {
        "directml" { "DmlExecutionProvider" }
        "cuda" { "CUDAExecutionProvider" }
        "qnn" { "HTP/NPU" }
    }
    Invoke-CheckedProcess $exe @(
        "--runtime-probe-ui-smoke",
        $modelRoot,
        "auto",
        $expectedProvider
    ) 180 $qtEnvironment

    Write-Host "Installed release acceptance passed: $variant"
    Write-Host "Install log: $installerLog"
} finally {
    if ($installed -and -not $KeepInstalled -and (Test-Path -LiteralPath $uninstaller)) {
        Invoke-CheckedProcess $uninstaller @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") 300
        Start-Sleep -Milliseconds 500
        if (Test-Path -LiteralPath (Join-Path $installDirectory "FscStudioQt.exe")) {
            throw "The uninstaller left application files in $installDirectory"
        }
        Write-Host "Uninstall acceptance passed: $variant"
    }
}
