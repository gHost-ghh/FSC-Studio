[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PackageDirectory,
    [Parameter(Mandatory)][string]$Database,
    [Parameter(Mandatory)][string]$ImageA,
    [Parameter(Mandatory)][string]$ImageB,
    [string]$ClusterDatabase = "",
    [string]$LegacyDtb = "",
    [string]$OutputRoot = "",
    [switch]$IncludePhysicalCamera,
    [int]$CameraIndex = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne [System.Runtime.InteropServices.Architecture]::Arm64) {
    throw "This acceptance script must run on native ARM64 Windows."
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$PackageDirectory = (Resolve-Path -LiteralPath $PackageDirectory).Path
$Database = (Resolve-Path -LiteralPath $Database).Path
$ImageA = (Resolve-Path -LiteralPath $ImageA).Path
$ImageB = (Resolve-Path -LiteralPath $ImageB).Path
if (-not $ClusterDatabase) { $ClusterDatabase = $Database }
$ClusterDatabase = (Resolve-Path -LiteralPath $ClusterDatabase).Path
if ($LegacyDtb) { $LegacyDtb = (Resolve-Path -LiteralPath $LegacyDtb).Path }
if (-not $OutputRoot) { $OutputRoot = Join-Path $projectRoot "out\snapdragon-acceptance" }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$modelRoot = Join-Path $PackageDirectory "models\insightface\models\buffalo_l"
foreach ($required in @(
    (Join-Path $PackageDirectory "FscStudioQt.exe"),
    (Join-Path $PackageDirectory "onnxruntime_providers_qnn.dll"),
    (Join-Path $PackageDirectory "QnnHtp.dll"),
    (Join-Path $PackageDirectory "QnnGpu.dll"),
    (Join-Path $modelRoot "qnn_htp\det_10g.onnx")
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "ARM64 package is missing $required" }
}

$common = @{
    BuildDirectory = $PackageDirectory
    Database = $Database
    ClusterDatabase = $ClusterDatabase
    ModelRoot = $modelRoot
    ImageA = $ImageA
    ImageB = $ImageB
    CameraIndex = $CameraIndex
}
if ($LegacyDtb) { $common.LegacyDtb = $LegacyDtb }

$npu = $common.Clone()
$npu.RuntimeMode = "qnn-npu"
$npu.ExpectedProvider = "HTP/NPU"
$npu.OutputRoot = Join-Path $OutputRoot "npu"
if ($IncludePhysicalCamera) { $npu.IncludePhysicalCamera = $true }
& (Join-Path $scriptRoot "test-native-release.ps1") @npu

$gpu = $common.Clone()
$gpu.RuntimeMode = "qnn-gpu"
$gpu.ExpectedProvider = "Adreno GPU"
$gpu.OutputRoot = Join-Path $OutputRoot "gpu"
$gpu.SkipPageRenders = $true
& (Join-Path $scriptRoot "test-native-release.ps1") @gpu

$cpu = $common.Clone()
$cpu.RuntimeMode = "cpu"
$cpu.ExpectedProvider = "CPUExecutionProvider"
$cpu.OutputRoot = Join-Path $OutputRoot "cpu"
$cpu.SkipPageRenders = $true
& (Join-Path $scriptRoot "test-native-release.ps1") @cpu

Write-Host "Snapdragon CPU, Adreno GPU, and HTP/NPU acceptance passed."
Write-Host "Artifacts: $OutputRoot"
