param(
    [string]$Python = "",
    [string]$ModelRoot = "",
    [string]$Calibration = "",
    [ValidateRange(1, 10000)]
    [int]$MaxImages = 128,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\..\.."))
if (-not $ModelRoot) {
    $ModelRoot = Join-Path $workspaceRoot "model\insightface\models"
}
if (-not $Calibration) {
    $Calibration = Join-Path $workspaceRoot "new_full.fscdb"
}
if (-not $Python) {
    $candidate = Join-Path $workspaceRoot ".venv312\Scripts\python.exe"
    if (Test-Path $candidate) {
        $Python = $candidate
    } else {
        $Python = (Get-Command python.exe -ErrorAction Stop).Source
    }
}

$arguments = @(
    (Join-Path $scriptRoot "quantize-insightface-qnn.py"),
    "--models", $ModelRoot,
    "--calibration", $Calibration,
    "--max-images", $MaxImages
)
if ($Overwrite) {
    $arguments += "--overwrite"
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "QNN HTP model quantization failed with exit code $LASTEXITCODE."
}
