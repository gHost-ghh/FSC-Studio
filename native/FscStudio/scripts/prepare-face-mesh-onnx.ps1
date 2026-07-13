param(
    [string]$Python = "",
    [string]$Output = "",
    [string]$Task = "",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\..\.."))
if (-not $Output) { $Output = Join-Path $workspaceRoot "model\mediapipe\face_landmarks_detector.onnx" }
if (-not $Python) {
    $converter = Join-Path $workspaceRoot ".deps\face-mesh-converter\Scripts\python.exe"
    $existing = Join-Path $workspaceRoot ".deps\tf2onnx-venv\Scripts\python.exe"
    if (Test-Path $converter) {
        $Python = $converter
    } elseif (Test-Path $existing) {
        $Python = $existing
    } elseif ($Bootstrap) {
        $systemPython = (Get-Command python.exe -ErrorAction Stop).Source
        & $systemPython -m venv (Split-Path -Parent $converter)
        if ($LASTEXITCODE -ne 0) { throw "Failed to create the face mesh converter environment." }
        $Python = $converter
        & $Python -m pip install --disable-pip-version-check "tensorflow-cpu==2.18.1" "tf2onnx==1.17.0" "onnx==1.17.0"
        if ($LASTEXITCODE -ne 0) { throw "Failed to install the face mesh converter dependencies." }
    } else {
        throw "A TensorFlow/tf2onnx Python environment is required at build time. Rerun with -Bootstrap to create one."
    }
}

$arguments = @((Join-Path $scriptRoot "prepare-face-mesh-onnx.py"), "--output", $Output)
if ($Task) { $arguments += @("--task", $Task) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Face mesh ONNX conversion failed." }
