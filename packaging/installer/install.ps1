param(
    [string]$DefaultTarget = "$env:LOCALAPPDATA\FSC Studio"
)

$ErrorActionPreference = "Stop"

function Show-Choice {
    param(
        [string]$Title,
        [string]$Message,
        [string]$DefaultMode
    )
    Add-Type -AssemblyName System.Windows.Forms
    $buttons = [System.Windows.Forms.MessageBoxButtons]::YesNoCancel
    $icon = [System.Windows.Forms.MessageBoxIcon]::Question
    $result = [System.Windows.Forms.MessageBox]::Show($Message, $Title, $buttons, $icon)
    if ($result -eq [System.Windows.Forms.DialogResult]::Cancel) {
        throw "Installation cancelled."
    }
    if ($DefaultMode -eq "gpu") {
        return $(if ($result -eq [System.Windows.Forms.DialogResult]::Yes) { "gpu" } else { "cpu" })
    }
    return $(if ($result -eq [System.Windows.Forms.DialogResult]::Yes) { "cpu" } else { "gpu" })
}

function Test-CudaSignal {
    $signals = New-Object System.Collections.Generic.List[string]
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        try {
            $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($gpu) { $signals.Add("NVIDIA GPU: $gpu") }
        } catch {}
    }
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin"))) {
        $signals.Add("CUDA_PATH: $env:CUDA_PATH")
    }
    foreach ($dir in ($env:PATH -split ';')) {
        if (-not $dir -or -not (Test-Path $dir)) { continue }
        foreach ($pattern in @("cudart64*.dll", "cublas64*.dll", "cudnn64*.dll")) {
            if (Get-ChildItem -LiteralPath $dir -Filter $pattern -File -ErrorAction SilentlyContinue | Select-Object -First 1) {
                $signals.Add("$pattern in $dir")
            }
        }
    }
    return $signals
}

function New-Shortcut {
    param([string]$Path, [string]$Target, [string]$WorkingDirectory)
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($Path)
    $shortcut.TargetPath = $Target
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Save()
}

$source = Split-Path -Parent $MyInvocation.MyCommand.Path
$sevenZip = Join-Path $source "7zr.exe"
if (-not (Test-Path $sevenZip)) {
    throw "Missing 7zr.exe in installer payload."
}

$commonPayload = Join-Path $source "common.7z"
$cpuPayload = Join-Path $source "cpu.7z"
$gpuPayload = Join-Path $source "gpu.7z"
if (-not (Test-Path $commonPayload)) { throw "Missing common.7z in installer payload." }
if (-not (Test-Path $cpuPayload)) { throw "Missing cpu.7z in installer payload." }

$cudaSignals = Test-CudaSignal
$hasGpuPayload = Test-Path $gpuPayload
if ($cudaSignals.Count -gt 0 -and $hasGpuPayload) {
    $message = "CUDA-capable signals were detected:`n`n$($cudaSignals -join "`n")`n`nChoose Yes for GPU runtime, No for CPU runtime, or Cancel to stop."
    $mode = Show-Choice -Title "FSC Studio Setup" -Message $message -DefaultMode "gpu"
} elseif ($hasGpuPayload) {
    $message = "No CUDA runtime signal was detected. CPU runtime is recommended.`n`nChoose Yes for CPU runtime, No to install GPU runtime anyway, or Cancel to stop."
    $mode = Show-Choice -Title "FSC Studio Setup" -Message $message -DefaultMode "cpu"
} else {
    Add-Type -AssemblyName System.Windows.Forms
    $answer = [System.Windows.Forms.MessageBox]::Show(
        "This installer contains only the CPU runtime. Install FSC Studio to $DefaultTarget?",
        "FSC Studio Setup",
        [System.Windows.Forms.MessageBoxButtons]::OKCancel,
        [System.Windows.Forms.MessageBoxIcon]::Information
    )
    if ($answer -ne [System.Windows.Forms.DialogResult]::OK) { throw "Installation cancelled." }
    $mode = "cpu"
}

$target = $DefaultTarget
if (Test-Path $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
}
New-Item -ItemType Directory -Path $target | Out-Null

& $sevenZip x "-o$target" -y $commonPayload | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to extract common runtime." }

$overlay = $(if ($mode -eq "gpu" -and $hasGpuPayload) { $gpuPayload } else { $cpuPayload })
if ($mode -eq "gpu" -and -not $hasGpuPayload) { $mode = "cpu" }
& $sevenZip x "-o$target" -y $overlay | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to extract $mode runtime." }

Set-Content -Path (Join-Path $target "runtime_mode.txt") -Value $mode -Encoding UTF8

$desktop = [Environment]::GetFolderPath("Desktop")
$programs = [Environment]::GetFolderPath("Programs")
$startMenuDir = Join-Path $programs "FSC Studio"
New-Item -ItemType Directory -Path $startMenuDir -Force | Out-Null
New-Shortcut -Path (Join-Path $desktop "FSC Studio.lnk") -Target (Join-Path $target "FSC Studio.cmd") -WorkingDirectory $target
New-Shortcut -Path (Join-Path $startMenuDir "FSC Studio.lnk") -Target (Join-Path $target "FSC Studio.cmd") -WorkingDirectory $target

Start-Process -FilePath (Join-Path $target "FSC Studio.cmd") -WorkingDirectory $target
