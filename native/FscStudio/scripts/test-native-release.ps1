[CmdletBinding()]
param(
    [string]$BuildDirectory = "",
    [string]$Database = "",
    [string]$ClusterDatabase = "",
    [string]$ModelRoot = "",
    [string]$ImageA = "",
    [string]$ImageB = "",
    [string]$RuntimeMode = "auto",
    [string]$ExpectedProvider = "",
    [long]$FaceId = 1,
    [long]$SecondFaceId = 2,
    [string]$LegacyDtb = "",
    [switch]$IncludePhysicalCamera,
    [int]$CameraIndex = 0,
    [switch]$SkipPageRenders,
    [string]$OutputRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$nativeRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $nativeRoot "..\.."))
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $nativeRoot "out\build\msvc-vs-qt-camera-dml-release\Release"
}
if (-not $Database) { $Database = Join-Path $repoRoot "new_full.fscdb" }
if (-not $ClusterDatabase) { $ClusterDatabase = $Database }
if (-not $ModelRoot) { $ModelRoot = Join-Path $repoRoot "model\insightface\models\buffalo_l" }
if (-not $ImageA) { $ImageA = Join-Path $repoRoot "test_img\test\baiyh.jpg" }
if (-not $ImageB) { $ImageB = Join-Path $repoRoot "test_img\test\bianyh.jpg" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repoRoot "out\native-release-tests" }

if (-not $ExpectedProvider) {
    $ExpectedProvider = switch ($RuntimeMode.ToLowerInvariant()) {
        "cpu" { "CPUExecutionProvider" }
        "directml" { "DmlExecutionProvider" }
        "cuda" { "CUDAExecutionProvider" }
        "qnn-npu" { "HTP/NPU" }
        "qnn-gpu" { "Adreno GPU" }
        default { "" }
    }
} elseif ($ExpectedProvider.Equals("DirectML", [System.StringComparison]::OrdinalIgnoreCase)) {
    $ExpectedProvider = "DmlExecutionProvider"
}

$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$Database = (Resolve-Path -LiteralPath $Database).Path
$ClusterDatabase = (Resolve-Path -LiteralPath $ClusterDatabase).Path
$ModelRoot = (Resolve-Path -LiteralPath $ModelRoot).Path
$ImageA = (Resolve-Path -LiteralPath $ImageA).Path
$ImageB = (Resolve-Path -LiteralPath $ImageB).Path
if ($LegacyDtb) { $LegacyDtb = (Resolve-Path -LiteralPath $LegacyDtb).Path }
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path (Get-Location).Path $OutputRoot
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

$exe = Join-Path $BuildDirectory "FscStudioQt.exe"
foreach ($required in @($exe, $Database, $ClusterDatabase, $ImageA, $ImageB)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required release-test input does not exist: $required"
    }
}
foreach ($model in @("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx", "1k3d68.onnx")) {
    $modelPath = Join-Path $ModelRoot $model
    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "Required InsightFace model does not exist: $modelPath"
    }
}

$runRoot = Join-Path $OutputRoot (Get-Date -Format "yyyyMMdd-HHmmss")
$databaseRoot = Join-Path $runRoot "databases"
$renderRoot = Join-Path $runRoot "renders"
$logRoot = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force $databaseRoot, $renderRoot, $logRoot | Out-Null

$results = [System.Collections.Generic.List[object]]::new()
$failures = [System.Collections.Generic.List[string]]::new()

function ConvertTo-NativeArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-NativeSmoke {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string[]]$Arguments,
        [int]$TimeoutSeconds = 120,
        [ValidateSet("minimal", "windows")][string]$Platform = "minimal"
    )

    Write-Host ("[{0}] {1}" -f $Name, ($Arguments -join " "))
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $exitCode = -1
    $stdout = ""
    $stderr = ""
    $message = ""
    try {
        $info = [System.Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $exe
        $info.WorkingDirectory = $BuildDirectory
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $nativeArguments = $Arguments | ForEach-Object { ConvertTo-NativeArgument -Value ([string]$_) }
        $info.Arguments = $nativeArguments -join " "
        $info.EnvironmentVariables["FSC_QT_SMOKE_PLATFORM"] = $Platform
        $process = [System.Diagnostics.Process]::Start($info)
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill()
            $process.WaitForExit()
            $message = "timed out after $TimeoutSeconds second(s)"
        } else {
            $exitCode = $process.ExitCode
        }
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
    } catch {
        $message = $_.Exception.Message
    }
    $watch.Stop()

    $ok = $exitCode -eq 0 -and -not $message
    $logPath = Join-Path $logRoot ($Name + ".log")
    @(
        "command=$exe $($Arguments -join ' ')"
        "platform=$Platform"
        "exit_code=$exitCode"
        "duration_ms=$($watch.ElapsedMilliseconds)"
        "message=$message"
        "--- stdout ---"
        $stdout
        "--- stderr ---"
        $stderr
    ) | Set-Content -LiteralPath $logPath -Encoding UTF8
    $results.Add([pscustomobject]@{
        name = $Name
        ok = $ok
        exit_code = $exitCode
        duration_ms = $watch.ElapsedMilliseconds
        log = $logPath
    })
    if (-not $ok) {
        $failures.Add("$Name (exit $exitCode): $message $stderr")
        Write-Warning "$Name failed; see $logPath"
    }
}

function New-TestDatabase([string]$Name, [string]$Source = $Database) {
    $target = Join-Path $databaseRoot ($Name + ".fscdb")
    Copy-Item -LiteralPath $Source -Destination $target -Force
    foreach ($suffix in @("-wal", "-shm")) {
        $sidecar = $Source + $suffix
        if (Test-Path -LiteralPath $sidecar) {
            Copy-Item -LiteralPath $sidecar -Destination ($target + $suffix) -Force
        }
    }
    return $target
}

Invoke-NativeSmoke "database-open" @("--smoke", $Database)
Invoke-NativeSmoke "overview-data" @("--overview-smoke", $Database)
Invoke-NativeSmoke "library-export" @("--library-export-smoke", $Database, (Join-Path $runRoot "faces.csv"))
Invoke-NativeSmoke "clusters-core" @("--cluster-smoke", $ClusterDatabase)
Invoke-NativeSmoke "mesh-cache" @("--mesh-smoke", $Database, "$FaceId")
Invoke-NativeSmoke "mesh-library-visual" @("--library-visual-smoke", $Database, "$FaceId")
Invoke-NativeSmoke "camera-opencv" @("--camera-smoke")

foreach ($language in @("en", "zh", "ja", "ko")) {
    Invoke-NativeSmoke "language-cycle-$language" @("--ui-language-smoke", $language)
    Invoke-NativeSmoke "language-coverage-$language" @("--ui-language-coverage-smoke", $language)
}

Invoke-NativeSmoke "runtime-provider" @("--runtime-probe-ui-smoke", $ModelRoot, $RuntimeMode, $ExpectedProvider) 180
Invoke-NativeSmoke "compare-core" @("--compare-smoke", $ModelRoot, $ImageA, $ImageB, $RuntimeMode) 180
Invoke-NativeSmoke "compare-ui" @("--compare-ui-smoke", $ModelRoot, $ImageA, $ImageB, $RuntimeMode) 180
Invoke-NativeSmoke "search-query-ui" @("--search-query-ui-smoke", $ModelRoot, $ImageA, $RuntimeMode) 180
Invoke-NativeSmoke "camera-static-result" @("--camera-result-smoke", $ModelRoot, $Database, $ImageA, $RuntimeMode) 180
Invoke-NativeSmoke "camera-static-ui" @("--camera-ui-smoke", $Database, $ModelRoot, $ImageA, $RuntimeMode) 180

$importDatabase = Join-Path $databaseRoot "library-import.fscdb"
Invoke-NativeSmoke "library-import-ui" @("--library-import-ui-smoke", $importDatabase, $ModelRoot, $ImageA, $RuntimeMode) 180

$reviewDatabase = New-TestDatabase "review"
Invoke-NativeSmoke "review-metadata" @("--review-smoke", $reviewDatabase, "$FaceId")
$reviewActionDatabase = New-TestDatabase "review-action"
Invoke-NativeSmoke "review-action" @("--review-action-smoke", $reviewActionDatabase, "$FaceId")
$peopleDatabase = New-TestDatabase "people-action"
Invoke-NativeSmoke "people-action" @("--people-action-smoke", $peopleDatabase, "$FaceId")
$clusterActionDatabase = New-TestDatabase "cluster-action" $ClusterDatabase
Invoke-NativeSmoke "cluster-action" @("--cluster-action-smoke", $clusterActionDatabase, "ReleaseClusterPerson") 180
$meshDatabase = New-TestDatabase "mesh-generate"
Invoke-NativeSmoke "mesh-generate" @("--mesh-generate-smoke", $meshDatabase, "$FaceId") 180
$meshUiDatabase = New-TestDatabase "mesh-ui"
Invoke-NativeSmoke "mesh-ui" @("--library-mesh-ui-smoke", $meshUiDatabase, "$FaceId") 180
$metadataDatabase = New-TestDatabase "metadata"
Invoke-NativeSmoke "metadata-action" @("--metadata-smoke", $metadataDatabase, "$FaceId")
$searchActionDatabase = New-TestDatabase "search-action"
Invoke-NativeSmoke "search-action" @("--search-action-smoke", $searchActionDatabase, "$FaceId", "$SecondFaceId", "ReleaseSearchPerson") 180
$searchFilterDatabase = New-TestDatabase "search-filter"
Invoke-NativeSmoke "search-filter" @("--search-filter-smoke", $searchFilterDatabase, "$FaceId")
$cameraActionDatabase = New-TestDatabase "camera-action"
Invoke-NativeSmoke "camera-action" @("--camera-action-smoke", $cameraActionDatabase, "$FaceId", "ReleaseCameraPerson") 180
$peopleTrainingDatabase = New-TestDatabase "people-training"
Invoke-NativeSmoke "people-training-ui" @("--people-training-ui-smoke", $peopleTrainingDatabase) 180
$reviewAiDatabase = New-TestDatabase "review-ai"
Invoke-NativeSmoke "review-ai-ui" @("--review-ai-ui-smoke", $reviewAiDatabase, "$FaceId") 180
$reviewSwitchDatabase = New-TestDatabase "review-switch"
Invoke-NativeSmoke "review-switch-ui" @("--review-suggestion-switch-ui-smoke", $reviewSwitchDatabase, "$FaceId", "$SecondFaceId") 180
Invoke-NativeSmoke "clusters-ui" @("--clusters-ui-smoke", $ClusterDatabase) 180
$clustersAssignDatabase = New-TestDatabase "clusters-assign" $ClusterDatabase
Invoke-NativeSmoke "clusters-assign-ui" @("--clusters-assign-ui-smoke", $clustersAssignDatabase, "ReleaseClusterAssignment") 180

foreach ($action in @("integrity", "checkpoint", "vacuum")) {
    $maintenanceDatabase = New-TestDatabase ("maintenance-" + $action)
    Invoke-NativeSmoke ("runtime-maintenance-" + $action) @("--runtime-maintenance-ui-smoke", $maintenanceDatabase, $action) 180
}
$backupDatabase = New-TestDatabase "maintenance-backup"
Invoke-NativeSmoke "runtime-maintenance-backup" @(
    "--runtime-maintenance-ui-smoke",
    $backupDatabase,
    "backup",
    (Join-Path $databaseRoot "runtime-backup-output.fscdb")
) 180
$maintenanceAllDatabase = New-TestDatabase "maintenance-all"
Invoke-NativeSmoke "maintenance-core" @(
    "--maintenance-smoke",
    $maintenanceAllDatabase,
    (Join-Path $databaseRoot "maintenance-core-backup.fscdb")
) 180

foreach ($view in @(
    @{ name = "front"; yaw = "0"; pitch = "0" },
    @{ name = "side"; yaw = "70"; pitch = "10" },
    @{ name = "back"; yaw = "180"; pitch = "0" }
)) {
    Invoke-NativeSmoke ("mesh-render-" + $view.name) @(
        "--mesh-render-smoke",
        $Database,
        "$FaceId",
        (Join-Path $renderRoot ("mesh-" + $view.name + ".png")),
        $view.yaw,
        $view.pitch
    ) 60 "windows"
}

if (-not $SkipPageRenders) {
    foreach ($size in @(
        @{ width = 1180; height = 760; suffix = "compact" },
        @{ width = 1600; height = 1000; suffix = "large" }
    )) {
        foreach ($page in @("Overview", "Library", "People", "Search", "Camera", "Review", "Clusters", "Compare", "Runtime")) {
            $name = "page-$($page.ToLowerInvariant())-$($size.suffix)"
            Invoke-NativeSmoke $name @(
                "--page-render-smoke",
                $Database,
                $page,
                (Join-Path $renderRoot ($name + ".png")),
                "$($size.width)",
                "$($size.height)",
                "zh"
            ) 60 "windows"
        }
    }
}

if ($LegacyDtb) {
    if (-not (Test-Path -LiteralPath $LegacyDtb)) {
        throw "Legacy DTB does not exist: $LegacyDtb"
    }
    Invoke-NativeSmoke "runtime-legacy-ui" @(
        "--runtime-legacy-ui-smoke",
        $LegacyDtb,
        (Join-Path $databaseRoot "legacy-converted.fscdb"),
        $ModelRoot,
        $RuntimeMode
    ) 600
}

if ($IncludePhysicalCamera) {
    Invoke-NativeSmoke "camera-device-open" @("--camera-open-smoke", "$CameraIndex") 30
    Invoke-NativeSmoke "camera-device-live-ui" @(
        "--camera-live-smoke",
        $Database,
        $ModelRoot,
        "$CameraIndex",
        $RuntimeMode
    ) 30 "windows"
}

$summaryPath = Join-Path $runRoot "summary.json"
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$passed = @($results | Where-Object { $_.ok }).Count
$failed = @($results | Where-Object { -not $_.ok }).Count
Write-Host "Native release tests: $passed passed, $failed failed."
Write-Host "Artifacts: $runRoot"
Write-Host "Summary: $summaryPath"
if ($failed -gt 0) {
    throw ("Native release tests failed:`n" + ($failures -join "`n"))
}
