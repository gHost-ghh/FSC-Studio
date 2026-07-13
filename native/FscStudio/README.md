# FSC Studio Native C++ Restart

This directory is the clean native Windows restart for FSC Studio.

The previous WinUI/C# prototype is preserved on `codex/native-winui-prototype`.
This branch starts from `origin/main` and uses C++/Qt + ONNX Runtime instead of
continuing that prototype.

## Goals

- Keep `.fscdb` v8 compatible with the Python FSC Studio.
- Port the algorithm core before building the full UI.
- Avoid Python runtime in the final Windows application.
- Use Qt 6 Widgets for the desktop UI after the native core reaches parity.

## Current Checkpoint

- CMake/vcpkg project scaffold.
- Native C++ data model.
- SQLite-backed `.fscdb` v8 reader.
- Native `.fscdb` face insertion for analyzed images.
- Native `.fscdb` v8 creation.
- Vector search over normalized InsightFace embeddings.
- Identity Gallery identification using existing profile tables.
- Identity Gallery profile rebuilding on existing databases.
- Minimal native People CRUD for adding people and assigning faces before profile rebuilding.
- Native SCRFD detection, ArcFace embeddings, 2D/3D landmarks, and face quality scoring.
- Windows Imaging Component loading for JPEG/PNG/BMP plus existing PPM support.
- CLI probes for database/search/identity/import and model-path parity checks.
- Qt Widgets desktop shell with the Python page order: Overview, Library, People, Search, Camera, Review, Clusters, Compare, and Runtime. Dense Mesh remains a selected-face visual tab inside Library, alongside Image and 3D Landmarks.
- Library supports selected-face and batch metadata edits for person, tags, review state, ignored state, and notes through native SQLite writes.
- Library image and recursive-folder imports run native ONNX analysis on a worker thread, commit accepted faces in batches of 50, and coalesce preview painting to one update per 50 ms so visual progress does not serialize inference or database writes.
- Library mirrors the Python page structure with a full-width Database/filter band, a horizontally scrollable face table, and a 430--500 px visual/detail column. Hidden tabs no longer force the main window above the requested size, so the page renders without clipping at 1180x760 and wider desktop sizes.
- People mirrors the Python three-column workflow for identity-health summaries, person members, representative/member preview, rename/notes, merge, and assignment clearing. Identity Gallery rebuilding runs on a worker connection and refreshes the cached Camera/Search state without blocking the window.
- Review mirrors the Python queue/detail workflow with fixed table columns, compact focus overlay, metadata actions, and AI person confirmation. Identity suggestions use cached profiles on background workers with generation guards, and confirmation/retraining runs on a separate database connection.
- Dense Mesh tabs render cached `face_mesh3d_json` in native Points or Textured modes, with depth-tested image texture, back-surface darkening, 3D rotation/zoom, an optional 68-point landmark overlay in Textured mode, and local eyelid/iris triangulation so the 468--477 MediaPipe iris points retain their textured eyeballs.
- Dense Mesh generation runs the official MediaPipe 478-point landmark network directly through ONNX Runtime. It uses the selected database face box and five-point eye alignment, stores only validated 478-point meshes, and never substitutes the unrelated 68-point landmark cache.
- Qt runtime deployment is staged by CMake beside `FscStudioQt.exe`: `platforms`, `imageformats`, and `qt.conf` are then copied verbatim into the portable package, together with the matching Qt and JPEG/PNG runtime DLLs.
- Search mirrors the Python query workflow: select an image, detect faces asynchronously, choose by list or preview click, then run filtered similarity and identity search with a throttled progressive preview that ends on the best match.
- Compare detects both selected images asynchronously through a reusable ONNX session, lists every detected face, synchronizes list/preview selection, and compares the selected pair without blocking the UI.
- Camera keeps native capture at a 33 ms UI cadence while recognition runs on a background worker at the configured interval. Database faces and Identity Gallery profiles are immutable snapshots refreshed only when the database changes; recent boxes, smoothed identities, similar hits, and a focusable best-match preview update on the UI thread.
- Review shows the selected face preview, runs native identity suggestions, and can confirm the suggested person while retraining profiles.
- Clusters mirrors the Python parameter band and three-column workflow, supports max-face/min-quality/unassigned/ignored filters, previews selected members with a compact face-focus control, and batch-assigns a selected cluster. Clustering uses OpenCV block matrix multiplication when available, while clustering, transactional assignment, and identity-profile rebuilding run off the UI thread.
- Runtime loads complete ONNX session groups on a worker and reports the actual provider instead of echoing the requested mode. x64 supports CPU, DirectML, and NVIDIA CUDA. Native ARM64 supports CPU plus Qualcomm QNN HTP/NPU and Adreno GPU. Auto falls back only after validating every InsightFace session for a candidate provider.
- Runtime can convert trusted legacy `.dtb` data without Python: a restricted reader extracts the old FSC tuple/NumPy RGB layout, native ONNX re-analyzes the images, and a sibling local preview directory is retained for the converted `.fscdb`. Conversion runs off the UI thread and reports progress through a thread-safe queue.

The final migration requires full UI and workflow parity with `fsc_studio.py`.
Track that explicitly in `docs/python-ui-parity.md`.

## Build

Install prerequisites:

- Visual Studio 2022 Build Tools with MSVC x64.
- CMake 3.28 or newer.
- Ninja.
- vcpkg with manifest mode.

Configure from a Developer PowerShell:

```powershell
cmake --preset msvc-debug
cmake --build --preset msvc-debug
ctest --preset msvc-debug
```

`msvc-debug` is dependency-light and builds the algorithm core first. For the
SQLite probe, configure with vcpkg and `FSC_CORE_ONLY=OFF`. Enable the vcpkg
`qt-app` feature when building the Qt shell.

To enable ONNX Runtime C++ model inspection without Qt/SQLite:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-onnxruntime.ps1 -Version 1.27.0 -Flavor cpu
cmake --preset msvc-vs-debug -DFSC_ENABLE_ONNX=ON -DONNXRUNTIME_ROOT="D:\FSC\.deps\onnxruntime"
cmake --build --preset msvc-vs-debug
.\out\build\msvc-vs-debug\Debug\fsc_vision_probe.exe onnx D:\FSC\model\insightface\models\buffalo_l\det_10g.onnx cpu
```

The build copies `onnxruntime.dll` next to the probe executable so Windows does
not accidentally load an older DLL from `System32`.

For the current SQLite + ONNX probe checkpoint, configure the Visual Studio
debug tree with vcpkg and the downloaded ONNX Runtime:

```powershell
$env:VCPKG_ROOT = "D:\FSC\.deps\vcpkg"
cmake -S . -B out\build\msvc-vs-db-debug -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DFSC_CORE_ONLY=OFF -DFSC_BUILD_QT_APP=OFF -DFSC_ENABLE_ONNX=ON `
  -DONNXRUNTIME_ROOT="D:\FSC\.deps\onnxruntime"
cmake --build out\build\msvc-vs-db-debug --config Debug
ctest --test-dir out\build\msvc-vs-db-debug -C Debug --output-on-failure
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe D:\FSC\new_full.fscdb image-search D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg 5 0.50 strict
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe .\out\probe\native_created.fscdb create-db
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe .\out\probe\native_created.fscdb import-image D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg 0.50
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe .\out\probe\native_created.fscdb add-person NativeTest
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe .\out\probe\native_created.fscdb assign-person 1 1
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe .\out\probe\native_created.fscdb train-profiles 0.35 12
.\out\build\msvc-vs-db-debug\Debug\fsc_native_probe.exe .\out\probe\native_created.fscdb update-review 1 reviewed 0 native-review-smoke
```

To build the current Qt desktop shell:

```powershell
cmake --preset msvc-vs-qt-debug
cmake --build --preset msvc-vs-qt-debug
ctest --preset msvc-vs-qt-debug
$p = Start-Process -FilePath .\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe `
  -ArgumentList @("--smoke", "D:\FSC\new_full.fscdb") -Wait -PassThru
$p.ExitCode
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --review-smoke D:\FSC\native\FscStudio\out\probe\native_review_qt.fscdb 1
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --cluster-smoke D:\FSC\new_full.fscdb
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --mesh-smoke D:\FSC\new_full.fscdb 1
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --mesh-render-smoke D:\FSC\new_full.fscdb 1 D:\FSC\native\FscStudio\out\probe\mesh_render.png
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --mesh-render-smoke D:\FSC\new_full.fscdb 1 D:\FSC\native\FscStudio\out\probe\mesh_render_side.png 1.1 -0.15
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --ui-language-smoke zh
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --library-import-ui-smoke D:\FSC\native\FscStudio\out\probe\library-import-smoke.fscdb D:\FSC\model\insightface\models D:\FSC\test_img\test\baiyh.jpg cpu
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --library-mesh-ui-smoke D:\FSC\native\FscStudio\out\probe\library-mesh-smoke.fscdb 1
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --people-training-ui-smoke D:\FSC\native\FscStudio\out\probe\people-training-smoke.fscdb
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --review-ai-ui-smoke D:\FSC\native\FscStudio\out\probe\review-ai-smoke.fscdb 1
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --review-suggestion-switch-ui-smoke D:\FSC\native\FscStudio\out\probe\review-switch-smoke.fscdb 1 2
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --clusters-ui-smoke D:\FSC\new_full.fscdb
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --clusters-assign-ui-smoke D:\FSC\native\FscStudio\out\probe\clusters-assign-smoke.fscdb NativeClusterUiSmoke
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --runtime-probe-ui-smoke D:\FSC\model\insightface\models auto
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --runtime-maintenance-ui-smoke D:\FSC\native\FscStudio\out\probe\runtime-maintenance-smoke.fscdb integrity
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --runtime-maintenance-ui-smoke D:\FSC\native\FscStudio\out\probe\runtime-maintenance-smoke.fscdb backup D:\FSC\native\FscStudio\out\probe\runtime-backup-smoke.fscdb
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --runtime-legacy-ui-smoke D:\path\legacy.dtb D:\path\converted.fscdb D:\FSC\model\insightface\models cpu
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --page-render-smoke D:\FSC\new_full.fscdb Library D:\FSC\native\FscStudio\out\probe\library.png 1180 760
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --overview-smoke D:\FSC\new_full.fscdb
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --compare-smoke D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg D:\FSC\test_img\123s2\baiyh.jpg
```

Use a copied database for `--review-smoke` and `--clusters-assign-ui-smoke`,
because they write review/identity state. Use a copied database for Runtime
checkpoint/VACUUM maintenance smokes. Use a copied database for Library
`--page-render-smoke` runs as well, because selecting the first row may populate
a missing Dense Mesh cache.

All `*-smoke` commands automatically use the deployed `qminimal` platform
plugin and stay off-screen. Installer creation opts into one explicit
`qwindows` smoke through `FSC_QT_SMOKE_PLATFORM=windows`; ordinary application
launches continue to use the Windows desktop platform plugin.

If vcpkg download through the local proxy fails while installing Qt, place the
failed archive into `D:\FSC\.deps\vcpkg\downloads` and rerun the preset. On this
machine `mity-md4c-release-0.5.3.tar.gz` and
`strawberry-perl-5.42.2.1-64bit-portable.zip` needed that manual cache step.

## Release Builds

Prepare or rebuild the standalone Dense Mesh model from the official MediaPipe
task asset with:

```powershell
.\scripts\prepare-face-mesh-onnx.ps1 -Bootstrap
```

The conversion Python environment is build-time tooling only. Python and the
MediaPipe task archive are not included in any release package.

Build the standard x64 DirectML release:

```powershell
.\scripts\fetch-onnxruntime-directml.ps1
cmake --preset msvc-vs-qt-camera-dml-release
cmake --build --preset msvc-vs-qt-camera-dml-release
ctest --preset msvc-vs-qt-camera-dml-release
.\scripts\build-installer.ps1 -Architecture x64 -Accelerator directml -AppVersion 0.2.0
```

Build the NVIDIA CUDA x64 release. The redistributable staging script collects
the tested CUDA 13/cuDNN 9 runtime subset, so the target PC needs a compatible
NVIDIA driver but does not need the CUDA Toolkit:

```powershell
.\scripts\fetch-onnxruntime.ps1 -Version 1.27.0 -Flavor cuda
.\scripts\fetch-cuda13-runtime.ps1
cmake --preset msvc-vs-qt-camera-cuda-release
cmake --build --preset msvc-vs-qt-camera-cuda-release
ctest --preset msvc-vs-qt-camera-cuda-release
.\scripts\build-installer.ps1 -Architecture x64 -Accelerator cuda -AppVersion 0.2.0
```

Build the native ARM64 Snapdragon release. Quantized InsightFace models are
used only for QNN HTP/NPU; the original float models remain available for CPU
and Adreno GPU fallback. Dense Mesh is sent to Adreno GPU or CPU because its
float display model is not an appropriate NPU workload.

```powershell
.\scripts\fetch-qt-arm64.ps1
.\scripts\fetch-onnxruntime-qnn.ps1
.\scripts\quantize-insightface-qnn.ps1
.\scripts\build-arm64-qnn.ps1 -Configuration Release
.\scripts\build-installer.ps1 -Architecture arm64 -Accelerator qnn -AppVersion 0.2.0 -SkipPackageSmoke
```

The three installers are written to:

- `out\installer\FSC-Studio-Setup-x64.exe`
- `out\installer\FSC-Studio-CUDA-Setup-x64.exe`
- `out\installer\FSC-Studio-Setup-arm64.exe`

Each packaging run validates executable architecture, required provider DLLs,
Qt plugins, models, VC++ redistributable, and a Windows-platform UI launch on
x64. The package manifest records version, architecture, accelerator, runtime,
and hashes. Releases intentionally exclude Python, user databases, personal
photos, source files, and intermediate build artifacts.

Legacy `.dtb` conversion is available from **Runtime > Convert Legacy DTB** or
by choosing a `.dtb` file from a native database-open control. The converter
never executes pickle globals; it accepts only the old FSC row shape and NumPy
RGB payload, then writes `<output>_legacy_images` beside the new `.fscdb` so
converted faces remain previewable without the original files.

The Inno Setup executable does not use MSIX package certificates, avoiding the
test-certificate trust failures of the retired WinUI prototype. Public releases
should still be Authenticode signed to reduce SmartScreen warnings.
