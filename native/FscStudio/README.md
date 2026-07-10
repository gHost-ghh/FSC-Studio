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
- Dense Mesh tabs render cached `face_mesh3d_json` in native Points or Textured modes, with depth-tested image texture, back-surface darkening, 3D rotation/zoom, an optional 68-point landmark overlay in Textured mode, and local eyelid/iris triangulation so the 468--477 MediaPipe iris points retain their textured eyeballs.
- Dense Mesh uses MediaPipe's native Windows C API and the same `face_landmarker.task` model as the Python application. It stores only validated 478-point meshes, uses the same source-image matching rule, and never synthesizes a mesh from the unrelated 68-point landmark cache.
- Qt runtime deployment is staged by CMake beside `FscStudioQt.exe`: `platforms`, `imageformats`, and `qt.conf` are then copied verbatim into the portable package, together with the matching Qt and JPEG/PNG runtime DLLs.
- Search can use either a stored face id or a standalone analyzed query image with detected-face selection.
- Search now has threshold/min-quality/include-ignored/person/tag controls, an identity candidate table, a throttled progressive result preview while filtered database faces are compared, and native result/identity assignment actions. It ends on the best match rather than cycling completed results.
- Compare analyzes both selected images, lists all detected faces, lets the user choose faces by list or preview click, and compares the selected pair.
- Camera captures native frames, downsizes recognition frames by a configurable process-size limit, overlays recent detected face boxes, and shows per-face smoothed identity/gallery candidates plus similar database hits beside a best-match preview.
- Review shows the selected face preview, runs native identity suggestions, and can confirm the suggested person while retraining profiles.
- Clusters supports max-face/min-quality/unassigned/ignored filters, shows known people and member tags, previews selected members, and can batch-assign a selected cluster.
- Runtime includes current database stats plus native maintenance actions for integrity check, backup, WAL checkpoint, and VACUUM.

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
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --overview-smoke D:\FSC\new_full.fscdb
.\out\build\msvc-vs-qt-debug\Debug\FscStudioQt.exe --compare-smoke D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg D:\FSC\test_img\123s2\baiyh.jpg
```

Use a copied database for `--review-smoke`, because it writes a review state
back to the selected face row.

If vcpkg download through the local proxy fails while installing Qt, place the
failed archive into `D:\FSC\.deps\vcpkg\downloads` and rerun the preset. On this
machine `mity-md4c-release-0.5.3.tar.gz` and
`strawberry-perl-5.42.2.1-64bit-portable.zip` needed that manual cache step.

To create a Python-free portable package from the current Qt build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-qt-portable.ps1
.\out\package\FSC-Studio-Native-Debug\FscStudioQt.exe --smoke D:\FSC\new_full.fscdb
.\out\package\FSC-Studio-Native-Debug\FscStudioQt.exe --compare-smoke .\out\package\FSC-Studio-Native-Debug\models\insightface\models D:\FSC\test_img\123s2\baiyh.jpg D:\FSC\test_img\123s2\baiyh.jpg
```

To create the standard x64 Windows installer, install Inno Setup 6 and run:

```powershell
.\scripts\build-installer.ps1 -AppVersion 0.1.0
```

The command stages the DirectML Camera package, verifies its Qt Windows platform
plugin by starting a no-window UI smoke test, checks ONNX Runtime and MediaPipe
files, then writes `out\installer\FSC-Studio-Setup-x64.exe`. The
installer contains no Python runtime and does not install, copy, or remove a
user database or any original photos.

For the release baseline:

```powershell
cmake --preset msvc-vs-qt-release
cmake --build --preset msvc-vs-qt-release
ctest --preset msvc-vs-qt-release
powershell -ExecutionPolicy Bypass -File .\scripts\package-qt-portable.ps1 -Configuration Release
powershell -ExecutionPolicy Bypass -File .\scripts\package-qt-portable.ps1 -Configuration Release -Zip
```

To build the OpenCV-backed native Camera flavor:

```powershell
cmake --preset msvc-vs-qt-camera-release
cmake --build --preset msvc-vs-qt-camera-release
ctest --preset msvc-vs-qt-camera-release
.\out\build\msvc-vs-qt-camera-release\Release\FscStudioQt.exe --camera-open-smoke 0
powershell -ExecutionPolicy Bypass -File .\scripts\package-qt-portable.ps1 -Configuration Release -Camera -Zip
```

To build the DirectML + Camera flavor:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch-onnxruntime-directml.ps1
cmake --preset msvc-vs-qt-camera-dml-release
cmake --build --preset msvc-vs-qt-camera-dml-release
ctest --preset msvc-vs-qt-camera-dml-release
.\out\build\msvc-vs-qt-camera-dml-release\Release\FscStudioQt.exe --compare-smoke D:\FSC\model\insightface\models D:\FSC\test_img\123s2\baiyh.jpg D:\FSC\test_img\123s2\baiyh.jpg directml
.\out\build\msvc-vs-qt-camera-dml-release\Release\FscStudioQt.exe --camera-result-smoke D:\FSC\model\insightface\models D:\FSC\new_full.fscdb D:\FSC\test_img\123s2\baiyh.jpg directml
powershell -ExecutionPolicy Bypass -File .\scripts\package-qt-portable.ps1 -Configuration Release -Camera -DirectML -Zip
```

The package contains the app, Qt runtime DLLs, Qt platform plugin, ONNX Runtime,
and the local InsightFace model directory. The Camera flavor also includes the
OpenCV runtime DLLs copied beside the executable. The DirectML flavor uses the
DirectML-enabled ONNX Runtime NuGet package. It intentionally does not include
Python, user databases, or personal photos.

The package also includes `Install-FSCStudioNative.ps1` /
`Install-FSCStudioNative.bat` and `Uninstall-FSCStudioNative.ps1`. By default the
installer copies the app to `%LOCALAPPDATA%\Programs\FSC Studio Native` and
creates a Start Menu shortcut. Use `-InstallDir` for a custom install directory.

The full Qt application is intentionally not the first milestone. The first
milestone is native algorithm correctness.
