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
- Vector search over normalized InsightFace embeddings.
- Identity Gallery identification using existing profile tables.
- Identity Gallery profile rebuilding on existing databases.
- Native SCRFD detection, ArcFace embeddings, 2D/3D landmarks, and face quality scoring.
- Windows Imaging Component loading for JPEG/PNG/BMP plus existing PPM support.
- CLI probes for database/search/identity/import and model-path parity checks.

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
`qt-app` feature later when the full Qt shell is ready.

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
```

The full Qt application is intentionally not the first milestone. The first
milestone is native algorithm correctness.
