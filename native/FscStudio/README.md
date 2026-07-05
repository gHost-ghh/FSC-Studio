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
- Vector search over normalized InsightFace embeddings.
- Identity Gallery identification using existing profile tables.
- CLI probes for database/search/identity and model-path parity checks.

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

The full Qt application is intentionally not the first milestone. The first
milestone is native algorithm correctness.
