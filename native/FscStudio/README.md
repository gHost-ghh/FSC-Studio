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
- CLI probe for database/search/identity parity checks.

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

The full Qt application is intentionally not the first milestone. The first
milestone is native algorithm correctness.
