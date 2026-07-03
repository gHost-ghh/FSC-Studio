# FSC Studio Release Packaging

This packaging flow builds a Windows installer that does not require Python to be installed on the target computer.

The installer contains:

- a Python embeddable runtime
- FSC Studio source files
- the InsightFace model files needed by the app
- a common dependency payload
- a CPU ONNX Runtime overlay
- optionally a GPU ONNX Runtime overlay

The setup executable detects CUDA/NVIDIA signals at install time. If a GPU payload is present, it asks the user whether to install the GPU or CPU runtime. CPU mode writes `runtime_mode.txt=cpu`; GPU mode writes `runtime_mode.txt=gpu`.

Build commands:

```powershell
# Optimized universal setup. Excludes MediaPipe Dense Mesh dependencies by default.
powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 -Variant universal

# Full Dense Mesh build. Larger because MediaPipe pulls extra dependencies.
powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 -Variant universal -IncludeDenseMesh $true

# GPU package that also bundles CUDA/cuDNN pip DLLs. Very large.
powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 -Variant universal -IncludeCudaDlls $true
```

Generated files go under `release/` and are intentionally ignored by Git.
