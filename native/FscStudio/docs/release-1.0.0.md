# FSC Studio Native 1.0.0

Build date: 2026-07-14

This is the first feature-complete native Windows release. It uses C++/Qt,
SQLite, OpenCV, and ONNX Runtime and does not include or require Python.

## License Boundary

FSC Studio source code is Apache-2.0. The bundled InsightFace `buffalo_l`
pretrained weights, including the derived ARM64 quantized copies, are governed
separately by InsightFace's non-commercial research terms. They are not covered
by the FSC Studio source license. See `THIRD-PARTY-NOTICES.txt` in the installed
directory and contact `recognition-oss-pack@insightface.ai` for model licensing.

## Installers

| File | Platform | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `FSC-Studio-Setup-x64.exe` | x64 DirectML/CPU | 327,574,914 | `386BDF94FC56BE329C6C0EC900515B866D070574FB62F150ACE7A143F172A2A2` |
| `FSC-Studio-CUDA-Setup-x64.exe` | x64 NVIDIA CUDA/CPU | 1,285,596,558 | `099D9D84BBFB7C2D0B2C90EC393D03CF8949C867579547971796DBFA48EB70F9` |
| `FSC-Studio-Setup-arm64.exe` | ARM64 Qualcomm QNN NPU/GPU/CPU | 410,866,946 | `A450B21D39CA2E27268815926B16D0CE27E0DC76E065C439ED27D6A3D594B1A5` |

Generated binaries remain under `out/installer` and are intentionally ignored
by Git. Public distribution should Authenticode-sign the final files.

## Compatibility

- Reads and writes the same SQLite `.fscdb` v8 format as Python FSC Studio.
- Native and Python Identity Gallery rebuilds both produced 119 profiles,
  117 weak profiles, and 127 samples on the validation database. Scalar and
  JSON calibration fields matched within `1e-6`.
- Native/Python portrait embedding cosine was `0.99970` to `0.99979`; the
  group-image minimum was `0.99798`. Twelve portrait searches had identical
  Top 1 results and at least 9/10 Top 10 overlap.
- Existing cached 3D landmarks and Dense Mesh data remain compatible. Native
  Dense Mesh generation writes validated 478-point MediaPipe meshes.

## Validation

- C++ CTest: DirectML and CUDA Release builds passed.
- Python identity reference: 3 tests passed.
- Final DirectML package: 66/66 release scenarios passed, including the
  physical camera and 18 page screenshots.
- Final CUDA package: 46/46 release scenarios passed; a separate packaged
  CUDA physical-camera recognition smoke also passed.
- DirectML installer QA: install, file audit, UI launch, provider load, and
  uninstall passed with an isolated AppId.
- ARM64 package: all 26 application PE files are ARM64; no x64 application DLL,
  Python file, user database, or sample photo is present. The HTP models contain
  quantized UINT8/UINT16 weights and the package includes QNN V73/V81 stubs.
  HTP handles the fixed 640 detector and main InsightFace models; a separate
  ARM CPU session handles only the small 128 SCRFD pass to retain dual-scale
  parity without forcing the NPU session to accept a dynamic shape.

Physical HTP/NPU and Adreno execution cannot run on the x64 build machine.
Run `scripts/test-snapdragon-release.ps1` on Surface Pro 11 to complete the
target-hardware acceptance matrix for NPU, GPU, and CPU independently.

## Intentional Native Adaptations

- Compact table columns and horizontal scrolling fix the Python window's
  clipping at 1180x760 and 150% Windows scaling.
- The UI follows the Windows light/dark palette instead of forcing one theme.
- Import, search, profile training, clustering, maintenance, Dense Mesh, and
  camera inference use guarded worker tasks to keep the UI responsive.
