# FSC Studio

FSC Studio is a native Windows face-similarity, identity-learning, and face-database application built with C++/Qt, SQLite, OpenCV, and ONNX Runtime. The original PySide6 implementation remains in the repository as the behavioral reference and research version.

The public repository is intentionally source-only. It does not include private face databases, face photos, model weights, generated installers, virtual environments, or mobile-port prototypes.

## Current Scope

- Multi-face image import into SQLite `.fscdb` databases.
- Face similarity search using normalized InsightFace embeddings.
- Apple Photos-style local identity gallery learning: canonical exemplars, hard negatives, calibrated thresholds, and strict/balanced/broad identity modes.
- One-to-one face comparison.
- Live webcam search against the current database.
- Library browsing, people assignment, tags, review state, ignored records, notes, duplicate detection, and CSV export.
- Similarity clustering for likely same-person groups.
- Runtime/database maintenance tools such as integrity check, backup, checkpoint, and vacuum.

## Repository Layout

```text
fsc_studio.py           Main PySide6 desktop UI
fsc_studio_services.py  Application service layer
fsc_face_engine.py      InsightFace / ONNX Runtime integration
fsc_face_database.py    SQLite .fscdb database layer
pyproject.toml          Python dependency metadata
native/FscStudio/       Native C++/Qt Windows application and installer
```

## Native Windows Install

The native installers include the application, Qt/OpenCV/ONNX Runtime libraries, and required models. End users do not need Python, Visual Studio, the CUDA Toolkit, or a separate model download.

- `FSC-Studio-Setup-x64.exe`: standard Intel/AMD Windows release using DirectML GPU with CPU fallback.
- `FSC-Studio-CUDA-Setup-x64.exe`: NVIDIA release using CUDA with CPU fallback.
- `FSC-Studio-Setup-arm64.exe`: native Snapdragon X release using Qualcomm HTP/NPU, Adreno GPU, and ARM64 CPU.

See the packaged `FSC-Studio-User-Guide.html` or [native/FscStudio/docs/user-guide.zh-CN.html](native/FscStudio/docs/user-guide.zh-CN.html).

## Python Reference Setup

Use Python 3.12 on Windows.

CPU runtime:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[cpu]"
```

GPU runtime with CUDA-capable ONNX Runtime:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[gpu]"
```

Optional research training tools for the local identity scorer:

```powershell
.\.venv312\Scripts\python.exe -m pip install -e ".[gpu,training]"
.\.venv312\Scripts\python.exe tools\train_identity_scorer.py D:\path\to\identity_folders
```

The training dataset folder must contain one subfolder per identity. The script does not download public face datasets; users are responsible for dataset permissions and privacy. If `model/identity_scorer.onnx` exists, FSC Studio will use it for the identity gallery scorer; otherwise it falls back to the built-in NumPy scorer.

## Model Files

FSC Studio uses InsightFace `buffalo_l`. Model weights are not stored in Git.

Place the model files locally under:

```text
model/insightface/models/buffalo_l/
```

Expected files:

```text
1k3d68.onnx
2d106det.onnx
det_10g.onnx
genderage.onnx
w600k_r50.onnx
```

## Run

```powershell
.\.venv312\Scripts\python.exe fsc_studio.py
```

## Release Builds

Native release and validation scripts live under `native/FscStudio/scripts/`. They generate Python-free DirectML, CUDA, and ARM64 QNN portable packages and Inno Setup installers. Build details and reproducible acceptance commands are in [native/FscStudio/README.md](native/FscStudio/README.md).

The older `packaging/` flow is retained only for reproducing the retired Python-bundled release. It is not the current distribution path.

## Data Privacy

Face databases and photos can contain sensitive biometric data. Keep `.fscdb`, `.dtb`, image folders, and generated previews out of public commits.

Identity Gallery data in `.fscdb` v8 stores learned local identity templates derived from face embeddings. Treat these templates as biometric data. Public datasets such as VGGFace2, Glint360K, and InsightFace training sets often have research or non-commercial restrictions; do not redistribute their images, embeddings, or trained scorer outputs unless their licenses allow it.

## License

FSC Studio source code is licensed under the Apache License 2.0. See `LICENSE`.

Third-party packages and model weights keep their own licenses. In particular, InsightFace model weights are not bundled in this repository and must be used under their upstream terms.
