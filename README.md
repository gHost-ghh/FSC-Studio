# FSC Studio

FSC Studio is a desktop face-similarity and face-database application built with PySide6, InsightFace, and ONNX Runtime.

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
native/FscStudio/       In-progress C++/Qt native Windows port
```

## Setup

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

Windows release packaging scripts live in `packaging/`. They build installers with an embedded Python runtime, so end users do not need to install Python.

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 -Variant universal
powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1 -Variant cpu -OutputRoot release\optimized_cpu
```

`universal` includes CPU and GPU ONNX Runtime overlays and asks the user which runtime to install after CUDA/NVIDIA detection. Generated installers are written under `release/`, which is intentionally ignored by Git.

The native Windows migration lives under `native/FscStudio/`. Its current Qt shell can open existing `.fscdb` files, import single images or folders, filter Library records, export the filtered Library table as CSV, and run native Search/Compare/Camera smoke checks without Python.

## Data Privacy

Face databases and photos can contain sensitive biometric data. Keep `.fscdb`, `.dtb`, image folders, and generated previews out of public commits.

Identity Gallery data in `.fscdb` v8 stores learned local identity templates derived from face embeddings. Treat these templates as biometric data. Public datasets such as VGGFace2, Glint360K, and InsightFace training sets often have research or non-commercial restrictions; do not redistribute their images, embeddings, or trained scorer outputs unless their licenses allow it.

## License

FSC Studio source code is licensed under the Apache License 2.0. See `LICENSE`.

Third-party packages and model weights keep their own licenses. In particular, InsightFace model weights are not bundled in this repository and must be used under their upstream terms.
